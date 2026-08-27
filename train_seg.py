"""Ridgepath segmentation training and evaluation (PyTorch).

Single-GPU by default; optional DDP when launched via torchrun (auto-detected from env, never
required). Supports config YAML + CLI overrides, AdamW, cosine/step/constant LR with warmup,
AMP (CUDA only), checkpoint save/resume, a smoke/max-iters short-run mode, eval (mean loss),
and optional debug-viz dumps. Decode/instance metrics are deferred (Phase 6); eval = mean loss.

Examples:
  # smoke (few iters, CPU or 1 GPU) -- sanity only
  python train_seg.py --config configs/legacy/r18_scratch.yaml --smoke
  # single GPU
  python train_seg.py --config configs/convnext_dino/convnext_dino_hires_unfrozen.yaml
  # optional DDP (4 GPUs)
  torchrun --nproc_per_node=4 train_seg.py \
    --config configs/convnext_dino/convnext_dino_hires_unfrozen.yaml --ddp
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from data.seg_dataset import RidgepathSegDataset, seg_worker_init_fn
from losses.ridgepath_loss import ridgepath_loss
from models.encoder_decoder import build_seg_model
from restore import (restore_dino_into_resnet18, restore_dino_into_tenxnet, restore_seg_encoder,
                     restore_seg_full)

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = Path(os.environ.get("PYTORCH_SEG_ARTIFACT_ROOT", REPO_ROOT)).expanduser().resolve()
CONFIG_PATH_KEYS = ("manifest", "val_manifest", "seg_checkpoint", "dino_checkpoint", "out_dir")

DEFAULTS = dict(
    encoder="resnet18", in_chans=2, num_classes=9,
    encoder_norm="bn", gn_max_groups=32,  # 'gn' -> GroupNorm encoder (for iBOT/DINOv2 GN checkpoints)
    manifest="cache/manifest.csv",
    val_manifest=None, augment=True,
    target_params=dict(dist_cutoff=15.0, weight_sigma=1.0, smooth_range=3, weighted_loss=True),
    aug=None,  # None -> Dataset uses tenxnet train-aug defaults (DEFAULT_AUG); or override per-config
    seg_channels=["boundary", "DAPI"],
    init="scratch", dino_checkpoint="/mnt/home/ruizhi.yuan/ssl_dino/out_full/checkpoint.pth",
    dino_which="student", ssl_channels=["DAPI", "boundary", "18S", "avim"],
    seg_checkpoint=None,  # init=seg_encoder: source pytorch_seg ckpt whose ENCODER is loaded (decoder/heads discarded)
    freeze_encoder_eval=False,  # keep encoder in .eval() during training (freezes SD/dropout + BN running stats)
    model_kwargs=dict(),  # extra kwargs forwarded to build_seg_model (e.g. convnext: pretrained/timm_model/hires_c0/c1)
    optimizer="adamw", lr=1e-3, encoder_lr=None, weight_decay=1e-4, betas=[0.9, 0.999],
    lr_schedule="cosine", warmup_epochs=0, min_lr=1e-6,
    lr_milestones=None, lr_values=None, lr_milestone_unit="steps",  # multistep: boundaries in "steps" or "epochs" + absolute LR per segment
    freeze_encoder_epochs=0,  # LP-FT: hold encoder lr=0 for the first N epochs, then unfreeze (protects SSL features)
    encoder_lr_follows_schedule=True,  # False -> encoder holds a constant encoder_lr (NOT scaled by the decoder's schedule factor)
    grad_clip=1.0,  # clip grad-norm to this (None/<=0 disables); watch clip_frac in the log to tune it
    epochs=100, batch_size=8, num_workers=4, amp=True,
    amp_dtype="fp16",  # AMP compute dtype: "fp16" (broad HW, needs GradScaler) or "bf16" (Ampere/Ada+, no scaler, robust for fine-tuning)
    out_dir="runs/run", seed=0, save_every=10, log_every=10,
)


# ----------------------------------------------------------------- distributed helpers
def setup_dist(use_ddp):
    if use_ddp and "RANK" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, dist.get_rank(), dist.get_world_size(), local_rank
    return False, 0, 1, 0


def is_main(rank):
    return rank == 0


def log_json(out_dir, record):
    """Append one JSON line to out_dir/log.txt (call on main rank only)."""
    with open(os.path.join(out_dir, "log.txt"), "a") as fh:
        fh.write(json.dumps(record) + "\n")


# ----------------------------------------------------------------- lr schedule
def lr_factor(step, total_steps, warmup_steps, schedule, min_ratio, milestones=None, factors=None):
    if warmup_steps and step < warmup_steps:
        return step / max(1, warmup_steps)
    if schedule == "constant":
        return 1.0
    if schedule == "multistep":
        # PiecewiseConstantDecay on ABSOLUTE optimizer steps (== tenxnet stepwise). `milestones` are the
        # step boundaries; `factors` (len = len(milestones)+1) are LR multipliers per segment relative to
        # base lr. e.g. milestones [12000,18000], factors [1.0,0.2,0.1] with base lr 0.01 -> 0.01/0.002/0.001.
        seg = sum(1 for m in (milestones or []) if step >= m)
        return factors[seg]
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    if schedule == "cosine":
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    if schedule == "step":
        return 1.0 if progress < 0.5 else (0.1 if progress < 0.75 else 0.01)
    raise ValueError(f"unknown lr_schedule {schedule}")


# ----------------------------------------------------------------- config
def load_config(path, overrides):
    cfg = dict(DEFAULTS)
    if path:
        with open(path) as fh:
            cfg.update(yaml.safe_load(fh) or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    for key in CONFIG_PATH_KEYS:
        value = cfg.get(key)
        if not isinstance(value, str) or not value:
            continue
        value = os.path.expandvars(os.path.expanduser(value))
        if not os.path.isabs(value):
            value = str(ARTIFACT_ROOT / value)
        cfg[key] = value
    return cfg


def build_loader(manifest, cfg, ddp, shuffle, augment):
    ds = RidgepathSegDataset(manifest, augment=augment, params=cfg["target_params"],
                             seed=cfg["seed"], aug=cfg.get("aug"))
    sampler = DistributedSampler(ds, shuffle=shuffle) if ddp else None
    loader = DataLoader(
        ds, batch_size=cfg["batch_size"], shuffle=(shuffle and sampler is None),
        sampler=sampler, num_workers=cfg["num_workers"],
        worker_init_fn=seg_worker_init_fn if cfg["num_workers"] else None,
        persistent_workers=False,  # re-fork each epoch so worker aug seeds vary per epoch
        prefetch_factor=(4 if cfg["num_workers"] else None), pin_memory=torch.cuda.is_available(),
        drop_last=ddp,
    )
    return ds, loader, sampler


# ----------------------------------------------------------------- checkpoint
def save_ckpt(path, model, opt, scaler, epoch, cfg):
    raw = model.module if isinstance(model, DDP) else model
    torch.save(dict(model=raw.state_dict(), optimizer=opt.state_dict(),
                    scaler=(scaler.state_dict() if scaler else None), epoch=epoch, config=cfg), path)


def load_ckpt(path, model, opt, scaler, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    raw = model.module if isinstance(model, DDP) else model
    raw.load_state_dict(ck["model"])
    if opt and ck.get("optimizer"):
        opt.load_state_dict(ck["optimizer"])
    if scaler and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    return ck.get("epoch", 0)


# ----------------------------------------------------------------- eval / viz
@torch.no_grad()
def evaluate(model, loader, device):
    """Mean val loss + per-term breakdown + FOREGROUND direction-argmax accuracy. Returns a dict.

    The direction terms are what the instance decode relies on, and the loss curve hides whether the
    row/col heads actually learn (they can collapse to the background bin while the weighted soft-CE
    still drops). So we report, over foreground pixels (target semantic > 0.5):
      dir_row_acc / dir_col_acc  -- fraction where the model's row/col argmax == the GT argmax,
      fg_pred_bg_frac            -- fraction predicted as the background direction bin (index 3);
                                    ~1.0 == collapsed, ~0.0 == firing real directions.
    """
    model.eval()
    s = {"loss": 0.0, "seg": 0.0, "row": 0.0, "col": 0.0}
    n = 0
    n_fg = 0
    row_ok = col_ok = fg_bg = 0
    for img, tgt in loader:
        img, tgt = img.to(device), tgt.to(device)
        logits = model(img)
        total, seg, row, col = ridgepath_loss(logits, tgt)
        b = img.shape[0]
        s["loss"] += total.item() * b; s["seg"] += seg.item() * b
        s["row"] += row.item() * b; s["col"] += col.item() * b
        n += b
        # foreground direction-argmax accuracy (bin 3 = background class; GT on fg is always a real dir)
        fg = tgt[:, 0] > 0.5                                    # [B,H,W]
        m = int(fg.sum().item())
        if m:
            pr = logits[:, 1:5].argmax(1)[fg]; gr = tgt[:, 1:5].argmax(1)[fg]
            pc = logits[:, 5:9].argmax(1)[fg]; gc = tgt[:, 5:9].argmax(1)[fg]
            row_ok += int((pr == gr).sum().item())
            col_ok += int((pc == gc).sum().item())
            fg_bg += int((pr == 3).sum().item())
            n_fg += m
    out = {k: v / max(1, n) for k, v in s.items()}
    out["n_val"] = n
    out["dir_row_acc"] = row_ok / max(1, n_fg)
    out["dir_col_acc"] = col_ok / max(1, n_fg)
    out["fg_pred_bg_frac"] = fg_bg / max(1, n_fg)
    return out


@torch.no_grad()
def save_viz(model, ds, device, out_dir, n=2):
    """Best-effort debug viz: image ch0, predicted semantic prob, target semantic."""
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    for i in range(min(n, len(ds))):
        img, tgt = ds[i]
        logits = model(img.unsqueeze(0).to(device))[0].cpu()
        pred_sem = torch.sigmoid(logits[0]).numpy()
        rec = dict(image_ch0=img[0].numpy(), pred_semantic=pred_sem, target_semantic=tgt[0].numpy())
        np.savez(os.path.join(out_dir, f"viz_{i}.npz"), **rec)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(9, 3))
            ax[0].imshow(rec["image_ch0"], cmap="gray"); ax[0].set_title("image ch0")
            ax[1].imshow(rec["pred_semantic"], cmap="magma", vmin=0, vmax=1); ax[1].set_title("pred sem")
            ax[2].imshow(rec["target_semantic"], cmap="magma", vmin=0, vmax=1); ax[2].set_title("target sem")
            for a in ax:
                a.axis("off")
            fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"viz_{i}.png"), dpi=110)
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            print(f"[viz] PNG skipped ({e}); saved npz")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ddp", action="store_true", help="enable DDP (also needs torchrun env)")
    ap.add_argument("--smoke", action="store_true", help="few-iter sanity run")
    ap.add_argument("--max-iters", type=int, default=None, help="stop after N optimizer steps")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-viz", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    overrides = dict(epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers,
                     out_dir=args.out_dir, manifest=args.manifest)
    cfg = load_config(args.config, overrides)
    if args.no_amp:
        cfg["amp"] = False
    if args.smoke:
        cfg["num_workers"] = min(cfg["num_workers"], 2)
        if args.batch_size is None:
            cfg["batch_size"] = 2
        if args.max_iters is None:
            args.max_iters = 20
        # SAFETY: a smoke run must never write into a real run's out_dir (its 1-epoch best.pth would
        # clobber a fully-trained checkpoint). Unless the user explicitly redirected --out-dir, sandbox
        # smoke output under a sibling "<out_dir>_smoke" directory.
        if args.out_dir is None:
            cfg["out_dir"] = cfg["out_dir"].rstrip("/") + "_smoke"

    ddp, rank, world, local_rank = setup_dist(args.ddp)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if ddp and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(cfg["seed"] + rank)

    if is_main(rank):
        os.makedirs(cfg["out_dir"], exist_ok=True)
        with open(os.path.join(cfg["out_dir"], "config.json"), "w") as fh:
            json.dump(cfg, fh, indent=2)  # snapshot the resolved config for this run
        print(f"device={device} ddp={ddp} world={world} | encoder={cfg['encoder']} "
              f"init={cfg['init']} amp={cfg['amp']}({cfg['amp_dtype'] if cfg['amp'] else 'off'})")

    # model
    model = build_seg_model(encoder_name=cfg["encoder"], in_chans=cfg["in_chans"],
                            num_classes=cfg["num_classes"], encoder_norm=cfg["encoder_norm"],
                            gn_max_groups=cfg["gn_max_groups"], **cfg["model_kwargs"]).to(device)
    if cfg["init"] == "dino":
        if is_main(rank):
            if cfg["encoder"] == "resnet18":
                restore_dino_into_resnet18(model, cfg["dino_checkpoint"], cfg["seg_channels"],
                                           cfg["ssl_channels"], which=cfg["dino_which"])
            elif cfg["encoder"] in ("tenxnet_recipe", "tenxnet_small"):
                restore_dino_into_tenxnet(model, cfg["dino_checkpoint"], cfg["seg_channels"],
                                          cfg["ssl_channels"], which=cfg["dino_which"])
            else:
                raise ValueError(
                    "init=dino requires encoder in {resnet18, tenxnet_recipe, tenxnet_small}")
        if ddp:
            dist.barrier()
    elif cfg["init"] == "seg_encoder":
        # Frozen-scratch-encoder control: load ONLY the encoder from a train-from-scratch seg ckpt;
        # decoder + heads stay at random init. Source encoder must match this config's arch+norm.
        if not cfg["seg_checkpoint"]:
            raise ValueError("init=seg_encoder requires 'seg_checkpoint' in the config")
        if is_main(rank):
            restore_seg_encoder(model, cfg["seg_checkpoint"])
        if ddp:
            dist.barrier()   # DDP wrap below broadcasts rank-0 weights to all replicas
    elif cfg["init"] == "seg_full":
        # LP-FT warm-start: load the WHOLE model (encoder + trained decoder + heads) from a completed
        # frozen-encoder run, then continue with a FRESH optimizer/schedule (small encoder_lr set below).
        # Unlike --resume, this keeps the optimizer pristine so the encoder does NOT inherit the probe's
        # hot base_lr. Build with model_kwargs.pretrained=false (weights come from the checkpoint).
        if not cfg["seg_checkpoint"]:
            raise ValueError("init=seg_full requires 'seg_checkpoint' in the config")
        if is_main(rank):
            restore_seg_full(model, cfg["seg_checkpoint"])
        if ddp:
            dist.barrier()   # DDP wrap below broadcasts rank-0 weights to all replicas
    elif cfg["init"] == "timm_convnext":
        pass  # weights are loaded by timm at build (build_seg_model / ConvNeXtDinoV3Features); no restore
    # LP-FT: TRULY freeze the encoder (requires_grad=False -> no encoder backward, no grads in the clip
    # norm, pristine optimizer state) for the first N epochs. Set BEFORE the DDP wrap so the reducer
    # excludes the encoder; we re-wrap DDP at unfreeze so the encoder's grads sync across ranks.
    enc_frozen = bool(cfg["freeze_encoder_epochs"])
    if enc_frozen:
        model.encoder.requires_grad_(False)
        if is_main(rank):
            print(f"[freeze] encoder requires_grad=False for epochs 0..{cfg['freeze_encoder_epochs']-1} "
                  f"(re-wraps DDP at unfreeze so grads sync)")
    if ddp:
        model = DDP(model, device_ids=([local_rank] if torch.cuda.is_available() else None))

    # optim + sched + amp
    # Two param groups so the encoder can take a different (typically lower) LR when fine-tuning a
    # pretrained encoder. `encoder_lr=None` (or == lr) makes both groups equal -> identical to a
    # single-LR run (train-from-scratch). Each group carries a `base_lr`; the schedule scales every
    # group's base by the same warmup/cosine factor. Group 0 = decoder (FPN+head), so the existing
    # `param_groups[0]` logging keeps reporting the base/decoder LR.
    base_lr = cfg["lr"]
    enc_lr = base_lr if cfg["encoder_lr"] is None else cfg["encoder_lr"]
    core = model.module if isinstance(model, DDP) else model
    enc_params = list(core.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    dec_params = [p for p in core.parameters() if id(p) not in enc_ids]
    param_groups = [
        {"params": dec_params, "lr": base_lr, "base_lr": base_lr},                    # FPN + head
        {"params": enc_params, "lr": enc_lr, "base_lr": enc_lr, "is_encoder": True},  # encoder
    ]
    opt = torch.optim.AdamW(param_groups, lr=base_lr, betas=tuple(cfg["betas"]),
                            weight_decay=cfg["weight_decay"])
    if is_main(rank):
        frz = cfg["freeze_encoder_epochs"]
        print(f"[optim] decoder lr={base_lr:.2e} | encoder lr={enc_lr:.2e} "
              f"({len(dec_params)} decoder + {len(enc_params)} encoder tensors)"
              + (f" | encoder FROZEN (lr=0) for first {frz} epochs, then unfreeze" if frz else ""))
    # AMP dtype is config-selectable (amp_dtype: "fp16" | "bf16").
    #   fp16: broadest HW support (incl. T4/Turing fp16 tensor cores) but a NARROW exponent range -> can
    #         overflow, so it needs the GradScaler (rescues non-finite GRADS by skipping the step + backing
    #         off the scale) and is prone to NaN once the deep trunk unfreezes.
    #   bf16: fp32-range exponent -> effectively no overflow, so NO GradScaler and far more robust for
    #         fine-tuning; but needs Ampere/Ada+ to be fast (bf16 is unaccelerated on T4/Turing).
    # A non-finite LOSS (forward overflow) is still handled in the loop (skip the batch). amp=false -> fp32.
    amp_on = cfg["amp"] and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(cfg["amp_dtype"]).lower() in ("bf16", "bfloat16") else torch.float16
    # GradScaler ONLY for fp16 (bf16 has no overflow to rescale; a bf16 scaler would be incorrect).
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_on and amp_dtype == torch.float16))

    _, loader, sampler = build_loader(cfg["manifest"], cfg, ddp, shuffle=True, augment=cfg["augment"])
    steps_per_epoch = max(1, len(loader))
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = steps_per_epoch * cfg["warmup_epochs"]

    ms_milestones, ms_factors = None, None
    if cfg["lr_schedule"] == "multistep":
        raw_ms = list(cfg["lr_milestones"] or [])
        unit = cfg["lr_milestone_unit"]
        if unit == "epochs":                              # convert epoch boundaries -> absolute step boundaries
            ms_milestones = [int(round(m * steps_per_epoch)) for m in raw_ms]
        elif unit == "steps":
            ms_milestones = [int(m) for m in raw_ms]
        else:
            raise ValueError(f"lr_milestone_unit must be 'steps' or 'epochs', got {unit!r}")
        vals = list(cfg["lr_values"] or [cfg["lr"]])
        if len(vals) != len(ms_milestones) + 1:
            raise ValueError(f"lr_values (len {len(vals)}) must be len(lr_milestones)+1 "
                             f"({len(ms_milestones)}+1) for multistep")
        ms_factors = [v / cfg["lr"] for v in vals]     # absolute LR values -> multipliers on base lr
        if is_main(rank):
            print(f"[lr] multistep: milestones={raw_ms} {unit} -> steps {ms_milestones} "
                  f"values={vals} (steps/epoch={steps_per_epoch}, factors={ms_factors})")

    # per-epoch validation (main rank only; un-sharded full val set) + best-checkpoint tracking.
    # We keep just two checkpoints: best.pth (lowest val loss, overwritten on improvement) and
    # checkpoint.pth (latest, for resume) -- not one file per epoch.
    vloader = None
    if is_main(rank) and cfg["val_manifest"]:
        # num_workers=0: eval on the small val set is cheap single-process and avoids re-spawning workers
        # every epoch. (This does NOT by itself protect the train RNG -- a DataLoader iterator draws a
        # base_seed from the global torch RNG at *any* worker count; that is handled by save/restore
        # around evaluate() below, which keeps the training augmentation stream identical to a no-val run.)
        _, vloader, _ = build_loader(cfg["val_manifest"], {**cfg, "num_workers": 0},
                                     False, shuffle=False, augment=False)
    eval_model = model.module if isinstance(model, DDP) else model  # eval on the raw module (no DDP collective)
    val_history = []                            # per-epoch {epoch, loss, seg, row, col, n_val} (main rank)
    best = {"loss": float("inf"), "epoch": -1}

    start_epoch = 0
    if args.resume:
        start_epoch = load_ckpt(args.resume, model, opt, scaler, device)
        if is_main(rank):
            print(f"resumed from {args.resume} @ epoch {start_epoch}")

    gstep = start_epoch * steps_per_epoch
    stop = False
    for epoch in range(start_epoch, cfg["epochs"]):
        if enc_frozen and epoch >= cfg["freeze_encoder_epochs"]:   # unfreeze (also fires on resume past the window)
            core.encoder.requires_grad_(True)
            if ddp:
                dist.barrier()   # re-wrap so the reducer now includes the encoder -> its grads all-reduce
                model = DDP(core, device_ids=([local_rank] if torch.cuda.is_available() else None))
            eval_model = model.module if isinstance(model, DDP) else model
            enc_frozen = False
            if is_main(rank):
                print(f"[freeze] epoch {epoch}: UNFROZE encoder + re-wrapped DDP (encoder grads now sync)")
        model.train()
        if cfg["freeze_encoder_eval"]:
            # keep the (frozen) encoder in eval mode so stochastic-depth/dropout are off and BN uses
            # its frozen running stats -- a truly fixed feature extractor. Re-applied every epoch
            # because model.train() above flips the whole tree back to train mode.
            core.encoder.eval()
        if sampler:
            sampler.set_epoch(epoch)
        # vary augmentation across epochs: workers reseed via seg_worker_init_fn (torch base_seed
        # advances per epoch); the num_workers=0 path reseeds numpy here.
        if cfg["num_workers"] == 0:
            np.random.seed(cfg["seed"] + epoch)
        ep = {"loss": 0.0, "seg": 0.0, "row": 0.0, "col": 0.0, "n": 0,
              "gnorm": 0.0, "nclip": 0, "nskip": 0}  # train accumulators (+ grad-norm/clip/skipped-batch counts)
        for it, (img, tgt) in enumerate(loader):
            img, tgt = img.to(device), tgt.to(device)
            factor = lr_factor(gstep, total_steps, warmup_steps,
                               cfg["lr_schedule"], cfg["min_lr"] / cfg["lr"],
                               milestones=ms_milestones, factors=ms_factors)
            frozen_enc = epoch < cfg["freeze_encoder_epochs"]   # LP-FT: encoder held at lr=0 early
            enc_follows = cfg["encoder_lr_follows_schedule"]
            for g in opt.param_groups:
                if g.get("is_encoder"):
                    # frozen -> 0; else follow the schedule factor, OR hold a constant encoder_lr (decoupled)
                    g["lr"] = 0.0 if frozen_enc else (g["base_lr"] * factor if enc_follows else g["base_lr"])
                else:
                    g["lr"] = g["base_lr"] * factor   # decoder/head always follows the schedule
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_on):
                logits = model(img)
                loss, seg, row, col = ridgepath_loss(logits, tgt)
            # fp16 forward overflow (esp. once the deep SSL encoder is unfrozen): an inf activation ->
            # NaN loss. Rescaling can't fix a forward NaN, so SKIP this batch instead of crashing --
            # params stay finite and training continues. Occasional skips are harmless; a persistent
            # stream of them means lower the LR / add encoder-LR warmup. (Non-finite GRADS from backward
            # overflow are handled below by the GradScaler: it skips the step and backs off the scale.)
            bad = torch.tensor(0.0 if torch.isfinite(loss) else 1.0, device=device)
            if ddp:
                dist.all_reduce(bad, op=dist.ReduceOp.MAX)   # all ranks skip together (no collective hang)
            if bad.item() > 0:
                opt.zero_grad(set_to_none=True)
                if is_main(rank):
                    ep["nskip"] += 1
                    if gstep % cfg["log_every"] == 0:
                        print(f"epoch {epoch} it {it} gstep {gstep} | NON-FINITE loss -> skipped batch")
                gstep += 1
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            gnorm = None
            if cfg["grad_clip"] and cfg["grad_clip"] > 0:
                scaler.unscale_(opt)                                   # AMP: unscale BEFORE clipping (no-op if amp off)
                gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]))
            scaler.step(opt)                                           # DDP grads already all-reduced -> all ranks clip identically
            scaler.update()
            if is_main(rank):
                ep["loss"] += loss.item(); ep["seg"] += seg.item()
                ep["row"] += row.item(); ep["col"] += col.item(); ep["n"] += 1
                if gnorm is not None:
                    ep["gnorm"] += gnorm; ep["nclip"] += int(gnorm > cfg["grad_clip"])
                if gstep % cfg["log_every"] == 0:
                    lr_dec = opt.param_groups[0]["lr"]; lr_enc = opt.param_groups[1]["lr"]
                    enc_str = f"/enc {lr_enc:.2e}" if lr_enc != lr_dec else ""
                    gstr = f" | gnorm {gnorm:.2f}{'*' if gnorm > cfg['grad_clip'] else ''}" if gnorm is not None else ""
                    print(f"epoch {epoch} it {it} gstep {gstep} | loss {loss.item():.4f} "
                          f"(seg {seg.item():.4f} row {row.item():.4f} col {col.item():.4f}) "
                          f"lr {lr_dec:.2e}{enc_str}{gstr}")
            gstep += 1
            if args.max_iters is not None and gstep >= args.max_iters:
                stop = True
                break
        # per-epoch validation + best-checkpoint (main rank only; vloader is None elsewhere).
        # Save/restore the global RNG around eval: building the val iterator draws a DataLoader base_seed
        # from the global torch RNG, which would otherwise shift the TRAIN workers' per-epoch aug seeds
        # and perturb the training trajectory. Restoring leaves the train RNG stream byte-identical.
        val = None
        if vloader is not None:
            cpu_rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            val = evaluate(eval_model, vloader, device)
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            val_history.append({"epoch": epoch, **val})
            if val["loss"] < best["loss"]:
                best = {"epoch": epoch, **val}
                save_ckpt(os.path.join(cfg["out_dir"], "best.pth"), model, opt, scaler, epoch + 1, cfg)
        if is_main(rank) and ep["n"]:
            rec = {
                "epoch": epoch, "train_loss": ep["loss"] / ep["n"],
                "train_seg": ep["seg"] / ep["n"], "train_row": ep["row"] / ep["n"],
                "train_col": ep["col"] / ep["n"], "lr": opt.param_groups[0]["lr"],
                "lr_encoder": opt.param_groups[1]["lr"], "n_steps": ep["n"],
            }
            if cfg["grad_clip"] and cfg["grad_clip"] > 0:
                rec["grad_norm"] = ep["gnorm"] / ep["n"]     # mean PRE-clip grad norm (tune grad_clip against this)
                rec["clip_frac"] = ep["nclip"] / ep["n"]     # fraction of steps that hit the clip (~1.0 => raise grad_clip)
            if ep["nskip"]:
                rec["n_skip"] = ep["nskip"]                  # batches skipped for non-finite (fp16 forward overflow)
            if val is not None:
                rec.update(val_loss=val["loss"], val_seg=val["seg"],
                           val_row=val["row"], val_col=val["col"],
                           val_dir_row_acc=val["dir_row_acc"], val_dir_col_acc=val["dir_col_acc"],
                           val_fg_pred_bg=val["fg_pred_bg_frac"])
            log_json(cfg["out_dir"], rec)
        if is_main(rank) and ((epoch + 1) % cfg["save_every"] == 0 or epoch + 1 == cfg["epochs"]):
            save_ckpt(os.path.join(cfg["out_dir"], "checkpoint.pth"), model, opt, scaler, epoch + 1, cfg)
        if stop:
            break

    # always persist a final checkpoint (lets smoke runs verify save/resume)
    if is_main(rank):
        save_ckpt(os.path.join(cfg["out_dir"], "checkpoint.pth"), model, opt, scaler,
                  epoch + 1 if not stop else epoch, cfg)

    if is_main(rank):
        summary = {"epochs_trained": (epoch + 1 if not stop else epoch),
                   "encoder": cfg["encoder"], "init": cfg["init"], "out_dir": cfg["out_dir"]}
        if val_history:  # per-epoch val recorded -> report best (best.pth) / last (checkpoint.pth) / last-K mean
            last = val_history[-1]
            K = min(5, len(val_history))
            last_k_mean = sum(h["loss"] for h in val_history[-K:]) / K
            summary["val"] = {"best": best, "last": last, "last_k": K, "last_k_mean_loss": last_k_mean}
            print(f"[eval] best val loss {best['loss']:.4f} @ epoch {best['epoch']} (-> best.pth) | "
                  f"last {last['loss']:.4f} @ epoch {last['epoch']} (-> checkpoint.pth) | "
                  f"last-{K} mean {last_k_mean:.4f}")
            print(f"[eval] last fg direction acc: row={last.get('dir_row_acc', float('nan')):.3f} "
                  f"col={last.get('dir_col_acc', float('nan')):.3f} | fg-pred-bg-frac="
                  f"{last.get('fg_pred_bg_frac', float('nan')):.3f} (~1.0 = collapsed to background bin)")
        with open(os.path.join(cfg["out_dir"], "eval.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        log_json(cfg["out_dir"], {"final_eval": summary})
        if args.save_viz:
            vds = RidgepathSegDataset(cfg["manifest"], augment=False, params=cfg["target_params"])
            save_viz(model.module if isinstance(model, DDP) else model, vds, device,
                     os.path.join(cfg["out_dir"], "viz"))
            print(f"[viz] wrote debug outputs to {os.path.join(cfg['out_dir'], 'viz')}")
        print("done.")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
