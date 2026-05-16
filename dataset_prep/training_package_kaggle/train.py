"""FaceRestore training script — Kaggle T4x2 Edition with AMP + DataParallel + TensorBoard."""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# TensorBoard (Kaggle-native)
from torch.utils.tensorboard import SummaryWriter

# W&B (optional cloud dashboard)
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from .data import make_dataloader
    from .module import InpaintLoss, InpaintNet
except ImportError:
    from data import make_dataloader
    from module import InpaintLoss, InpaintNet


def expand_mask(mask_batch, n_channels):
    return mask_batch.unsqueeze(1).repeat(1, n_channels, 1, 1)


def preprocess_batch(batch, device, n_channels):
    image_batch, mask_batch = batch
    image_batch = image_batch.float().to(device)
    mask_batch = expand_mask(mask_batch.float().to(device), n_channels)
    image_masked_batch = image_batch * mask_batch
    return image_batch, image_masked_batch, mask_batch


def get_model_state_dict(model):
    """Get state dict without DataParallel module. prefix.
    Ensures checkpoints are portable between single-GPU and multi-GPU."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module.state_dict()
    return model.state_dict()


def run_epoch(model, criterion, dataloader, device, optimizer=None, scaler=None, epoch_num=0):
    training = optimizer is not None
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.train(training)
    total_loss = 0.0

    desc = f"Epoch {epoch_num} {'[train]' if training else '[val]'}"
    pbar = tqdm(dataloader, desc=desc, unit="batch", leave=True)

    for batch in pbar:
        image_batch, image_masked_batch, mask_batch = preprocess_batch(
            batch, device, raw_model.in_channels
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        # Mixed precision: forward pass in float16 for ~1.5-2x speedup on T4
        with torch.set_grad_enabled(training):
            with autocast(enabled=(scaler is not None)):
                output_batch, _ = model(image_masked_batch, mask_batch)
                loss = criterion(output_batch, mask_batch, image_batch).mean()

        if training:
            if scaler is not None:
                # AMP: scale loss, backward, unscale, step, update
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(len(dataloader), 1)


def make_loader(path, masks_dir, batch_size, workers, val):
    return make_dataloader(
        path,
        masks_dir,
        val=val,
        batch_size=batch_size,
        shuffle=not val,
        drop_last=not val,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Train FaceRestore — Kaggle T4x2 + AMP.')
    parser.add_argument('--train-dir', required=True, help='Directory with 256x256 face images.')
    parser.add_argument('--val-dir', help='Optional validation image directory.')
    parser.add_argument('--masks-dir', required=True, help='Quick Draw ndjson mask source directory.')
    parser.add_argument('--output-dir', default='checkpoints/facerestore', help='Checkpoint output directory.')
    parser.add_argument('--log-dir', default='logs', help='TensorBoard log directory.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--fine-tune-epoch', type=int, default=50)
    parser.add_argument('--fine-tune-gamma', type=float, default=0.1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Override starting epoch (for transitioning from old checkpoints)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision (AMP). Use if training is unstable.')
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_gpus = torch.cuda.device_count()
    device = torch.device(args.device)
    use_amp = (not args.no_amp) and torch.cuda.is_available()

    # === TensorBoard (Kaggle-native) ===
    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"TensorBoard logging to: {args.log_dir}")

    # === W&B (optional) ===
    if HAS_WANDB:
        wandb.init(
            project="facerestore-training",
            config={
                "train_dir": args.train_dir,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "device": args.device,
                "n_gpus": n_gpus,
                "amp": use_amp,
                "resume": args.resume or "scratch",
            },
            resume="allow",
        )
        print("W&B cloud logging: ACTIVE")
    else:
        print("W&B cloud logging: DISABLED")

    print(f"\nTraining config:")
    print(f"  Images:     {args.train_dir}")
    print(f"  Masks:      {args.masks_dir}")
    print(f"  Device:     {args.device}")
    print(f"  GPUs:       {n_gpus}")
    print(f"  Batch size: {args.batch_size} ({args.batch_size // max(n_gpus, 1)} per GPU)")
    print(f"  Workers:    {args.num_workers}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  LR:         {args.lr}")
    print(f"  AMP:        {'ON (float16 mixed precision)' if use_amp else 'OFF'}")
    print(f"  Resume:     {args.resume or 'None (from scratch)'}")
    print()

    train_loader = make_loader(
        args.train_dir, args.masks_dir, args.batch_size, args.num_workers, val=False
    )
    val_loader = None
    if args.val_dir:
        val_loader = make_loader(
            args.val_dir, args.masks_dir, args.batch_size, args.num_workers, val=True
        )

    model = InpaintNet(bn=True).to(device)
    start_epoch = 1

    if args.resume:
        raw = torch.load(args.resume, map_location=device, weights_only=False)

        # New format: dict with 'model' key
        if isinstance(raw, dict) and 'model' in raw:
            model_state = raw['model']
            start_epoch = raw.get('epoch', 0) + 1
            print(f"Resumed from {args.resume} → continuing at epoch {start_epoch}")
        else:
            # Old format: plain state_dict
            model_state = raw
            print(f"Resumed from {args.resume} (legacy weights)")

        # --start-epoch overrides auto-detected epoch
        if args.start_epoch > 0:
            start_epoch = args.start_epoch
            print(f"Manual override: starting at epoch {start_epoch}")

        # Strip module. prefix (portability between single/multi-GPU)
        for key in list(model_state.keys()):
            new_key = key.replace('module.', '')
            model_state[new_key] = model_state.pop(key)
        model.load_state_dict(model_state, strict=False)

    # === Multi-GPU: DataParallel ===
    if n_gpus >= 2:
        model = nn.DataParallel(model)
        print(f"DataParallel enabled: {n_gpus} GPUs")
    else:
        print(f"Single GPU mode")

    # === AMP: Mixed Precision Scaler ===
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("AMP (mixed precision): ENABLED — expect ~1.5-2x speedup on T4")

    criterion = InpaintLoss().to(device)
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[args.fine_tune_epoch],
        gamma=args.fine_tune_gamma
    )

    # Restore optimizer/scheduler if available
    if args.resume and isinstance(raw, dict) and 'optimizer' in raw:
        optimizer.load_state_dict(raw['optimizer'])
        if 'scheduler' in raw:
            scheduler.load_state_dict(raw['scheduler'])
        print("Restored optimizer & scheduler state")

    best_val = None
    end_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch, end_epoch):
        train_loss = run_epoch(model, criterion, train_loader, device, optimizer, scaler, epoch_num=epoch)
        val_loss = None
        if val_loader is not None:
            with torch.no_grad():
                val_loss = run_epoch(model, criterion, val_loader, device, epoch_num=epoch)

        scheduler.step()

        # Save clean checkpoint (no module. prefix)
        state_dict = get_model_state_dict(model)
        checkpoint = output_dir / 'epoch_{:03d}.pth'.format(epoch)
        torch.save(state_dict, checkpoint)
        if val_loss is not None and (best_val is None or val_loss < best_val):
            best_val = val_loss
            torch.save(state_dict, output_dir / 'best.pth')

        # Save last.pth with full training state
        torch.save({
            'model': state_dict,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, output_dir / 'last.pth')

        current_lr = optimizer.param_groups[0]['lr']
        message = 'epoch={} train_loss={:.6f} lr={:.6g}'.format(epoch, train_loss, current_lr)
        if val_loss is not None:
            message += ' val_loss={:.6f}'.format(val_loss)
        print(message, flush=True)

        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        if val_loss is not None:
            writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        writer.flush()

        # W&B
        if HAS_WANDB:
            log_data = {"train_loss": train_loss, "lr": current_lr, "epoch": epoch}
            if val_loss is not None:
                log_data["val_loss"] = val_loss
            wandb.log(log_data)

    writer.close()
    if HAS_WANDB:
        wandb.finish()
        print("\nLogging complete.")
        print("  TensorBoard: Output tab → TensorBoard button")
        print("  W&B: https://wandb.ai")


if __name__ == '__main__':
    main()
