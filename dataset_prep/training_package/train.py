import argparse
from pathlib import Path

import torch
from tqdm import tqdm

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


def run_epoch(model, criterion, dataloader, device, optimizer=None, epoch_num=0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    desc = f"Epoch {epoch_num} {'[train]' if training else '[val]'}"
    pbar = tqdm(dataloader, desc=desc, unit="batch", leave=True)

    for batch in pbar:
        image_batch, image_masked_batch, mask_batch = preprocess_batch(
            batch, device, model.in_channels
        )

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            output_batch, _ = model(image_masked_batch, mask_batch)
            loss = criterion(output_batch, mask_batch, image_batch).mean()

        if training:
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
    parser = argparse.ArgumentParser(description='Train FaceRestore partial-convolution model.')
    parser.add_argument('--train-dir', required=True, help='Directory with square face training images.')
    parser.add_argument('--val-dir', help='Optional validation image directory.')
    parser.add_argument('--masks-dir', required=True, help='Quick Draw ndjson mask source directory.')
    parser.add_argument('--output-dir', default='checkpoints/facerestore', help='Checkpoint output directory.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--fine-tune-epoch', type=int, default=50)
    parser.add_argument('--fine-tune-gamma', type=float, default=0.1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Override starting epoch (use when transitioning from old checkpoints)')
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B cloud logging (survives Colab crashes)
    if HAS_WANDB:
        wandb.init(
            project="facerestore-training",
            config={
                "train_dir": args.train_dir,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "device": args.device,
                "resume": args.resume or "scratch",
            },
            resume="allow",  # allows resuming a previous W&B run
        )
        print("W&B cloud logging: ACTIVE (logs at https://wandb.ai)")
    else:
        print("W&B cloud logging: DISABLED (pip install wandb to enable)")

    print(f"\nTraining config:")
    print(f"  Images:     {args.train_dir}")
    print(f"  Masks:      {args.masks_dir}")
    print(f"  Device:     {args.device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Workers:    {args.num_workers}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  LR:         {args.lr}")
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

    device = torch.device(args.device)
    model = InpaintNet(bn=True).to(device)
    start_epoch = 1

    if args.resume:
        raw = torch.load(args.resume, map_location=device, weights_only=False)

        # New format: dict with 'model' key (saved by this script)
        if isinstance(raw, dict) and 'model' in raw:
            model_state = raw['model']
            start_epoch = raw.get('epoch', 0) + 1
            print(f"Resumed from {args.resume} → continuing at epoch {start_epoch}")
        else:
         # Old format: plain state_dict (original model.state_dict)
            model_state = raw
            print(f"Resumed from {args.resume} (legacy weights)")

        # --start-epoch overrides auto-detected epoch (for transitioning old checkpoints)
        if args.start_epoch > 0:
            start_epoch = args.start_epoch
            print(f"Manual override: starting at epoch {start_epoch}")

        # Handle multi-GPU saved models
        for key in list(model_state.keys()):
            new_key = key.replace('module.', '')
            model_state[new_key] = model_state.pop(key)
        model.load_state_dict(model_state, strict=False)

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

    # Restore optimizer/scheduler if resuming from new-format checkpoint
    if args.resume and isinstance(raw, dict) and 'optimizer' in raw:
        optimizer.load_state_dict(raw['optimizer'])
        if 'scheduler' in raw:
            scheduler.load_state_dict(raw['scheduler'])
        print("Restored optimizer & scheduler state")

    best_val = None
    end_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch, end_epoch):
        train_loss = run_epoch(model, criterion, train_loader, device, optimizer, epoch_num=epoch)
        val_loss = None
        if val_loader is not None:
            with torch.no_grad():
                val_loss = run_epoch(model, criterion, val_loader, device, epoch_num=epoch)

        scheduler.step()

        # Save epoch checkpoint (numbered correctly across sessions)
        checkpoint = output_dir / 'epoch_{:03d}.pth'.format(epoch)
        torch.save(model.state_dict(), checkpoint)
        if val_loss is not None and (best_val is None or val_loss < best_val):
            best_val = val_loss
            torch.save(model.state_dict(), output_dir / 'best.pth')

        # Save last.pth with full state (so next session knows what epoch to continue from)
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, output_dir / 'last.pth')

        current_lr = optimizer.param_groups[0]['lr']
        message = 'epoch={} train_loss={:.6f} lr={:.6g}'.format(epoch, train_loss, current_lr)
        if val_loss is not None:
            message += ' val_loss={:.6f}'.format(val_loss)
        print(message, flush=True)

        # Log to W&B cloud (persists even if Colab crashes)
        if HAS_WANDB:
            log_data = {"train_loss": train_loss, "lr": current_lr, "epoch": epoch}
            if val_loss is not None:
                log_data["val_loss"] = val_loss
            wandb.log(log_data)

    if HAS_WANDB:
        wandb.finish()
        print("\nW&B logging complete. View your logs at https://wandb.ai")


if __name__ == '__main__':
    main()
