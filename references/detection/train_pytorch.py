# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os

os.environ['USE_TORCH'] = '1'

import time
import datetime
import multiprocessing as mp
import numpy as np
from fastprogress.fastprogress import master_bar, progress_bar
import torch
from torchvision.transforms import Compose, Lambda, Normalize, ColorJitter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from contiguous_params import ContiguousParams
import wandb

from doctr.models import detection
from doctr.utils.metrics import LocalizationConfusion
from doctr.datasets import DetectionDataset
from doctr import transforms as T

from utils import plot_samples


def fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, mb):
    model.train()
    train_iter = iter(train_loader)
    # Iterate over the batches of the dataset
    for _ in progress_bar(range(len(train_loader)), parent=mb):
        images, targets = next(train_iter)

        if torch.cuda.is_available():
            images = images.cuda()
        images = batch_transforms(images)

        train_loss = model(images, targets)['loss']

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        mb.child.comment = f'Training loss: {train_loss.item():.6}'


@torch.no_grad()
def evaluate(model, val_loader, batch_transforms, val_metric):
    # Model in eval mode
    model.eval()
    # Reset val metric
    val_metric.reset()
    # Validation loop
    val_loss, batch_cnt = 0, 0
    val_iter = iter(val_loader)
    for images, targets in val_iter:
        if torch.cuda.is_available():
            images = images.cuda()
        images = batch_transforms(images)
        out = model(images, targets, return_boxes=True)
        # Compute metric
        loc_preds, _ = out['preds']
        for boxes_gt, boxes_pred in zip([t['boxes'] for t in targets], loc_preds):
            # Remove scores
            val_metric.update(gts=boxes_gt, preds=boxes_pred[:, :-1])

        val_loss += out['loss'].item()
        batch_cnt += 1

    val_loss /= batch_cnt
    recall, precision, mean_iou = val_metric.summary()
    return val_loss, recall, precision, mean_iou


def main(args):

    print(args)

    if not isinstance(args.workers, int):
        args.workers = min(16, mp.cpu_count())

    torch.backends.cudnn.benchmark = True

    st = time.time()
    val_set = DetectionDataset(
        img_folder=os.path.join(args.data_path, 'val'),
        label_folder=os.path.join(args.data_path, 'val_labels'),
        sample_transforms=T.Resize((args.input_size, args.input_size)),
        rotated_bbox=args.rotation
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        sampler=SequentialSampler(val_set),
        pin_memory=True,
        collate_fn=val_set.collate_fn,
    )
    print(f"Validation set loaded in {time.time() - st:.4}s ({len(val_set)} samples in "
          f"{len(val_loader)} batches)")

    batch_transforms = Normalize(mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287))

    # Load doctr model
    model = detection.__dict__[args.model](pretrained=args.pretrained)

    # Resume weights
    if isinstance(args.resume, str):
        print(f"Resuming {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint)

    # GPU
    if isinstance(args.device, int):
        if not torch.cuda.is_available():
            raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
        if args.device >= torch.cuda.device_count():
            raise ValueError("Invalid device index")
        torch.cuda.set_device(args.device)
        model = model.cuda()

    # Metrics
    val_metric = LocalizationConfusion(rotated_bbox=args.rotation, mask_shape=(args.input_size, args.input_size))

    if args.test_only:
        print("Running evaluation")
        val_loss, recall, precision, mean_iou = evaluate(model, val_loader, batch_transforms, val_metric)
        print(f"Validation loss: {val_loss:.6} (Recall: {recall:.2%} | Precision: {precision:.2%} | "
              f"Mean IoU: {mean_iou:.2%})")
        return

    st = time.time()
    # Load both train and val data generators
    train_set = DetectionDataset(
        img_folder=os.path.join(args.data_path, 'train'),
        label_folder=os.path.join(args.data_path, 'train_labels'),
        sample_transforms=Compose([
            T.Resize((args.input_size, args.input_size)),
            # Augmentations
            T.RandomApply(T.ColorInversion(), .1),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
        ]),
        rotated_bbox=args.rotation
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        drop_last=True,
        num_workers=args.workers,
        sampler=RandomSampler(train_set),
        pin_memory=True,
        collate_fn=train_set.collate_fn,
    )
    print(f"Train set loaded in {time.time() - st:.4}s ({len(train_set)} samples in "
          f"{len(train_loader)} batches)")

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, target, rotation=args.rotation)
        return

    # Backbone freezing
    if args.freeze_backbone:
        for p in model.feat_extractor.parameters():
            p.reguires_grad_(False)

    # Optimizer
    model_params = ContiguousParams([p for p in model.parameters() if p.requires_grad]).contiguous()
    optimizer = torch.optim.Adam(model_params, args.lr,
                                 betas=(0.95, 0.99), eps=1e-6, weight_decay=args.weight_decay)
    # Scheduler
    if args.sched == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, args.epochs * len(train_loader), eta_min=args.lr / 25e4)
    elif args.sched == 'onecycle':
        scheduler = OneCycleLR(optimizer, args.lr, args.epochs * len(train_loader))

    # Training monitoring
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.model}_{current_time}" if args.name is None else args.name

    # W&B
    if args.wb:

        run = wandb.init(
            name=exp_name,
            project="text-detection",
            config={
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "architecture": args.model,
                "input_size": args.input_size,
                "optimizer": "adam",
                "exp_type": "text-detection",
                "framework": "pytorch",
            }
        )

    # Create loss queue
    min_loss = np.inf

    # Training loop
    mb = master_bar(range(args.epochs))
    for epoch in mb:
        fit_one_epoch(model, train_loader, batch_transforms, optimizer, scheduler, mb)
        # Validation loop at the end of each epoch
        val_loss, recall, precision, mean_iou = evaluate(model, val_loader, batch_transforms, val_metric)
        if val_loss < min_loss:
            print(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
            torch.save(model.state_dict(), f"./{exp_name}.pt")
            min_loss = val_loss
        mb.write(f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} "
                 f"(Recall: {recall:.2%} | Precision: {precision:.2%} | Mean IoU: {mean_iou:.2%})")
        # W&B
        if args.wb:
            wandb.log({
                'epochs': epoch + 1,
                'val_loss': val_loss,
                'recall': recall,
                'precision': precision,
                'mean_iou': mean_iou,
            })
        # Reset val metric
        val_metric.reset()

    if args.wb:
        run.finish()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR train text-detection model (PyTorch)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('data_path', type=str, help='path to data folder')
    parser.add_argument('model', type=str, help='text-detection model to train')
    parser.add_argument('--name', type=str, default=None, help='Name of your training experiment')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train the model on')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='batch size for training')
    parser.add_argument('--device', default=None, type=int, help='device')
    parser.add_argument('--input_size', type=int, default=1024, help='model input size, H = W')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for the optimizer (Adam)')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float, help='weight decay', dest='weight_decay')
    parser.add_argument('-j', '--workers', type=int, default=None, help='number of workers used for dataloading')
    parser.add_argument('--resume', type=str, default=None, help='Path to your checkpoint')
    parser.add_argument("--test-only", dest='test_only', action='store_true', help="Run the validation loop")
    parser.add_argument('--freeze-backbone', dest='freeze_backbone', action='store_true',
                        help='freeze model backbone for fine-tuning')
    parser.add_argument('--show-samples', dest='show_samples', action='store_true',
                        help='Display unormalized training samples')
    parser.add_argument('--wb', dest='wb', action='store_true',
                        help='Log to Weights & Biases')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='Load pretrained parameters before starting the training')
    parser.add_argument('--rotation', dest='rotation', action='store_true',
                        help='train with rotated bbox')
    parser.add_argument('--sched', type=str, default='cosine', help='scheduler to use')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
