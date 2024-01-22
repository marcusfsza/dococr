# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os

os.environ["USE_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import datetime
import multiprocessing as mp
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tqdm.auto import tqdm

from doctr.models import login_to_hub, push_to_hf_hub

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from doctr import transforms as T
from doctr.datasets import DataLoader, OrientationDataset
from doctr.models import classification
from doctr.models.utils import export_model_to_onnx
from doctr.transforms.functional import rotated_img_tensor
from utils import EarlyStopper, plot_recorder, plot_samples

CLASSES = [0, 90, 180, 270]


def rnd_rotate(img: tf.Tensor, target):
    angle = int(np.random.choice(CLASSES))
    idx = CLASSES.index(angle)
    # clockwise rotation
    rotated_img = rotated_img_tensor(img, -angle, expand=False)
    return rotated_img, idx


def record_lr(
    model: tf.keras.Model,
    train_loader: DataLoader,
    batch_transforms,
    optimizer,
    start_lr: float = 1e-7,
    end_lr: float = 1,
    num_it: int = 100,
    amp: bool = False,
):
    """Gridsearch the optimal learning rate for the training.
    Adapted from https://github.com/frgfm/Holocron/blob/master/holocron/trainer/core.py
    """
    if num_it > len(train_loader):
        raise ValueError("the value of `num_it` needs to be lower than the number of available batches")

    # Update param groups & LR
    gamma = (end_lr / start_lr) ** (1 / (num_it - 1))
    optimizer.learning_rate = start_lr

    lr_recorder = [start_lr * gamma**idx for idx in range(num_it)]
    loss_recorder = []

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = batch_transforms(images)

        # Forward, Backward & update
        with tf.GradientTape() as tape:
            out = model(images, training=True)
            train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(targets, out)
        grads = tape.gradient(train_loss, model.trainable_weights)

        if amp:
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        optimizer.learning_rate = optimizer.learning_rate * gamma

        # Record
        train_loss = train_loss.numpy()
        if np.any(np.isnan(train_loss)):
            if batch_idx == 0:
                raise ValueError("loss value is NaN or inf.")
            else:
                break
        loss_recorder.append(train_loss.mean())
        # Stop after the number of iterations
        if batch_idx + 1 == num_it:
            break

    return lr_recorder[: len(loss_recorder)], loss_recorder


def fit_one_epoch(model, train_loader, batch_transforms, optimizer, amp=False):
    # Iterate over the batches of the dataset
    pbar = tqdm(train_loader, position=1)
    for images, targets in pbar:
        images = batch_transforms(images)

        with tf.GradientTape() as tape:
            out = model(images, training=True)
            train_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(targets, out)
        grads = tape.gradient(train_loss, model.trainable_weights)
        if amp:
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        pbar.set_description(f"Training loss: {train_loss.numpy().mean():.6}")


def evaluate(model, val_loader, batch_transforms):
    # Validation loop
    val_loss, correct, samples, batch_cnt = 0.0, 0.0, 0.0, 0.0
    val_iter = iter(val_loader)
    for images, targets in tqdm(val_iter):
        images = batch_transforms(images)
        out = model(images, training=False)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(targets, out)
        # Compute metric
        correct += int((out.numpy().argmax(1) == targets.numpy()).sum())

        val_loss += loss.numpy().mean()
        batch_cnt += 1
        samples += images.shape[0]

    val_loss /= batch_cnt
    acc = correct / samples
    return val_loss, acc


def collate_fn(samples):
    images, targets = zip(*samples)
    images = tf.stack(images, axis=0)

    return images, tf.convert_to_tensor(targets)


def main(args):
    print(args)

    if args.push_to_hub:
        login_to_hub()

    if not isinstance(args.workers, int):
        args.workers = min(16, mp.cpu_count())

    input_size = (256, 256) if args.type == "page" else (32, 32)

    # AMP
    if args.amp:
        mixed_precision.set_global_policy("mixed_float16")

    st = time.time()
    val_set = OrientationDataset(
        img_folder=os.path.join(args.val_path, "images"),
        img_transforms=T.Compose([
            T.Resize(input_size, preserve_aspect_ratio=True, symmetric_pad=True),
        ]),
        sample_transforms=T.SampleCompose([
            lambda x, y: rnd_rotate(x, y),
            T.Resize(input_size),
        ]),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )
    print(
        f"Validation set loaded in {time.time() - st:.4}s ({len(val_set)} samples in "
        f"{val_loader.num_batches} batches)"
    )

    # Load doctr model
    model = classification.__dict__[args.arch](
        pretrained=args.pretrained,
        input_shape=(*(input_size), 3),
        num_classes=len(CLASSES),
        classes=CLASSES,
        include_top=True,
    )

    # Resume weights
    if isinstance(args.resume, str):
        model.load_weights(args.resume)

    batch_transforms = T.Compose([
        T.Normalize(mean=(0.694, 0.695, 0.693), std=(0.299, 0.296, 0.301)),
    ])

    if args.test_only:
        print("Running evaluation")
        val_loss, acc = evaluate(model, val_loader, batch_transforms)
        print(f"Validation loss: {val_loss:.6} (Acc: {acc:.2%})")
        return

    st = time.time()
    train_set = OrientationDataset(
        img_folder=os.path.join(args.train_path, "images"),
        img_transforms=T.Compose([
            T.Resize(input_size, preserve_aspect_ratio=True, symmetric_pad=True),
            # Augmentations
            T.RandomApply(T.ColorInversion(), 0.1),
            T.RandomApply(T.ToGray(3), 0.1),
            T.RandomJpegQuality(60),
            T.RandomSaturation(0.3),
            T.RandomContrast(0.3),
            T.RandomBrightness(0.3),
            # Blur
            T.RandomApply(T.GaussianBlur(kernel_shape=(3, 3), std=(0.1, 3)), 0.1),
        ]),
        sample_transforms=T.SampleCompose([
            lambda x, y: rnd_rotate(x, y),
            T.Resize(input_size),
        ]),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )
    print(
        f"Train set loaded in {time.time() - st:.4}s ({len(train_set)} samples in "
        f"{train_loader.num_batches} batches)"
    )

    if args.show_samples:
        x, target = next(iter(train_loader))
        plot_samples(x, [CLASSES[t] for t in target])
        return

    # Optimizer
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        args.lr,
        decay_steps=args.epochs * len(train_loader),
        decay_rate=1 / (1e3),  # final lr as a fraction of initial lr
        staircase=False,
        name="ExponentialDecay",
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=scheduler,
        beta_1=0.95,
        beta_2=0.99,
        epsilon=1e-6,
    )
    if args.amp:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    # LR Finder
    if args.find_lr:
        lrs, losses = record_lr(model, train_loader, batch_transforms, optimizer, amp=args.amp)
        plot_recorder(lrs, losses)
        return

    # Tensorboard to monitor training
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.arch}_{current_time}" if args.name is None else args.name

    config = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "architecture": args.arch,
        "input_size": input_size,
        "optimizer": "adam",
        "framework": "pytorch",
        "classes": CLASSES,
        "scheduler": args.sched,
        "pretrained": args.pretrained,
    }

    # W&B
    if args.wb:
        import wandb

        run = wandb.init(name=exp_name, project="orientation-classification", config=config)
    # ClearML
    if args.clearml:
        from clearml import Task

        task = Task.init(project_name="docTR/orientation-classification", task_name=exp_name, reuse_last_task_id=False)
        task.upload_artifact("config", config)

    # Create loss queue
    min_loss = np.inf

    # Training loop
    if args.early_stop:
        early_stopper = EarlyStopper(patience=args.early_stop_epochs, min_delta=args.early_stop_delta)
    for epoch in range(args.epochs):
        fit_one_epoch(model, train_loader, batch_transforms, optimizer, args.amp)

        # Validation loop at the end of each epoch
        val_loss, acc = evaluate(model, val_loader, batch_transforms)
        if val_loss < min_loss:
            print(f"Validation loss decreased {min_loss:.6} --> {val_loss:.6}: saving state...")
            model.save_weights(f"./{exp_name}/weights")
            min_loss = val_loss
        print(f"Epoch {epoch + 1}/{args.epochs} - Validation loss: {val_loss:.6} (Acc: {acc:.2%})")
        # W&B
        if args.wb:
            wandb.log({
                "val_loss": val_loss,
                "acc": acc,
            })

        # ClearML
        if args.clearml:
            from clearml import Logger

            logger = Logger.current_logger()
            logger.report_scalar(title="Validation Loss", series="val_loss", value=val_loss, iteration=epoch)
            logger.report_scalar(title="Accuracy", series="acc", value=acc, iteration=epoch)
        if args.early_stop and early_stopper.early_stop(val_loss):
            print("Training halted early due to reaching patience limit.")
            break
    if args.wb:
        run.finish()

    if args.push_to_hub:
        push_to_hf_hub(model, exp_name, task="classification", run_config=args)

    if args.export_onnx:
        print("Exporting model to ONNX...")
        if args.arch == "vit_b":
            # fixed batch size for vit
            dummy_input = [tf.TensorSpec([1, *(input_size), 3], tf.float32, name="input")]
        else:
            # dynamic batch size
            dummy_input = [tf.TensorSpec([None, *(input_size), 3], tf.float32, name="input")]
        model_path, _ = export_model_to_onnx(model, exp_name, dummy_input)
        print(f"Exported model saved in {model_path}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="DocTR training script for orientation classification (TensorFlow)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("train_path", type=str, help="path to training data folder")
    parser.add_argument("val_path", type=str, help="path to validation data folder")
    parser.add_argument("arch", type=str, help="classification model to train")
    parser.add_argument("type", type=str, choices=["page", "crop"], help="type of data to train on")
    parser.add_argument("--name", type=str, default=None, help="Name of your training experiment")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train the model on")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="batch size for training")
    parser.add_argument("--device", default=None, type=int, help="device")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate for the optimizer (Adam)")
    parser.add_argument("--wd", "--weight-decay", default=0, type=float, help="weight decay", dest="weight_decay")
    parser.add_argument("-j", "--workers", type=int, default=None, help="number of workers used for dataloading")
    parser.add_argument("--resume", type=str, default=None, help="Path to your checkpoint")
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Run the validation loop")
    parser.add_argument(
        "--show-samples", dest="show_samples", action="store_true", help="Display unormalized training samples"
    )
    parser.add_argument("--wb", dest="wb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--clearml", dest="clearml", action="store_true", help="Log to ClearML")
    parser.add_argument("--push-to-hub", dest="push_to_hub", action="store_true", help="Push to Huggingface Hub")
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="Load pretrained parameters before starting the training",
    )
    parser.add_argument("--export-onnx", dest="export_onnx", action="store_true", help="Export the model to ONNX")
    parser.add_argument("--sched", type=str, default="cosine", help="scheduler to use")
    parser.add_argument("--amp", dest="amp", help="Use Automatic Mixed Precision", action="store_true")
    parser.add_argument("--find-lr", action="store_true", help="Gridsearch the optimal LR")
    parser.add_argument("--early-stop", action="store_true", help="Enable early stopping")
    parser.add_argument("--early-stop-epochs", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--early-stop-delta", type=float, default=0.01, help="Minimum Delta for early stopping")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
