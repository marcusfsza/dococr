# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


import os

os.environ['USE_TORCH'] = '1'

import argparse
import logging

import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch

from doctr.models import obj_detection


def plot_predictions(image, tg_boxes, tg_labels, cl_map, cm):
    for ind_2, val_2 in enumerate(tg_boxes):
        cv2.rectangle(image, (int(val_2[0]), int(val_2[1])), (int(val_2[2]), int(val_2[3])),
                      cm[str(int(tg_labels[ind_2]))][0], 2)
        text_size, _ = cv2.getTextSize(cl_map[str(int(tg_labels[ind_2]))], cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        text_w, text_h = text_size
        cv2.rectangle(image, (int(val_2[0]), int(val_2[1])), (int(val_2[0]) + text_w, int(val_2[1]) - text_h),
                      cm[str(int(tg_labels[ind_2]))][0], -1)
        cv2.putText(image, cl_map[str(int(tg_labels[ind_2]))], (int(val_2[0]), int(val_2[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    figure(figsize=(9, 7), dpi=100)
    plt.imshow(image)
    plt.show()


def main(args):
    print(args)

    model = obj_detection.__dict__[args.arch](pretrained=True, num_classes=5)
    model.eval()
    # GPU
    if isinstance(args.device, int):
        if not torch.cuda.is_available():
            raise AssertionError("PyTorch cannot access your GPU. Please investigate!")
        if args.device >= torch.cuda.device_count():
            raise ValueError("Invalid device index")
    # Silent default switch to GPU if available
    elif torch.cuda.is_available():
        args.device = 0
    else:
        logging.warning("No accessible GPU, target device set to CPU.")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        model = model.cuda()

    cm = {'1': [(0, 0, 150)], '2': [(0, 0, 0)], '3': [(0, 150, 0)], '4': [(150, 0, 0)]}
    cl_map = {'1': "QR_Code", "2": "Bar_Code", "3": "Logo", "4": "Photo"}
    for val in os.listdir(args.root_dir):
        im_read = cv2.imread(os.path.join(args.root_dir, val))
        imm = cv2.cvtColor(im_read, cv2.COLOR_BGR2RGB)
        imm2 = imm / 255
        if torch.cuda.is_available():
            imm2 = [torch.tensor(imm2, dtype=torch.float32).permute(2, 0, 1).cuda()]
        else:
            imm2 = [torch.tensor(imm2, dtype=torch.float32).permute(2, 0, 1)]
        pred = model(imm2)
        tg_labels = pred[0]['labels'].detach().cpu().numpy()
        tg_boxes = pred[0]['boxes'].detach().cpu().numpy()
        tg_boxes = [list(i) for i in tg_boxes]
        plot_predictions(imm, tg_boxes, tg_labels, cl_map, cm)


def parse_args():
    parser = argparse.ArgumentParser(description="Artefact Detection Training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('arch', type=str, help='text-detection model to train')
    parser.add_argument('root_dir', type=str, help='path to image folder')
    parser.add_argument('--device', default=None, type=int, help='device')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
