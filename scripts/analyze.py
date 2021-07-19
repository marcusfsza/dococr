# Copyright (C) 2021, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

from doctr.models import ocr_predictor
from doctr.documents import DocumentFile


def main(args):

    model = ocr_predictor(args.detection, args.recognition, pretrained=True)
    if args.path.endswith(".pdf"):
        doc = DocumentFile.from_pdf(args.path).as_images()
    else:
        doc = DocumentFile.from_images(args.path)

    out = model(doc, training=False)

    for page, img in zip(out.pages, doc):
        page.show(img, block=not args.noblock, interactive=not args.static)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DocTR end-to-end analysis',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('path', type=str, help='Path to the input document (PDF or image)')
    parser.add_argument('--detection', type=str, default='db_resnet50',
                        help='Text detection model to use for analysis')
    parser.add_argument('--recognition', type=str, default='crnn_vgg16_bn',
                        help='Text recognition model to use for analysis')
    parser.add_argument("--noblock", dest="noblock", help="Disables blocking visualization", action="store_true")
    parser.add_argument("--static", dest="static", help="Switches to static visualization", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
