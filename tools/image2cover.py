from argparse import ArgumentParser

import cv2
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('cover', help='Cover file')
    args = parser.parse_args()
    return args


def main(args):
    img = np.load(args.img)[:, :, 3:6].astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.cover, img)


if __name__ == '__main__':
    args = parse_args()
    main(args)
