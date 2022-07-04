import argparse
from curses import meta
import glob
import os
import cv2
import tqdm

from seodaang_parser.config import Config
from seodaang_parser.parser.contour import get_problem


def setup_cfg(args):
    """load config from file and command-line arguments"""
    cfg = Config(args.config_file)
    return cfg


def get_parser():
    """get command-line arguments"""
    parser = argparse.ArgumentParser(
        description="seodaang-parser demo for builtin configs"
    )
    parser.add_argument(
        "--config-file",
        default="configs/math.json",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input-file",
        default="data/test.png",
        metavar="FILE",
        help="path to input image file",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    print(args.input_file)

    img = cv2.imread(args.input_file)

    cv2.imshow("roi", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # get_problem(img, cfg)

