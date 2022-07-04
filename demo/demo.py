import argparse
import os
import cv2

from tqdm import tqdm
from seodaang_parser.config import Config
from seodaang_parser.parser.contour import get_problem


def setup_cfg(args):
    """load config from file and command-line arguments"""
    return Config(args.config)


def get_parser():
    """get command-line arguments"""
    parser = argparse.ArgumentParser(
        description="seodaang-parser demo for builtin configs"
    )
    parser.add_argument(
        "--config", default="configs/math.json", help="path to config file",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output", help="path to output directory", default="data/crop/"
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    for page_num, path in enumerate(tqdm(args.input)):
        img = cv2.imread(path)
        for problem_num, roi in enumerate(get_problem(img, cfg)):
            output_path = os.path.join(args.output, f"{page_num}_{problem_num}.png")
            cv2.imwrite(output_path, roi)
