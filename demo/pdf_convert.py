import argparse

from seodaang_parser.converter import convert_pdf_to_image


def get_parser():
    """get command-line arguments"""
    parser = argparse.ArgumentParser(
        description="seodaang-parser demo for builtin configs"
    )
    parser.add_argument(
        "--input", default="data/pdf/1.pdf", help="path to pdf file",
    )
    parser.add_argument(
        "--output", default="data/raw/problem/2", help="path to output file",
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    convert_pdf_to_image(args.input, args.output)
