import os

from pathlib import Path
from typing import Union
from pdf2image import convert_from_path


def convert_pdf_to_image(
    path: Union[Path, str], output_path: Union[Path, str], dpi=100
):
    """https://stackoverflow.com/questions/46184239/extract-a-page-from-a-pdf-as-a-jpeg"""
    pages = convert_from_path(path, dpi, thread_count=2, fmt="png")
    for page_num, page in enumerate(pages):
        page.save(os.path.join(output_path, f"{page_num}.png"))
