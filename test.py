import os.path

import ksatparser

pdf_path = "pdf/problem/2021_7_media.pdf"
output_dir = "./tmp"

ksatparser.parse_problem(pdf_path, output_dir)