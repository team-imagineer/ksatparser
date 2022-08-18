import os.path

import ksatparser

pdf_path = "pdf/problem/2022_7_writing.pdf"
output_dir = "./output"

ksatparser.parse_problem(pdf_path, output_dir)