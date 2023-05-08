import os.path

import ksatparser

pdf_path = "pdf/problem/2017_10.pdf"
output_dir = "./tmp"

ksatparser.parse_problem(pdf_path, output_dir)