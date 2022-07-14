import os.path

import ksatparser

pdf_path = "pdf/2010_11.pdf"
output_dir = "./output"

ksatparser.parse_pdf(pdf_path, output_dir)