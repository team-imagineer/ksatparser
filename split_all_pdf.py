import os
import ksatparser
from pathlib import Path
from tqdm import tqdm

output_dir = Path("./output")
pdf_dir = Path("./pdf")

num = 1000
i = 0
for pdf_file in tqdm(os.listdir(pdf_dir)):
    if pdf_file[-4:] != ".pdf":
        continue
    if num <= i:
        break
    pdf_path = pdf_dir/pdf_file
    ksatparser.parse_pdf(pdf_path, output_dir)
    i += 1
