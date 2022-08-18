import os
import ksatparser
from pathlib import Path

output_dir = Path("./output/")
pdf_dir = Path("./pdf/problem/")
done = ['6','9','11']
# todo = ['2022_4'
# ,'2016_10'
# ,'2018_10'
# ,'2022_4_writing'
# ,'2019_10'
# ,'2018_4'
# ,'2016_3'
# ,'2021_3'
# ,'2017_3'
# ,'2019_3'
# ,'2020_3'
# ,'2018_3']


for pdf_file in os.listdir(pdf_dir):
    try:
        month = pdf_file[:-4].split('_')[1]
        if month in done:
            continue
        if pdf_file[-4:] != ".pdf":
            continue
        # do = False
        # for td in todo:
        #     if td in pdf_file:
        #         do = True
        # if not do:
        #     continue
        print(pdf_file)
        pdf_path = pdf_dir/pdf_file
        ksatparser.parse_problem(pdf_path, output_dir)
    except Exception as e:
        print(e)
