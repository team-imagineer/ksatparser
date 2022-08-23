import os
import ksatparser
from pathlib import Path

output_dir = Path("./tmp/")
pdf_dir = Path("./pdf/problem/")
# ['2022_4'
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

# = ['6','9','11']

years = [str(y) for y in range(2016, 2023)]
months = ['6','9','11']

todo = [y+'_'+m for m in months for y in years]
print(todo)
# todo = ['2019_11']



for pdf_file in os.listdir(pdf_dir):
    try:
        file_type = pdf_file[-4:]
        file_name = pdf_file[:-4]
        month = pdf_file[:-4].split('_')[1]
        if file_type != ".pdf":
            continue
        if not file_name in todo:
            continue
        print(pdf_file)

        pdf_path = pdf_dir/pdf_file
        ksatparser.parse_problem(pdf_path, output_dir)
    except Exception as e:
        print(e)
