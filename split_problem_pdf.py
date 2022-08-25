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

years = [str(y) for y in range(2016, 2021)]
months = ['3','4','7','10']
# months

todo = [y+'_'+m for m in months for y in years]
# todo = ['2016_3']
print(todo)

error_files = []
error_msgs = []

for pdf_file in os.listdir(pdf_dir):
    try:
        file_type = pdf_file[-4:]
        file_name = pdf_file[:-4]
        month = pdf_file[:-4].split('_')[1]
        file_ym = file_name.split('_')[0]+'_'+file_name.split('_')[1]
        if file_type != ".pdf":
            continue
        if not file_ym in todo:
            continue
        print(pdf_file)

        pdf_path = pdf_dir/pdf_file
        ksatparser.parse_problem(pdf_path, output_dir)
    except Exception as e:
        error_files.append(pdf_file)
        error_msgs.append(e)
