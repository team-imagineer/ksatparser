import os
import numpy as np
from pdf2image import convert_from_path
import cv2
from pathlib import Path
import pdfplumber
import tempfile
from .utils import *
from collections import defaultdict


def parse_ebs_solution(pdf_path, output_dir):
    """
    Description:
        parse pdf into content(지문) & problem(문제)
        save png files into output_dir
    Args:
        path: pdf_path
        output_dir: str or pathlib.Path
    """
    # pdf_path, output_dir to pathlib.Path
    if isinstance(pdf_path, str):
        pdf_path = Path(pdf_path)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    imgs = pdf2pngs(pdf_path)

    blocks, jimoons_list, jimoon_names, probs_list, prob_names = get_ebs_solution_elements(pdf_path, imgs)

    # for test
    for b in blocks:
        print(b.shape)

    # cut blocks list
    min_width = min([blocks[i].shape[1] for i in range(len(blocks))])
    for i in range(len(blocks)):
        blocks[i] = blocks[i][:, :min_width]

    # concatenate
    long_block = np.concatenate(blocks)

    # concatenate jimoons_list, probs_list
    add_height = 0
    jimoons = []

    tmp_prob_names = []  # to prevent not problem names are included. ex) 보기문의 문장 안에 "3."
    probs = []
    prob_idx = -1 # 문제 세는
    num_dict = defaultdict(bool) # 해설지의 경우, 답이 먼저 나오기 때문에 이전에 나온지 있는지 확인

    for i, block in enumerate(blocks):

        for x0, y0, x1, y1 in jimoons_list[i]:
            # ebs해설에서는 사용 X
            # if x0 >= 60:  # delete not a jimoon
            #     continue
            y0 += add_height
            y1 += add_height
            jimoons.append((x0, y0, x1, y1))

        for x0, y0, x1, y1 in probs_list[i]:
            prob_idx += 1
            # ebs해설에서는 사용 X
            # if x0 >= 60:  # delete not a problem
            #     continue
            y0 += add_height
            y1 += add_height
            probs.append((x0, y0, x1, y1))
            tmp_prob_names.append(prob_names[prob_idx])

        add_height += block.shape[0]

    # prob_names change
    prob_names = tmp_prob_names

    contour_ends = []

    save_ebs_solution(pdf_path, long_block, jimoons, jimoon_names, probs, prob_names, contour_ends, output_dir)

    print(f"parse {pdf_path} complete")


def get_top_bottom_horizontal_lines(img):
    # canny preprocessing
    thresh1 = 200
    thresh2 = 20
    dst = cv2.Canny(img, thresh1, thresh2)

    # problematic hough
    lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 100, 2)

    horizontal_lines = []
    for line in lines:
        xlen = abs(line[0][2] - line[0][0])
        width_thresh = img.shape[1] * 0.6
        if xlen >= width_thresh:
            horizontal_lines.append(line[0])

    horizontal_lines.sort(key=lambda x:x[1])

    top_line = horizontal_lines[0]
    bottom_line = horizontal_lines[-1]
    return top_line, bottom_line


def get_ebs_solution_elements(pdf_path, imgs):
    blocks = []
    probs_list = []
    jimoons_list = []
    jimoon_names = []  # string
    prob_names = []  # string

    y_top_padding = 7  # fix pdfplumber position error

    x0,x1,x2,y0,y1 = 0,0,0,0,0

    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            img = imgs[i]
            h, w = img.shape
            words = page.extract_words()

            # jimoon
            jimoon_words = find_jimoon_words(words)
            jimoons = []
            for jw in jimoon_words:
                jx0, jy0, jx1, jy1 = get_xy(jw, page.width, page.height)
                xys = jx0, jy0, jx1, jy1 = int(jx0 * w), int(jy0 * h) - y_top_padding, int(jx1 * w), int(jy1 * h)

                # replace [p0~p1] to [p0-p1]
                jimoon_name = jw['text'].split('[')[1].split(']')[0]
                jimoon_name = jimoon_name.replace('～', '-')
                jimoon_name = jimoon_name.replace('~', '-')

                jimoons.append((xys, jimoon_name))
            # print(jimoons)
            # print(jimoon_words)

            # problem
            prob_words = find_problem_words(words)
            probs = []
            for pw in prob_words:
                px0, py0, px1, py1 = get_xy(pw, page.width, page.height)
                xys = px0, py0, px1, py1 = int(px0 * w), int(py0 * h) - y_top_padding, int(px1 * w), int(py1 * h)

                probs.append((xys, pw['text'].split('.')[0]))
            # print(probs)
            # print(prob_words)

            top_hline, bottom_hline = get_top_bottom_horizontal_lines(img)
            ymin = top_hline[1]
            ymax = bottom_hline[1]
            xs = (top_hline[0], top_hline[2], bottom_hline[0], bottom_hline[2])
            xmin, xmax = min(xs), max(xs)

            # 수평선이 남아있는 것을 방지
            ymin += 7
            ymax -= 7

            blocks.append(img[ymin:ymax, xmin:xmax])

            probs_list.append([])
            for prob in probs:
                (px0, py0, px1, py1), name = prob
                if xmin <= px0 < xmax:
                    prob = px0 - xmin, py0 - ymin, px1 - xmin, y1 - ymin
                    probs_list[-1].append(prob)
                    prob_names.append(name)

            jimoons_list.append([])
            for jimoon in jimoons:
                (jx0, jy0, jx1, jy1), name = jimoon
                if xmin <= jx0 < xmax:
                    jimoon = jx0 - xmin, jy0 - ymin, jx1 - xmin, jy1 - ymin
                    jimoons_list[-1].append(jimoon)
                    jimoon_names.append(name)

    return blocks, jimoons_list, jimoon_names, probs_list, prob_names


def save_ebs_solution(pdf_path, long_block, jimoons, jimoon_names, probs, prob_names, contour_ends, output_dir):
        '''
            preresquite : run make_output_dir() method before run this method
            description : split long block into jimoons and probs and save it into already maden output dir
            save jimoon and prob according to test year's specific problem structure.
            refer to make_output_dir
            split algorithm is implemented by 3 elements (jimoons, probs, contour_end)
        '''
        test_name = os.path.basename(pdf_path).split('.')[0]

        output_dir = Path(output_dir)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # make elements list with jimoons' y0, probs' y0, contour ends' y value
        # elements : list((y, category, name))
        # element category : jimoon - 0, prob - 1, contour end -
        # element name : ex) jimoon - "11-13", prob - "12", contour end - ""
        elements = []

        for i, (_, y, _, _) in enumerate(jimoons):
            if y < 0:
                y = 0
            name = jimoon_names[i]
            elements.append((y, 0, name))

        # problem이 아닌 것들 처리
        problem_dict = defaultdict(bool)
        problem_index = 1
        for i, (_, y, _, _) in enumerate(probs):
            if y < 0:
                y = 0

            name = prob_names[i]

            for k in reversed(range(len(name))):
                if not name[k].isdigit():
                    name = name[k + 1:]
                    break

            # ebs해설지에서 정답으로 한번 낳온 후에 해설이 나옴
            if not problem_dict[int(name)]:
                problem_dict[int(name)] = True
                continue

            # 문제 번호와 맞지 않는 경우 제외
            if name != str(problem_index):
                continue

            problem_index += 1

            elements.append((y, 1, name))

        for y in contour_ends:
            if y < 0:
                y = 0
            elements.append((y, 2, ''))

        elements.sort()

        # 마지막 문제를 살리기 위해 추가
        long_height = long_block.shape[0]
        elements.append((long_height, 2, ''))

        tmp = test_name.split('_')
        year = int(tmp[0])
        month = int(tmp[1])

        # split and save
        last_jimoon = "0-0"  # check if problem is in lastest jimoon
        jimoon_index = 1
        for i in range(len(elements) - 1):
            y0, cat, element_name = elements[i]
            y1, ncat, nen = elements[i + 1]

            # 지문해설은 번호가 반복적으로 나오는 경우가 있음 (생략)
            if cat == ncat == 0:
                continue
            part = long_block[y0:y1, :]

            # jimoon
            if cat == 0:
                file_name = '_'.join([test_name, "p" + str(jimoon_index), "solution"]) + ".png"
                last_jimoon = element_name
                jimoon_index += 1

            # problem
            elif cat == 1:

                file_name = '_'.join([test_name, element_name, "solution"]) + ".png"
                # if int(last_jimoon.split('-')[1]) < int(element_name):  # not in last jimoon
                # else:  # concluded in last jimoon

            else:
                continue

            print(f'save {element_name}')
            print(f'next element : {nen}')
            try:
                cv2.imwrite(str(output_dir / file_name), cut_bottom(part))
            except Exception as e:
                print(e)
                print(file_name, part.shape)
                print(f"y0 : {y0}, y1 : {y1}")
                print(f"{cat}, {element_name}, {ncat}, {nen}")