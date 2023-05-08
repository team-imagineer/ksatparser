import os
import numpy as np
import cv2
from pathlib import Path
import pdfplumber
from .utils import pdf2pngs, get_contour_ends, get_xy, find_jimoon_words, find_problem_words, get_min_max_x, cut_bottom


def parse_problem(pdf_path, output_dir):
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

    # canny preprocessing
    thresh1 = 200
    thresh2 = 20
    dst = cv2.Canny(imgs[0], thresh1, thresh2)

    # problematic hough
    lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 100, 2)

    T = extract_T(lines)
    x0, _, x2, _ = T[1]

    # jimoon box width

    jbox_width = int(((x2 - x0) / 2) * 0.85)  # T의 width의 절반의 0.85배 이상을 jimoon box 의 width라고 봄

    blocks, jimoons_list, jimoon_names, probs_list, prob_names = get_elements_list(pdf_path, imgs, jbox_width)


    # for test
    for b in blocks:
        print(b.shape)

    # cut blocks list
    min_width = min([blocks[i].shape[1] for i in range(len(blocks))])
    for i in range(len(blocks)):
        blocks[i] = blocks[i][:, :min_width]

    # concatenate
    long_block = np.concatenate(blocks)

    # contour ends (공백 부분을 찾는 함수)
    vertical_height = blocks[3].shape[0]
    print(vertical_height)
    contour_ends = get_contour_ends(long_block, vertical_height)

    # concatenate jimoons_list, probs_list
    add_height = 0
    jimoons = []
    tmp_prob_names = []  # to prevent not problem names are included. ex) 보기문의 문장 안에 "3."
    probs = []
    prob_idx = -1

    for i, block in enumerate(blocks):

        for x0, y0, x1, y1 in jimoons_list[i]:
            y0 += add_height
            y1 += add_height
            jimoons.append((x0, y0, x1, y1))

        for x0, y0, x1, y1 in probs_list[i]:
            prob_idx += 1
            # print(prob_names[prob_idx])
            if x0 >= 70:  # delete not a problem
                continue
            y0 += add_height
            y1 += add_height
            probs.append((x0, y0, x1, y1))
            tmp_prob_names.append(prob_names[prob_idx])
            # print(prob_names[prob_idx])

        add_height += block.shape[0]

    # prob_names change
    prob_names = tmp_prob_names
    print(len(probs))
    print(prob_names)

    save_problems(pdf_path, long_block, jimoons, jimoon_names, probs, prob_names, contour_ends, output_dir)

    print(f"parse {pdf_path} complete")


def extract_T(lines):
    """
    Args:
        lines:
        xmid: 원래 vertical line의 x값

    Returns:(max length vertical line, max length horizontal line
    """
    xlen = []
    ylen = []
    maxx = 0
    maxy = 0
    for i, line in enumerate(lines):
        x1,y1,x2,y2 = line[0]
        xlen.append(abs(x2-x1))
        ylen.append(abs(y2-y1))
    xi = xlen.index(max(xlen))
    yi = ylen.index(max(ylen))
    return lines[yi][0], lines[xi][0]


def get_elements_list(pdf_path, imgs, jbox_width):
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
            # print("problem!!!@")

            # problem
            prob_words = find_problem_words(words)
            probs = []
            for pw in prob_words:
                px0, py0, px1, py1 = get_xy(pw, page.width, page.height)
                xys = px0, py0, px1, py1 = int(px0 * w), int(py0 * h) - y_top_padding, int(px1 * w), int(py1 * h)

                probs.append((xys, pw['text'].split('.')[0]))
            # print(probs)
            # print(prob_words)

                # divide left and right
            # canny preprocess
            thresh1 = 500
            thresh2 = 300
            dst = cv2.Canny(img, thresh1, thresh2)

            # problematic hough
            lines = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 100, 1)

            # extract T
            vertline, horline = extract_T(lines)

            if i==0:
                x1, y0, _, y1 = vertline # x1은 첫페이지에서 고정
                if y0 > y1:
                    y0, y1 = y1, y0
            _, tmp_y0, _, tmp_y1 = vertline
            y1 = max(tmp_y0, tmp_y1)
            _, y0, _, _ = horline

            x0,x2 = get_min_max_x(img)

            # T가 없는 페이지는 제외
            if x2-x0 < 2*jbox_width:
                continue

            # extract jbox candidates
            jbox_candidates = []
            for l in lines:
                l = l[0]
                if abs(l[2] - l[0]) >= jbox_width:
                    overlap = False
                    for ll in jbox_candidates:
                        if abs(ll[1] - l[1]) < 6:
                            overlap = True
                    if not overlap:
                        jbox_candidates.append(l)

            # T preprocess (dilate spaces)
            x0 -= 20
            x2 += 20
            y0 += 7
            mid_margin = 7

            # divide left&right and preprocess
            for xmin, ymin, xmax, ymax in [(x0, y0, x1 - mid_margin, y1), (x1 + mid_margin, y0, x2, y1)]:
                blocks.append(img[ymin:ymax, xmin:xmax])

                probs_list.append([])
                for prob in probs:
                    (px0, py0, px1, py1), name = prob
                    if xmin <= px0 < xmax:
                        prob = px0 - xmin, max(py0 - ymin,0), px1 - xmin, y1 - ymin
                        probs_list[-1].append(prob)
                        prob_names.append(name)

                jimoons_list.append([])
                for jimoon in jimoons:
                    (jx0, jy0, jx1, jy1), name = jimoon
                    if xmin <= jx0 < xmax:
                        jimoon = jx0 - xmin, max(jy0 - ymin,0), jx1 - xmin, jy1 - ymin # 지문 에러 제거 (block 바깥으로 좌표가 넘어가는 경우)
                        jimoons_list[-1].append(jimoon)
                        jimoon_names.append(name)
    return blocks, jimoons_list, jimoon_names, probs_list, prob_names


def save_problems(pdf_path, long_block, jimoons, jimoon_names, probs, prob_names, contour_ends, output_dir):
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
    # element category : jimoon - 0, prob - 1, contour end - 2
    # element name : ex) jimoon - "11-13", prob - "12", contour end - ""
    elements = []

    for i, (_, y, _, _) in enumerate(jimoons):
        if y<0:
            y = 0
        name = jimoon_names[i]
        elements.append((y, 0, name))

    # problem이 아닌 것들 처리
    problem_index = 1
    for i, (_, y, _, _) in enumerate(probs):
        if y<0:
            y = 0
        name = prob_names[i]

        for k in reversed(range(len(name))):
            if not name[k].isdigit():
                name = name[k + 1:]
                break

        # 문제 번호와 맞지 않는 경우 제외
        if name != str(problem_index):
            continue

        problem_index += 1

        elements.append((y, 1, name))

    for y in contour_ends:
        if y<0:
            y = 0
        elements.append((y, 2, ''))

    elements.sort()

    # 마지막 문제를 살리기 위해 추가
    long_height = long_block.shape[0]
    elements.append((long_height,2,''))

    tmp = test_name.split('_')

    # split and save
    last_jimoon = "0-0"  # check if problem is in lastest jimoon
    jimoon_index = 1
    for i in range(len(elements) - 1):
        y0, cat, element_name = elements[i]
        y1, ct, en = elements[i + 1]

        part = long_block[y0:y1, :]

        # jimoon
        if cat == 0:
            file_name = '_'.join([test_name, "p"+str(jimoon_index)]) + ".png"
            last_jimoon = element_name
            jimoon_index += 1

        # problem
        elif cat == 1:

            file_name = '_'.join([test_name, element_name]) + ".png"
            # if int(last_jimoon.split('-')[1]) < int(element_name):  # not in last jimoon
            # else:  # concluded in last jimoon

        else:
            continue

        print(f'save {element_name}')
        print(f'next element : {en}')
        try:
            cv2.imwrite(str(output_dir / file_name), cut_bottom(part))
        except Exception as e:
            print(e)
            print(file_name, part.shape)
            print(f"y0 : {y0}, y1 : {y1}")
            print(f"{cat}, {element_name}, {ct}, {en}")

    # save long block
    cv2.imwrite(str(output_dir / ('long_' + test_name + '.png')), long_block)