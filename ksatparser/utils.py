from pdf2image import convert_from_path
from pathlib import Path
import tempfile
import cv2
import numpy as np
import matplotlib.pyplot as plt


def imshow(img, figsize=(20,30)):
    fig, ax= plt.subplots(1,1, figsize=figsize)
    ax.imshow(img, cmap = "gray")


def pdf2pngs(pdf_path):
    """
    Description:
    pdf를 이미지로 저장하고, 저장한 이미지를 cv2 imread 함수로 읽어 numpy array 로 반환
    Args:
        pdf_path: str, Path 타입의 pdf 경로

    Returns:
        list(numpy array)
    """
    # pdf to png
    pages = convert_from_path(str(pdf_path))
    pdf_len = len(pages)
    png_dir = Path(tempfile.mkdtemp()) # save in temp directory
    for i, page in enumerate(pages):
        png_path = png_dir / (str(i) + ".png")
        page.save(str(png_path), "PNG")

    # read png & Load the image
    pngs = []
    for i in range(pdf_len):
        png_path = str(png_dir / (str(i) + ".png"))
        img = cv2.imread(png_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pngs.append(gray)
    return pngs


def cut_bottom(part):
    # preprocessing
    _, thresh = cv2.threshold(part, 220, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_not(thresh)

    thresh_sum = np.sum(thresh, axis=1)
    cut_y = part.shape[0]
    bottom_padding = 7
    for i in reversed(range(thresh.shape[0])):
        value = thresh_sum[i]
        if value != 0:
            cut_y = i+bottom_padding
            break

    return part[:cut_y][:]



def get_contour_ends(long_block, vertical_height, kernel_width=15, kernel_height=25, iterations=5):
    """
    Descriptions:
        make contours
    Args:
        long_block: numpy array
        vertical_height: T에서 vertical line의 높이
        kernel_width: dilate/erode 할 때 kernel의 width
        kernel_height: dilate/erode 할 때 kernel의 height
        iterations: dilate/erode iteration 횟수

    Returns:
        contours' end y value
    """

    # pre-processing
    _, thresh = cv2.threshold(long_block, 220, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_not(thresh)

    kernel = np.ones((kernel_width, kernel_height))
    iterations = iterations

    thresh = cv2.dilate(thresh, kernel, iterations=iterations)
    thresh = cv2.erode(thresh, kernel, iterations=iterations)

    # find the contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort y-axis
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[1])

    # ends list
    ends = []

    height_rate = 0.6
    min_width, min_height = 250, vertical_height*height_rate
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w >= min_width and h > min_height:
            ends.append(int(y + h) + 4)
    return ends


def get_xy(word, width, height):
    """
    Descriptions:
        pdf_plumber에서 추출한 word의 위치를 비율로 바꿔서 변환
    Args:
        word:
        width:
        height:

    Returns:

    """
    x0 = word['x0'] / width
    y0 = word['top'] / height
    x1 = word['x1'] / width
    y1 = word['bottom'] / height
    return x0, y0, x1, y1


def find_problem_words(words):
    """
    Descriptions:
        pdf_plumber에서 추출한 word들에서 문제번호 word (1.) 만 골라서 반환
    Args:
        words:
    Returns:
    """
    probs = []
    digits = [str(i) for i in range(10)]
    for w in words:
        for d in digits:
            if d in w:
                print(w)

    for w in words:
        if '.' in w['text'] and w['text'].split('.')[0][-1].isdigit():
            probs.append(w)

    print(f"문제 word\n{probs}")
    return probs


def find_jimoon_words(words):
    """
    Discriptions:
        pdf_plumber에서 추출한 word들에서 지문번호 word ([1~3]) 만 골라서 반환
    Args:
        words:
    Returns:
    """
    ##############
    print('##########')
    for w in words:
        if '[' in w['text']:
            print(w['text'])
    print('#########')
    ###############
    jimoons = []
    for word in words:
        if ']' in word["text"] and '[' in word["text"]:
            inword = word['text'].split('[')[1].split(']')[0]
            print(inword)
            if ('～' in inword or '~' in inword or '∼' in inword or '-' in inword):
                jimoons.append(word)
            continue

        # 예외처리 (교육청) 2016_10 [20~ 23] - word가 20만 인식
        if '[' in word['text'] and (word['text'][-1]=='~' or word['text'][-1]=='～' or word['text'][-1]=='∼'):
            jimoons.append(word)
            continue
        if '[' in word['text'] and word['text'][-1].isdigit():
            jimoons.append(word)
            continue
    print(f"지문 word\n{jimoons}")
    return jimoons


def get_min_max_x(img):
    _, thresh = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
    width = thresh.shape[1]
    xmin, xmax = 0, thresh.shape[1]-1
    for i in range(width):
        if not all(thresh[:, i]):
            xmin = i
            break
    for i in reversed(range(width)):
        if not all(thresh[:, i]):
            xmax = i
            break
    return xmin, xmax