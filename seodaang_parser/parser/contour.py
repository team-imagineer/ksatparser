import cv2
import numpy as np


from seodaang_parser.config import Config


def get_problem(img: np.ndarray, cfg: Config):
    # pre-processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, cfg.threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((cfg.kernel.size[0], cfg.kernel.size[1]), np.uint8)
    iterations = cfg.iterations

    # stackoverflow: merging-regions-in-mser-for-identifying-text-lines-in-ocr
    thresh = cv2.erode(thresh, kernel, iterations=iterations)
    thresh = cv2.dilate(thresh, kernel, iterations=iterations)

    # for cv2.RETR_EXTERNAL
    thresh = cv2.bitwise_not(thresh)

    # find the contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort y-axis
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[1])

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > cfg.min_width and h > cfg.min_height:
            roi = img[y : y + h, x : w + w]
            if roi.size != 0:
                cv2.imshow("roi", roi)
                cv2.waitKey()
                cv2.destroyAllWindows()
