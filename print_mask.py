import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt


def print_mask(x1, y1, x2, y2, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    x1 = int(x1 * w) 
    y1 = int(y1 * h)
    x2 = int(x2 * w)
    y2 = int(y2 * h)
    mask[y1:y2, x1:x2] = 255
    return mask

def xyhw2xyxy(*xywh):
    x, y, w, h = xywh
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return [x1, y1, x2, y2]


if __name__ == '__main__':
    x1, y1, x2, y2 = xyhw2xyxy(0.6, 0.4, 0.2, 0.35)
    mask = print_mask(x1, y1, x2, y2, 512, 768)
    img = Image.fromarray(np.uint8(mask))
    img.save('mask/robot.png')