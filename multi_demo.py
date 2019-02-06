import argparse
import os
import time

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import HyperLPRLite as pr

fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)


def drawRectBox(image1, rect1, addText):
    cv2.rectangle(image1, (int(rect1[0]), int(rect1[1])), (int(rect1[0] + rect1[2]), int(rect1[1] + rect1[3])),
                  (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(image1, (int(rect1[0] - 1), int(rect1[1]) - 16), (int(rect1[0] + 115), int(rect1[1])),
                  (0, 0, 255), -1, cv2.LINE_AA)
    img = Image.fromarray(image1)
    draw = ImageDraw.Draw(img)
    draw.text([int(rect1[0] + 1), int(rect1[1] - 16)], addText, (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Multiple rec demo')

    parser.add_argument('--detect_parent_path', action='store', dest='detect_parent_path')
    parser.add_argument('--cascade_model_path', action='store', default='model/cascade.xml')
    parser.add_argument('--mapping_vertical_model_path', action='store', default='model/model12.h5')
    parser.add_argument('--ocr_plate_model_path', action='store', default='model/ocr_plate_all_gru.h5')
    parser.add_argument('--save_result_flag', action='store', default='True')
    parser.add_argument('--plot_result_flag', action='store', default='True')
    parser.add_argument('--save_path', action='store', default=None)

    args = parser.parse_args()

    model = pr.LPR(args.cascade_model_path, args.mapping_vertical_model_path, args.ocr_plate_model_path)

    for filename in os.listdir(args.detect_parent_path):
        path = os.path.join(args.detect_parent_path, filename)
        if path.endswith(".jpg") or path.endswith(".png"):
            grr = cv2.imread(path)
            t0 = time.time()
            image = grr
            for pstr, confidence, rect in model.SimpleRecognizePlateByE2E(grr):
                if confidence > 0.7:
                    image = drawRectBox(image, rect, pstr + " " + str(round(confidence, 3)))
                    print("plate_str:")
                    print(pstr)
                    print("plate_confidence")
                    print(confidence)
            t = time.time() - t0
            print("Image size :" + str(grr.shape[1]) + "x" + str(grr.shape[0]) + " need " + str(
                round(t * 1000, 2)) + "ms")

            if args.plot_result_flag == 'True' or args.plot_result_flag == 'true':
                cv2.imshow("image", image)
                cv2.waitKey(500)
            if args.save_result_flag == 'True' or args.save_result_flag == 'true':
                (filepath, tempfilename) = os.path.split(filename)
                (name, extension) = os.path.splitext(tempfilename)
                cv2.imwrite(args.save_path + name + ".png", image)
