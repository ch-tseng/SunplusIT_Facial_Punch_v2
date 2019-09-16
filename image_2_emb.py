#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os, sys
import cv2
import numpy as np
import imutils
from libFaceDoor import ov_FaceDect
from libFaceDoor import ov_FaceRecognize
from libFaceDoor import ov_Face5Landmarks
from PIL import ImageFont, ImageDraw, Image
from sklearn.externals import joblib

image_waiting_path = "/media/pi/KINGSTON/embs/uploads"
path_emb_output = "/media/pi/KINGSTON/embs/employees"

image_list = [()]

if not os.path.exists(path_emb_output):
    os.makedirs(path_emb_output)
if not os.path.exists(image_waiting_path):
    print("Cannot find the folder:", image_waiting_path)
    sys.exit(1)

if __name__ == '__main__':
    FACE_Detect = ov_FaceDect("models/face-detection-adas-0001.bin", "models/face-detection-adas-0001.xml")
    FACE_RECOG = ov_FaceRecognize("models/face-reidentification-retail-0095.bin","models/face-reidentification-retail-0095.xml")
    faceLandmarks = ov_Face5Landmarks(bin_path="models/landmarks-regression-retail-0009.bin", xml_path="models/landmarks-regression-retail-0009.xml")

    for file in os.listdir(image_waiting_path):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
            file_path = os.path.join(image_waiting_path, file)
            image = cv2.imread(file_path)

            face_boxes, face_scores = FACE_Detect.detect_face(image, score=0.5, target_device=cv2.dnn.DNN_TARGET_MYRIAD)
            if(len(face_boxes)>0):
                max_face_area = 0
                cx, cy, ww, hh, xx, yy = 0, 0, 0, 0, 0, 0
                for i, (x,y,w,h) in enumerate(face_boxes):
                    if( w*h > max_face_area):
                        max_face_area = w*h
                        cx, cy, ww, hh, xx, yy = (x+w/2), (y+h/2), w, h, x, y

                if(max_face_area>0):
                    face_ref = image[yy:yy+hh, xx:xx+ww]
                    rotated_img = faceLandmarks.align_face(face_ref, (xx, yy, ww, hh), image)

                    face_boxes2, face_scores2 = FACE_Detect.detect_face(rotated_img, score=0.5, target_device=cv2.dnn.DNN_TARGET_MYRIAD)
                    if(len(face_boxes2)>0):
                        max_face_area = 0
                        cx, cy, ww, hh, xx, yy = 0, 0, 0, 0, 0, 0
                        for i, (x,y,w,h) in enumerate(face_boxes2):
                            if( w*h > max_face_area):
                                max_face_area = w*h
                                cx, cy, ww, hh, xx, yy = (x+w/2), (y+h/2), w, h, x, y

                        face_ref = rotated_img[yy:yy+hh, xx:xx+ww]
                        if(max_face_area>0 and xx*yy>0):
                            face_embs = FACE_RECOG.detect_face(face_ref, target_device=cv2.dnn.DNN_TARGET_MYRIAD)

                            cv2.imwrite(os.path.join(path_emb_output, filename+".png"), face_ref)
                            joblib.dump(face_embs, os.path.join(path_emb_output, filename+".embs"))
                            #os.remove(file_path)
                            print(os.path.join(path_emb_output, filename+".embs"))
