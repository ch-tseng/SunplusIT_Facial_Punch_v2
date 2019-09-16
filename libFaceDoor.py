import time, datetime
import imutils
import cv2
import math
import numpy as np
import serial
import socket  # Import socket module
import paho.mqtt.client as mqtt

class webCam:
    def __init__(self, id=0, videofile="", size=(1920, 1080)):
        self.camsize = size
        #for FPS count
        self.start_time = time.time()
        self.last_time = time.time()
        self.total_frames = 0
        self.last_frames = 0
        self.fps = 0

        if(len(videofile)>0):
            self.cam = cv2.VideoCapture(videofile)
            self.playvideo = True
        else:
            self.cam = cv2.VideoCapture(id)
            #self.cam = cv2.VideoCapture(cv2.CAP_DSHOW+id)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            self.playvideo = False

    def fps_count(self, seconds_fps=10):
        fps = self.fps

        timenow = time.time()
        if(timenow - self.last_time)>seconds_fps:
            fps  = (self.total_frames - self.last_frames) / (timenow - self.last_time)
            self.last_frames = self.total_frames
            self.last_time = timenow
            self.fps = fps

        return round(fps,2)

    def working(self):
        webCam = self.cam
        if(webCam.isOpened() is True):
            return True
        else:
            if(self.playvideo is True):
                return True
            else:
                return False

    def camRealSize(self):
        webcam = self.cam
        width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(webcam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def getFrame(self, rotate=0, vflip=False, hflip=False, resize=None):
        webcam = self.cam
        hasFrame, frame = webcam.read()
        if(frame is not None):
            if(vflip==True):
                frame = cv2.flip(frame, 0)
            if(hflip==True):
                frame = cv2.flip(frame, 1)
    
            if(rotate>0):
                frame = imutils.rotate_bound(frame, rotate)
            if(resize is not None):
                frame_resized = cv2.resize(frame, resize, interpolation=cv2.INTER_CUBIC)
            else:
                frame_resized = None

        else:
            frame = None
            hasFrame = False
            frame_resized = None

        self.total_frames += 1


        return hasFrame, frame_resized, frame

    def release(self):
        webcam = self.cam
        webcam.release()

#--------------------------------------------------------#
# 1. wget --no-check-certificate https://download.01.org/opencv/2019/open_model_zoo/R1/models_bin/face-detection-adas-0001/FP16/face-detection-adas-0001.bin
# 2. wget --no-check-certificate https://download.01.org/opencv/2019/open_model_zoo/R1/models_bin/face-detection-adas-0001/FP16/face-detection-adas-0001.xml
#
class ov_FaceDect:
    def __init__(self, bin_path, xml_path):
        #load the model
        net = cv2.dnn.readNet(xml_path, bin_path)
        self.net = net

    def detect_face(self, frame, score=0.5, target_device=cv2.dnn.DNN_TARGET_MYRIAD):
        net = self.net
        # Specify target device.
        net.setPreferableTarget(target_device)

        #Prepare input blob and perform an inference.
        blob = cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U)
        net.setInput(blob)
        out = net.forward()

        faces, scores = [], []
        for detection in out.reshape(-1, 7):
            confidence = float(detection[2])
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])

            if confidence > score:
                faces.append([xmin,ymin,xmax-xmin,ymax-ymin])
                scores.append(confidence)

        return faces, scores

class ov_FaceRecognize:
    def __init__(self, bin_path, xml_path):
        #load the model
        net = cv2.dnn.readNet(xml_path, bin_path)
        self.net = net

    def detect_face(self, face_img, target_device=cv2.dnn.DNN_TARGET_MYRIAD):
        net = self.net
        # Specify target device.
        net.setPreferableTarget(target_device)

        #Prepare input blob and perform an inference.
        try:
            blob = cv2.dnn.blobFromImage(face_img, size=(128, 128), ddepth=cv2.CV_8U)
            net.setInput(blob)
            out = net.forward()
        except:
            out = None

        return out

class mqttFACE():
    def __init__(self, host, chnl, portnum):
        self.host = host
        self.channel = chnl
        self.port = portnum

    def sendMQTT(self, msg):
        mqttc = mqtt.Client("Face-Checkin")
        mqttc.username_pw_set("chtseng", "chtseng")
        mqttc.connect(self.host, self.port)
        mqttc.publish(self.channel, msg)

class ov_Face5Landmarks:
    def __init__(self, bin_path, xml_path):
        #load the model
        net = cv2.dnn.readNet(xml_path, bin_path)
        self.net = net

    def getLandmarks(self, face_img, target_device=cv2.dnn.DNN_TARGET_MYRIAD):
        net = self.net
        # Specify target device.
        net.setPreferableTarget(target_device)

        #Prepare input blob and perform an inference.
        try:
            blob = cv2.dnn.blobFromImage(face_img, size=(48, 48), ddepth=cv2.CV_8U)
            net.setInput(blob)
            out = net.forward()
        except:
            out = None

        return out

    def renderFace(self, im, landmarks, color=(0, 255, 0), radius=5):
        for p in landmarks:
            cv2.circle(im, (p[0], p[1]), radius, color, -1)

        return im

    def angle_2_points(self, p1, p2):
        r_angle = math.atan2(p1[1] - p2[1], p1[0] - p2[0])
        rotate_angle = r_angle * 180 / math.pi

        return rotate_angle

    def area_expand(self, bbox, ratio):
        ew = int(bbox[3] * ratio)
        eh = int(bbox[2] * ratio)
        nx = int(bbox[0] - ((ew - bbox[2]) / 2))
        ny = int(bbox[1] - ((eh - bbox[3]) / 2))
        if(nx<0):
            nx = 0
        if(ny<0):
            ny = 0

        return (nx,ny,ew,eh)

    def align_face(self, face, face_bbox, orgImage):  #臉, bbox, 原圖片
        face_area_align = None
        points = self.getLandmarks(face, target_device=cv2.dnn.DNN_TARGET_MYRIAD)

        if(points is not None):
            landmarks = points[0]
            #print(landmarks)
            reye = (landmarks[0][0], landmarks[1][0])
            leye = (landmarks[2][0], landmarks[3][0])

            rotate_angle = self.angle_2_points(leye, reye)

            (xx,yy,ww,hh) = face_bbox
            (nx, ny, ew, eh) = self.area_expand((xx,yy,ww,hh), 1.5)
            face_area_expand = orgImage[ny:ny+eh, nx:nx+ew]

            face_area_align = imutils.rotate_bound(face_area_expand, -rotate_angle)

        return face_area_align

