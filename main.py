#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os, sys
import signal
import logging
import random
import time
import math
from datetime import datetime
import cv2
import numpy as np
import imutils
from libFaceDoor import webCam
from libFaceDoor import ov_FaceDect
from libFaceDoor import ov_FaceRecognize
from libFaceDoor import mqttFACE
from libFaceDoor import ov_Face5Landmarks
from PIL import ImageFont, ImageDraw, Image
import multiprocessing as mp
from scipy.spatial import distance
from sklearn.externals import joblib
import requests
import RPi.GPIO as GPIO 
GPIO.setmode(GPIO.BCM)

#--Configurations for users ------------------------------------
# web camera
cam_id = 0
webcam_size = (1920, 1080) #(1920, 1080)
screen_size = (640, 360)  #size for video frame
lcd_size = (800, 480)
cam_rotate = 0
flip_vertical = False
flip_horizontal = True
face_detect_interval = 3  #frames
rectangle_detected_face = False
rectangle_face_detect_area = True

th_score = 15.9
#th_score = 15.2  #需小於此值才能認定為此人
th_score_special = { "200361":15.5, "200012":15.2, "200316": 15.6, "200094":16.1, "200281":15.7, "200181":15.5, "200280":15.2  }

onlyWorkDay = True
notWorkDay = [ "2/4", "2/5", "2/6", "2/7", "2/8", "2/28", "3/1", "4/4", "4/5", "5/1", "6/7", "9/13", "10/10", "10/11" ]

#--Configurations for developers -------------------------------
#for Demo ir debug, use video to replace camera
simulate = ""
appStatus = True   # the status for this program, if False then exit
win_name = "Face-Punch"  #cv2 window name
reference_size = screen_size  # frame will resized for inference
pinLight = 18 #GPIO
GPIO.setup(pinLight, GPIO.OUT)
GPIO.output(pinLight,GPIO.LOW)
pinButton = 15 #GPIO
GPIO.setup(pinButton, GPIO.IN, pull_up_down=GPIO.PUD_UP)
full_screen = True
min_light_time = 10  #power on the lighr at least for x seconds
#Recognize area
center_area_x = [(screen_size[0]/2) - 120, (screen_size[0]/2) + 120]
center_area_y = [(screen_size[1]/2) - 100, (screen_size[1]/2) + 100]

#Ratio of Target face size on screen
# * screen_size
ratio_face_rect = 0.4
#threshold_clear = 400  # higher is more clear, lower is more blurly for the photo
#path for face embs saved (each time picture was taken)
path_embs_store = "/media/pi/3A72-2DE1/embs"
path_embs_history = "history"
path_embs_employees = "employees"

toWebserver = "/home/pi/works/fdoor.openvino/web/static/"
posturl="http://api.sunplusit.com/api/DoorFaceDetection"
logging.basicConfig(level=logging.INFO, filename='/home/pi/works/fdoor.openvino/logging.txt')

wav_path = "/home/pi/works/fdoor.openvino/wav/"

#used for adding new employees
image_waiting_path = "/media/pi/3A72-2DE1/embs/uploads"
path_emb_output = "/media/pi/3A72-2DE1/embs/employees"

#--Used in application
#---------------------------------------------------------------
#Face dected list
#win_name = "FRAME"

list_preview_faces = []
today_punch_list = {}
today_now = datetime.today().hour
face_bar_left_xy = (670, 110)
preview_listbar_size = (lcd_size[0]-face_bar_left_xy[0]-20, lcd_size[1]-face_bar_left_xy[1]-20)
max_preview_faces = int(lcd_size[1]/preview_listbar_size[0]) + 2  #max numbers for faces in the bar
face_list_bar = None   #bar image
img_resize_ratio = (screen_size[0]/webcam_size[0], screen_size[1]/webcam_size[1])

sleep_mode = False   #blank screen
blank_screen = np.zeros((lcd_size[1], lcd_size[0], 3), np.uint8)

def init_env():
    if not os.path.exists(os.path.join(path_embs_store, path_embs_history)):
        os.makedirs(os.path.join(path_embs_store, path_embs_history))

    if(full_screen is True):
        cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)        # Create a named window
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)


    if not os.path.exists(os.path.join(toWebserver, "fail")):
        os.makedirs(os.path.join(toWebserver, "fail"))

    if not os.path.exists(os.path.join(toWebserver, "pass")):
        os.makedirs(os.path.join(toWebserver, "pass"))

def play_wav(uid):
    if(uid in today_punch_list):
        if(today_punch_list[uid]==1):
            file =  os.path.join(wav_path,uid) + '.wav'
            if(os.path.exists(file)):
                os.system('/usr/bin/aplay ' + file)

def chkWorkDay():
    if(onlyWorkDay == True):
        today = datetime.today()
        now = datetime.now()
        month = today.month
        day = today.day
        weekDay = today.weekday() + 1

        if( weekDay==6 or weekDay==7) or (str(month)+"/"+str(day) in notWorkDay):
            return False
        else:
            if(now.hour<23 and now.hour>=6):
                return True
            else:
                return False
    else:
        return True

def printText(txt, bg, color=(0,255,0,0), size=0.7, pos=(0,0), type="Chinese"):
    (b,g,r,a) = color

    if(type=="English"):
        ## Use cv2.FONT_HERSHEY_XXX to write English.
        cv2.putText(bg,  txt, pos, cv2.FONT_HERSHEY_SIMPLEX, size,  (b,g,r), 2, cv2.LINE_AA)

    else:
        ## Use simsum.ttf to write Chinese.
        fontpath = "/home/pi/works/fdoor.openvino/wt009.ttf"
        #print("TEST", txt)
        font = ImageFont.truetype(fontpath, int(size*20))
        img_pil = Image.fromarray(bg)
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos,  txt, font = font, fill = (b, g, r, a))
        bg = np.array(img_pil)

    return bg

def desktop_bg():
    img_desktop = np.zeros((lcd_size[1], lcd_size[0], 3), np.uint8)
    #print("LOGO:", logo.shape, "LCD:", img_desktop.shape)
    img_desktop[10:logo.shape[0]+10, 10:logo.shape[1]+10] = logo

    return img_desktop

def target_face_size(pic_size):
    w, h = pic_size[0], pic_size[1]
    face_h = int(h * 0.8)
    ratio = int(face_h/h)

def exit_app():
    pool_logging.close()
    pool_logging.join()

    print("End.....")
    sys.exit(0)

def signal_handler(sig):
    print('You pressed Ctrl+C!')
    exit_app()

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def light_poweron():
    global light_poweron_time, sleep_mode
    #print("run light poweron")
    GPIO.output(pinLight,GPIO.HIGH)
    light_poweron_time = time.time()
    sleep_mode = False
    #black_screen(sleepnow=False)

def light_auto_poweroff():
    #print("Time test:", light_poweron_time)
    #black_screen(sleepnow=True)
    global sleep_mode
    if(time.time()-light_poweron_time)>min_light_time:
        GPIO.output(pinLight,GPIO.LOW)
        sleep_mode = True

def calc_dist(face0, face1):
    try:
        face_0 = l2_normalize(np.concatenate(face0))
        face_1 = l2_normalize(np.concatenate(face1))
    except:
        logging.info("[error except] def calc_dist() error except")
        return None

    if (len(face_0) != len(face_1)):
        return None

    dist = distance.euclidean(face_0, face_1)
    #dist2 = np.linalg.norm(face_0-face_1, axis=1)

    return dist

def preview_bar(face_img, list_faces):
    #print("list_preview_faces 1:", len(list_faces))

    img_desktop = np.zeros((preview_listbar_size[1], preview_listbar_size[0], 3), np.uint8)

    while len(list_faces)>max_preview_faces:
        list_faces.pop(0)

    sx = 0
    sy = 10
    preview_h = 60  #height for the face
    interval_h = 10
    #print("list_preview_faces 2:", len(list_faces))
    for face in list_faces:
        try:
            face = imutils.resize(face, height=(preview_h-interval_h))
        except:
            continue
 
        w = face.shape[1]
        h = face.shape[0]
        img_desktop[sy:sy+h, sx:sx+w] = face
        sy += 6+face.shape[0]

    return img_desktop

def save_pics(img_face, img_org, embs, filename):
    #joblib.dump(embs, filename+'.embs')
    cv2.imwrite(filename+'_org.png', img_org)
    #cv2.imwrite(filename+'_face.png', img_face)

def save_face_embs(bbox, img_org, id):
    global list_preview_faces

    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    ox, oy, ow, oh = int(x/img_resize_ratio[0]), int(y/img_resize_ratio[1]), int(w/img_resize_ratio[0]), int(h/img_resize_ratio[1])

    #preview_face = img_prev[y:y+h, x:x+w]
    org_face = img_org[oy:oy+oh, ox:ox+ow]
    #cv2.imshow("TEST", preview_face)
    #cv2.imshow("TEST2", org_face)
    #cv2.waitKey(0)

    #Align the face
    rotated_img = faceLandmarks.align_face(org_face, (ox, oy, ow, oh), img_org)
    if(rotated_img is not None):
        face_boxes, face_scores = FACE_Detect.detect_face(rotated_img, score=0.5, target_device=cv2.dnn.DNN_TARGET_MYRIAD)
        cx, cy, ww, hh, xx, yy = 0, 0, 0, 0, 0, 0
        max_face_area = 0
        for i, (x,y,w,h) in enumerate(face_boxes):
            if( x*y>0 and w*h>max_face_area):
                max_face_area = w*h
                cx, cy, ww, hh, xx, yy = (x+w/2), (y+h/2), w, h, x, y

        if(max_face_area>0 and x*y>0):
            org_face = rotated_img[yy:yy+hh, xx:xx+ww]

    list_preview_faces.append(org_face)

    bar_img = None
    face_embs = FACE_RECOG.detect_face(org_face, target_device=cv2.dnn.DNN_TARGET_MYRIAD)
    if(face_embs is not None):
        hour_foler = time.strftime("%Y-%m-%d_%H", time.localtime())
        dump_path = os.path.join(path_embs_store, path_embs_history, hour_foler)
        if(not os.path.exists(dump_path)):
            os.makedirs(dump_path)

        filename_path = os.path.join(dump_path, time.strftime("%Y-%m-%d_%H%M%S", time.localtime()))
        filename_path = filename_path + '_' + str(id)

        r = pool_logging.apply_async(save_pics, (org_face, img_org, face_embs, filename_path, ))
        #pool_result.append(r)

        bar_img = preview_bar(org_face, list_preview_faces)

    return face_embs, org_face, bar_img

def load_embs_memory():
    all_embs = []

    embs_path = os.path.join(path_embs_store, path_embs_employees)
    for id, file in enumerate(os.listdir(embs_path)):
        file_path = os.path.join(path_embs_store, path_embs_employees, file)

        if(os.path.isfile(file_path)):
            filename, file_extension = os.path.splitext(file)
            if(file_extension.lower() == '.embs'):
                 try:
                      all_embs.append((filename, joblib.load(file_path)) )

                 except:
                      logging.info("[error except] load_embs_memory() --> Error on loading embs: {}".format(file_path))
                      pass

    return all_embs

def compare_embs(sb_emb):
    min_score = 999
    data_people = "999999_unknow"
    for id, (nameData, emb) in enumerate(embsALL):
        cac_emb = calc_dist(emb, sb_emb)
        if(cac_emb is not None):
            #print("emb diff = ", cac_emb, nameData)
            if(cac_emb<min_score):
                min_score = cac_emb
                data_people = nameData

    #print(data_people, min_score)
    return (data_people, min_score)

def doorAction(openDoor, peopleID, camFace):
    id = str(peopleID)
    if(openDoor is True):
        faceMQTT.sendMQTT('{ "EmpCName":"facial-door", "DeptNo":"facial-door", "EmpNo":"'+str(peopleID)+'", "Uid":"'+str(peopleID)+'", "TagType":"E", "People":"1", "Time":"2019" }' )

        callWebServer(str(peopleID), camFace, openDoor)
        print("Door opened")

    else:
        callWebServer(str(peopleID), camFace, openDoor)
        print("Door open refuse")

def add_new_employees():
    global embsALL

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

                face_ref = image[yy:yy+hh, xx:xx+ww]

                face_embs = None
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

                if(face_embs is not None):
                    cv2.imwrite(os.path.join(path_emb_output, filename+".png"), face_ref)
                    joblib.dump(face_embs, os.path.join(path_emb_output, filename+".embs"))
                    os.remove(file_path)
                    logging.info("add new employee from upload to :{}".format(os.path.join(path_emb_output, filename+".embs")))
                else:
                    os.remove(file_path)
                    logging.info("add new employee failed, get no embs from image.")

    embsALL = load_embs_memory()

def clear_level(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clear_rate = cv2.Laplacian(gray, cv2.CV_64F).var()
    print("CLEAR:", clear_rate)
    if(clear_rate>threshold_clear):
        return True
    else:
        return False

def punch_update(uid):
    global today_now, today_punch_list

    print("punch:", today_punch_list)

    if(today_now != datetime.today().day):
        print("Punch list cleared!", today_now)
        today_punch_list = {}
        today_now = datetime.today().day

    if(uid in today_punch_list):
        num = today_punch_list[uid]
    else:
        num = 0

    today_punch_list.update( { uid:num+1} )


def play_welcome(uid):
    if(uid in today_punch_list):
        now = datetime.now()
        if(today_punch_list[uid]==1):
            if(now.hour<=10 and now.hour>4):
                os.system('/usr/bin/aplay /home/pi/works/fdoor.openvino/wav/goodmorning.wav')
            elif(now.hour<=15 and now.hour>10):
                os.system('/usr/bin/aplay /home/pi/works/fdoor.openvino/wav/goodafternoon.wav')
            elif(now.hour<=23 and now.hour>17):
                os.system('/usr/bin/aplay /home/pi/works/fdoor.openvino/wav/goodnight.wav')
            else:
                os.system('/usr/bin/aplay /home/pi/works/fdoor.openvino/wav/hello.wav')
        #else:
        #    os.system('/usr/bin/aplay /home/pi/works/fdoor.openvino/wav/hello.wav')

def callWebServer(id, pic, result):
    filename = str(time.time())

    if(result is True):
        folder = "pass"

    else:
        folder = "fail"

    path_write = os.path.join(toWebserver, folder, id + "_" + filename + "_cam.jpg")
    cv2.imwrite(path_write, pic)
    print("write to www folder:", path_write)
    url = "http://facial-door.sunplusit.com.tw:5000/"+folder+"/"+id+"_"+filename+"_cam.jpg"
    print("URL:", url)
    data= {
        'EmpNo': id,
        'FrontFace': url,
        'SideFace': '',
        'Detection': result
    }
    try:
        r = requests.post(posturl, data=data).text
    except requests.exceptions.RequestException as e:
        logging.info("[{}] {}".format(datetime.now(), e) )
        os.system('/usr/bin/aplay /home/pi/works/fdoor.openvino/wav/error_punch.wav')
        pass

if __name__ == '__main__':
    init_env()
    signal.signal(signal.SIGINT, signal_handler)
    frameID = 0

    #desktop_bg
    logo = cv2.imread("/home/pi/works/fdoor.openvino/logo.png")

    #multi-process
    pool_result = []
    pool_logging = mp.Pool(processes = 3)

    #web camera device
    CAMERA = webCam(id=cam_id, videofile=simulate, size=webcam_size)
    if(CAMERA.working() is False):
        print("webcam cannot work.")
        appStatus = False
        sys.exit()

    #face detection
    FACE_Detect = ov_FaceDect("/home/pi/works/fdoor.openvino/models/face-detection-adas-0001.bin", "/home/pi/works/fdoor.openvino/models/face-detection-adas-0001.xml")
    FACE_RECOG = ov_FaceRecognize("/home/pi/works/fdoor.openvino/models/face-reidentification-retail-0095.bin","/home/pi/works/fdoor.openvino/models/face-reidentification-retail-0095.xml")
    faceLandmarks = ov_Face5Landmarks(bin_path="models/landmarks-regression-retail-0009.bin", xml_path="models/landmarks-regression-retail-0009.xml")
    faceMQTT = mqttFACE("172.30.16.137", "Door-camera", 1883)

    nx, ny = 10, 75   #nx, ny is the left point for the video frame on LCD desktop
    #Draw a face target rectangle on the screen
    fh = int(screen_size[1] * ratio_face_rect)
    fw = int(fh * ratio_face_rect)
    light_poweron_time = 0
    pdata = []  #最近一次face detect成功的人員資料
    frame_id = 0
    last_punch_id = 0
    last_punch_time = time.time()
    #Draw a face target rectangle on the screen
    f_start_x, f_start_y = int((screen_size[0]/2)-(fw/2)), int((screen_size[1]/2)-(fh/2))

    embsALL = load_embs_memory()

    #先跑一下face detect, 避免第一次使用者用時, 會load很久
    print("hot run ......")
    hot_test = FACE_RECOG.detect_face(cv2.imread("/home/pi/works/fdoor.openvino/hot_test.png"), target_device=cv2.dnn.DNN_TARGET_MYRIAD)
    hot_test = None
    print("hot run finished...")
    while appStatus:

        #env check and setup
        light_auto_poweroff()

        hasFrame, frame_screen, frame_org = CAMERA.getFrame(rotate=cam_rotate, vflip=flip_vertical, hflip=flip_horizontal, resize=screen_size)
        frame_screen_org = frame_screen.copy()

        desktop = desktop_bg()

        btn_status = GPIO.input(pinButton)
        #print(btn_status)
        if(btn_status==0):
            light_poweron()
            add_new_employees()

        if(hasFrame is False):
            exit_app()

        face_ref = None
        door_pass = False
        if(frame_id % face_detect_interval == 0):
            frame_id = 0
            face_boxes, face_scores = FACE_Detect.detect_face(frame_screen, score=0.5, target_device=cv2.dnn.DNN_TARGET_MYRIAD)

            if(len(face_boxes)>0):
                #print(face_boxes, face_scores)
                light_poweron()
                #sleep_mode = False

                max_face_area = 0
                cx, cy, ww, hh, xx, yy = 0, 0, 0, 0, 0, 0
                for i, (x,y,w,h) in enumerate(face_boxes):
                    if(rectangle_detected_face):
                        cv2.ellipse(frame_screen, (int(x+(w/2)),int(y+(h/2))), (int(w/2), int(h/2)), 0, 0, 360, (0, 255, 0), 1)
                        cv2.rectangle(frame_screen, (x, y), (x+w, y+h), (0,255,0), 2)

                    if( w*h > max_face_area):
                        max_face_area = w*h
                        cx, cy, ww, hh, xx, yy = (x+w/2), (y+h/2), w, h, x, y
                #---> added for quickly display
                desktop[ny:ny+frame_screen.shape[0], nx:nx+frame_screen.shape[1]] = frame_screen
                if(face_list_bar is not None):
                    desktop[60:60+face_list_bar.shape[0], 680:680+face_list_bar.shape[1]] = face_list_bar

                #rect the detect area
                if(rectangle_face_detect_area):
                    if(face_ref is not None):
                        cv2.ellipse(frame_screen, (int(screen_size[0]/2), int(screen_size[1]/2)), (int(screen_size[1]*2/8), int(screen_size[1]*3/8)), 0, 0, 360, (0, 255, 255), 3)
                        #cv2.rectangle(frame_screen, (f_start_x, f_start_y),(f_start_x+fw, f_start_y+fh),(0,255,255),5)
                    else:
                        cv2.ellipse(frame_screen, (int(screen_size[0]/2), int(screen_size[1]/2)), (int(screen_size[1]*2/8), int(screen_size[1]*3/8)), 0, 0, 360, (0, 255, 0), 2)
                        #cv2.rectangle(frame_screen, (f_start_x, f_start_y),(f_start_x+fw, f_start_y+fh),(0,255,0),2)

                cv2.imshow(win_name, desktop)
                cv2.waitKey(1)
                #<--- end display
                face_ref = None
                if( (cx<center_area_x[1] and cx>center_area_x[0] and cy<center_area_y[1] and cy>center_area_y[0]) and (ww>32 and hh>32) ):
                    #face_ref = frame_screen_org[yy:yy+hh, xx:xx+ww]
                    #使用原尺寸較大的來取face area
                    #face_ref = frame_org[ int(yy/img_resize_ratio): int((yy+hh)/img_resize_ratio), int(xx/img_resize_ratio):int( int((xx+ww)/img_resize_ratio)) ]
                    #if(clear_level(face_ref) is False):
                    #    continue

                    #list_preview_faces.append(face_ref)
                    this_emb, face_ref, face_list_bar = save_face_embs((x,y,w,h), frame_org, i)
                    if((this_emb is None) or (face_list_bar is None)):
                        print("this_emb is None or face_list_bar is None, Ignore nad Continue.")
                        continue

                    name_data, recognize_scores = compare_embs(this_emb)
                    if(name_data is not None):
                        pdata = name_data.split('_')

                    #相同ID至少要超過X秒才能再刷卡
                    if(not (pdata[0]==last_punch_id and time.time()-last_punch_time<6)):
                        last_punch_id = pdata[0]
                        last_punch_time = time.time()

                        if(pdata[0] in th_score_special):
                            this_score = th_score_special[pdata[0]]
                        else:
                            this_score = th_score

                        if(recognize_scores<this_score):
                            if(chkWorkDay() is True):
                                punch_update(pdata[0])
                                logging.info("{}[{}] {} {}: embs diff:{} Door open:Yes".format(btn_status, datetime.now(), pdata[0], pdata[1], recognize_scores))
                                doorAction(openDoor=True, peopleID=pdata[0], camFace=frame_screen)
                                door_pass = True
                                play_welcome(pdata[0])
                                play_wav(pdata[0])
                            else:
                                logging.info("[{}] {} {}: embs diff:{} Door open:not working-day".format(datetime.now(), pdata[0], pdata[1], recognize_scores))
                                doorAction(openDoor=False, peopleID='000000', camFace=frame_screen)
                                os.system('/usr/bin/aplay /home/pi/works/fdoor.openvino/wav/workday.wav')
                        else:
                            pdata = []
                            logging.info("[{}] {} : embs diff:{} Door open:No".format(datetime.now(), pdata, recognize_scores))
                            doorAction(openDoor=False, peopleID='000000', camFace=frame_screen)

        if(face_list_bar is not None):
            desktop[60:60+face_list_bar.shape[0], 680:680+face_list_bar.shape[1]] = face_list_bar

        #rect the detect area
        if(rectangle_face_detect_area):
            if(face_ref is not None):
                cv2.ellipse(frame_screen, (int(screen_size[0]/2), int(screen_size[1]/2)), (int(screen_size[1]*2/8), int(screen_size[1]*3/8)), 0, 0, 360, (0, 255, 255), 3)
                #cv2.rectangle(frame_screen, (f_start_x, f_start_y),(f_start_x+fw, f_start_y+fh),(0,255,255),5)
            else:
                cv2.ellipse(frame_screen, (int(screen_size[0]/2), int(screen_size[1]/2)), (int(screen_size[1]*2/8), int(screen_size[1]*3/8)), 0, 0, 360, (0, 255, 0), 2)
                #cv2.rectangle(frame_screen, (f_start_x, f_start_y),(f_start_x+fw, f_start_y+fh),(0,255,0),2)

        #print(frame_screen.shape, desktop.shape)
        desktop[ny:ny+frame_screen.shape[0], nx:nx+frame_screen.shape[1]] = frame_screen
        if(len(pdata)>0 and door_pass is True):
            cname_display = ''.join([i for i in pdata[1] if not i.isdigit()])
            #display face on the screen
            face_ref = imutils.resize(face_ref, width=200)
            desktop = blank_screen.copy()
            desktop[10:10+face_ref.shape[0], 300:500] = face_ref
            desktop = printText(cname_display, desktop, color=(0,255,0,0), size=6, pos=(220,30+face_ref.shape[0]), type="Chinese")
            cv2.imshow(win_name, desktop)
            cv2.waitKey(1)
            time.sleep(1.5)

        if(sleep_mode is True):
            cv2.imshow(win_name, blank_screen)
        else:
            cv2.imshow(win_name, desktop)

        frame_id += 1

        key = cv2.waitKey(1)
        if(key==113):
            exit_app()
        #print("FPS:", CAMERA.fps_count(10))
