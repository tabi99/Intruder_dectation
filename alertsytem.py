<<<<<<< HEAD
# limit the number of cpus used by high performance libraries

from audioop import reverse
from enum import Flag, unique
from itertools import count
import os
from sre_parse import FLAGS
from tarfile import DIRTYPE
from turtle import color
from typing import Counter
import sys
sys.path.insert(0, './yolov5')
import argparse
import os
import platform
from twilio.rest import Client
import pyodbc
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import datetime

import detect_face

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
# from utils.general_2 import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                #   check_imshow, xyxy2xywh, increment_path)
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path

from utils.torch_utils import select_device, time_synchronized
from utils.plots_2 import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import numpy as np
from imutils.video import WebcamVideoStream
from statistics import mean, median
import time
import requests
import mysql.connector

# def letterbox(im, new_shape=(160, 160), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
#     # Resize and pad image while meeting stride-multiple constraints
#     shape = im.shape[:2]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)

#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better val mAP)
#         r = min(r, 1.0)

#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

#     dw /= 2  # divide padding into 2 sides
#     dh /= 2

#     if shape[::-1] != new_unpad:  # resize
#         im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#     return im, ratio, (dw, dh)

# def insert_db(mydb,cam_id,out_id,ctime,shelf_no,c_id,res_area):
    # send to database
    
    # mycursor = mydb.cursor()
    # sql = '''INSERT INTO Disrupt.dbo.dwell (cam_id,out_id,customer_id,ctime,shelf_no,restricted_area) VALUES (?,?,?,?,?,?)'''
    # val=(cam_id,out_id,c_id,ctime,shelf_no,int(res_area))
    # mycursor.execute(sql,val )
    # mydb.commit()
    
# def upadate_db(mydb,ctime,shelf_no,c_id):
#     # send to database
#     mycursor = mydb.cursor()

#     sql = "UPDATE Disrupt.dbo.dwell SET ctime = ? where customer_id=? and shelf_no= ? "
#     val = (ctime,int(c_id),shelf_no)
#     mycursor.execute(sql, val)
#     mydb.commit()



def detect():
    sid='ACb5dc80eb75200cf68f6b4be52dd7f79e'
    authToken='a1fa42e08325891afcf851a5a5880f3e'
    sec=0
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
    deepsort = DeepSort('osnet_x0_25', max_dist=cfg.DEEPSORT.MAX_DIST,max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)

    # Initialize
    half=False
    # half &= device.type != 'cpu' # half precision only supported on CUDA

    # Load model
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    weightPath = "./yolov5s-face.pt"

    model = detect_face.load_model(weightPath, device)

    # model.classes=[0]
    # stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    # imgsz = check_img_size(416, s=stride)  # check image size

    # Half
    # half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    # if p/t:
        # model.model.half() if half else model.model.float()

    # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names

    # cap=WebcamVideoStream(src='rtsp://admin:pass1105@192.168.1.108:554/live').start()
    cap=cv2.VideoCapture('rtsp://admin:OMJANM@192.168.1.128:554/stream1')
    
    # cap =cv2.VideoCapture(0)
    object_id_list = {}
    sus_id_list={}
    mov={}
    # dtime[id] =[time.time(),0]
    dwell_time = dict()
    dtime = dict()
    res_list=[]
    
    fps_start_time = datetime.datetime.now()
    fps = 0
    
    frameCount=1

    shelfs_roi={0:[(500,200),(1000,615)],1:[(1100,600),(1600,1200)]}
    # 
    # shelfs_roi={ 2 :[(540,160),(1270,420)] }

    for shelf_id in shelfs_roi:
        object_id_list[shelf_id]=[]
        sus_id_list[shelf_id]=[]
        dtime[shelf_id]={}
        
        
    

    while True:
        frameCount +=1
        ret ,im0s = cap.read()
        # im0s=cv2.resize(im0s,(1920,1080))
        if ret is None:
            # cap=WebcamVideoStream(src='rtsp://admin:pass1105@192.168.1.108:554/live').start()
            cap=cv2.VideoCapture('rtsp://admin:pass1105@192.168.1.108:554/stream1')
            
            # cap =cv2.VideoCapture(0)
            print('Camera Reloaded')
            break
     
        # img = letterbox(im0s, 160 , stride=32 , auto=True)[0]
        # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # img = np.ascontiguousarray(img)
        # img = torch.from_numpy(img).to(device)
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
   
        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)
        # live_id_list =[]
        
        # Inference1q
        img=im0s
        pred = model(img)

        # Apply NMS
        pred = non_max_suppression_face(pred, 0.7, 0.1)
        # count_customer_Shelf_1 = 1

        print("\n\n")
    
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 =im0s.copy()
            annotator = Annotator(im0, line_width=2, pil=not ascii)

            w , h =im0.shape[1] ,im0.shape[0]

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        object_id = output[4]
                        cls = output[5]
                        
                        
                        
                        # count_obj(bboxes,w,h,id)
                       
                        # label =f'{id} {names[c]} {conf:.2f}'
                        x1 = output[0]
                        y1 = output[1]
                        x2 = output[2]
                        y2 = output[3]  
                        center=(int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
                        cv2.circle(im0,center,2,(0,0,255),1)
                        img2=(im0[int(y1):int(y2),int(x1):int(x2)])
                
                        # live_id_list.append(id)
                        c = int(cls)  # integer class
                        for shelf_id in shelfs_roi.keys():
                            # object_id_list[shelf_id]=[]
                            shelf_roi=shelfs_roi[shelf_id]
                            x1=shelf_roi[0][0]
                            y1=shelf_roi[0][1]
                            x2=shelf_roi[1][0]
                            y2=shelf_roi[1][1]
                            
  #---------------------------------------------Restricted Area---------------------------------------#
                            
                            if center[0]>x1 and center[0] < x2 and center[1] > y1 and center[1] < y2:
                                
                                if shelf_id ==0:
                                    if object_id not in res_list:
                                        # print("some_one")
                                        res_list.append(object_id)              
                            #             res_area=object_id
                                        today = datetime.datetime.now()
                                        t = time.strftime("%I-%M-%S%p")
                                        file_name = today.strftime('%d-%m-%Y') + '__' + t
                                        save_path=('restricted_area/person.jpg')
                                        cv2.imwrite(save_path,img2)
                                        url='https://twillio.thedisruptlabs.com/upload_file'
                                        files={'files[]':open(save_path,'rb')}
                                        r=requests.post(url,files=files)
                                        print(r.text)
                                        client=Client(sid,authToken)
                                        waurl='https://twillio.thedisruptlabs.com/upload_file/restricted_area/person.jpg'
                                        
                                        client.messages.create(body=f'This person in restricted area  {file_name}  ',
                                                                media_url= 'https://twillio.thedisruptlabs.com/api/person.jpg',
                                                                from_='whatsapp:+14155238886',
                                                                to='whatsapp:+923003538083')
                                                                            
                                # print(f'Someoone {object_id} in Restricted Area ')
                                # insert_db(my_db,cam_id,out_id,ctime=1,shelf_no=0,c_id=1,res_area=res_area)
                            
#-------------------------------------------suspicious person--------------------------------------------------------------#
                            if center[0]>x1 and center[0] < x2 and center[1] > y1 and center[1] < y2:
                                # if shelf_id not in object_id_list[object_id]:
                                if object_id not in sus_id_list[shelf_id]:                                                                   
                                    dtime[shelf_id][object_id] =[time.time(),sec]
                                    sus_id_list[shelf_id].append(object_id)
                                
                                else:                                    
                                    dtime[shelf_id][object_id][1] = time.time() - dtime[shelf_id][object_id][0] 
                                if shelf_id == 1:
                                    
                                    if dtime[shelf_id][object_id][1] >10 and dtime[shelf_id][object_id][1] <10.5 :
                                        annotator.box_label(bboxes, label="id:"+ str(object_id)+" Customer:"+str(shelf_id)+" time:"+str(int(dtime[shelf_id][object_id][1])),color=colors(c, True))
                                        
                                        save_path=('C:/xampp/htdocs/suspicious/person{}.jpg'.format(object_id))
                                        # cloud_path=('G:/My Drive/suspicious/person{}.jpg'.format(object_id))
                                        # cv2.imwrite(cloud_path,img2)
                                        cv2.imwrite(save_path,img2)
                                        client=Client(sid,authToken)
                                        waurl='https://02ae-192-140-145-194.in.ngrok.io/suspicious/person{}.jpg'.format(object_id)
                                 
                                        client.messages.create(body=f' This suspicious person sitting more than 10 second here ',
                                                                media_url= waurl,#'https://fcdb-118-103-235-199.jp.ngrok.io/tmp/person3.jpg',
                                                                from_='whatsapp:+14155238886',
                                                                to='whatsapp:+923003538083')
                                                                           
                            
                            #--------------------NO Movemnt ------------------#
                        
                            if center[0]>x1 and center[0] < x2 and center[1] > y1 and center[1] < y2:
                                # if shelf_id not in object_id_list[object_id]:
                                if object_id not in object_id_list[shelf_id]:                                                                   
                                    dtime[shelf_id][object_id] =[time.time(),sec]
                                    object_id_list[shelf_id].append(object_id) 
                                else:                                    
                                    dtime[shelf_id][object_id][1] = time.time() - dtime[shelf_id][object_id][0] 
                                if shelf_id == 1:
                                    if object_id not in  mov.keys():
                                        
                                        mov[object_id]=bboxes
                                    if dtime[shelf_id][object_id][1] >14.5 and dtime[shelf_id][object_id][1] <16.5:
                                        mov==mov
                                          
                                        save_path=('C:/xampp/htdocs/movent/person{}.jpg'.format(object_id))
                                        # cloud_path=('G:/My Drive/suspicious/person{}.jpg'.format(object_id))
                                        # cv2.imwrite(cloud_path,img2)
                                        cv2.imwrite(save_path,img2)
                                        client=Client(sid,authToken)
                                        waurl='https://02ae-192-140-145-194.in.ngrok.io/movent/person{}.jpg'.format(object_id)
                                 
                                        client.messages.create(body=f' This person not moving last 15 seconds ',
                                                                media_url= waurl,#'https://fcdb-118-103-235-199.jp.ngrok.io/tmp/person3.jpg',
                                                                from_='whatsapp:+14155238886',
                                                                to='whatsapp:+923003538083')
                                                                           
                                        
                                
                                    
                                        
                            #         annotator.box_label(bboxes, label="id:"+ str(object_id)+" Customer:"+str(shelf_id)+" time:"+str(int(dtime[shelf_id][object_id][1])),color=colors(c, True))
                            #         save_path=('C:/xampp/htdocs/tmp/person{}.jpg'.format(object_id))
                                            
                                        
                                    
                    
                            else:
                                if object_id  in object_id_list[shelf_id]:
                                    dtime[shelf_id][object_id][0]= time.time() - dtime[shelf_id][object_id][1]
                                
                    try:        
                        annotator.box_label(bboxes, label="Id: " +str(object_id)+" Time: "+str(int(dtime[shelf_id][object_id][1]))+" Second",color=colors(c, True)) 
                    except:
                        annotator.box_label(bboxes, label="Id: " +str(object_id)+" Time: "+str(int(0))+" Second",color=colors(c, True)) 
                                               
                    

        # else:
            #     deepsort.increment_ages()
                     
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (frameCount / time_diff.seconds)

        fps_text = "FPS: {}".format(int(fps))
        for shelf_id in shelfs_roi.keys():
            shelf_roi=shelfs_roi[shelf_id]
            cv2.rectangle(im0,shelf_roi[0],shelf_roi[1],(0,0,255),3)
            cv2.putText(im0,'Restricted Area',(500,194),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),2)
            cv2.putText(im0,'ROI Area',(1100,570),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),2)

        # cv2.rectangle(im0,(1000,1000),(1300,1200),(0,255,0),1)
        
        cv2.putText(im0, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                    
        cv2.namedWindow("finalImg",cv2.WINDOW_NORMAL)     
        cv2.imshow('finalImg', im0)
       
        if cv2.waitKey(1) == ord('q'):  # q to quit
            break
    cv2.destroyAllWindows()        

if __name__ == '__main__':
    # try:
    #     conn = pyodbc.connect('Driver={SQL Server};'
    #                             'Server=DESKTOP-TIF11FM;'
    #                             'Database=Disrupt;'
    #                             'Trusted_Connection=yes;')
    #     cursor = conn.cursor()
    #     print("Connection succeeded")
    # except Exception as e:
    #     print("Exception: ",e)


    # cam_id=1
    # out_id=5
    # commpany_id=9

    
    
    detect()
    # conn,cam_id,out_id
   
   
    
=======
# limit the number of cpus used by high performance libraries

from audioop import reverse
from enum import Flag, unique
from itertools import count
import os
from sre_parse import FLAGS
from tarfile import DIRTYPE
from turtle import color
from typing import Counter
import sys
sys.path.insert(0, './yolov5')
import argparse
import os
import platform
from twilio.rest import Client
import pyodbc
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import datetime

import detect_face

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
# from utils.general_2 import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                #   check_imshow, xyxy2xywh, increment_path)
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path

from utils.torch_utils import select_device, time_synchronized
from utils.plots_2 import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import numpy as np
from imutils.video import WebcamVideoStream
from statistics import mean, median
import time
import requests
import mysql.connector

# def letterbox(im, new_shape=(160, 160), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
#     # Resize and pad image while meeting stride-multiple constraints
#     shape = im.shape[:2]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)

#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better val mAP)
#         r = min(r, 1.0)

#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

#     dw /= 2  # divide padding into 2 sides
#     dh /= 2

#     if shape[::-1] != new_unpad:  # resize
#         im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#     return im, ratio, (dw, dh)

# def insert_db(mydb,cam_id,out_id,ctime,shelf_no,c_id,res_area):
    # send to database
    
    # mycursor = mydb.cursor()
    # sql = '''INSERT INTO Disrupt.dbo.dwell (cam_id,out_id,customer_id,ctime,shelf_no,restricted_area) VALUES (?,?,?,?,?,?)'''
    # val=(cam_id,out_id,c_id,ctime,shelf_no,int(res_area))
    # mycursor.execute(sql,val )
    # mydb.commit()
    
# def upadate_db(mydb,ctime,shelf_no,c_id):
#     # send to database
#     mycursor = mydb.cursor()

#     sql = "UPDATE Disrupt.dbo.dwell SET ctime = ? where customer_id=? and shelf_no= ? "
#     val = (ctime,int(c_id),shelf_no)
#     mycursor.execute(sql, val)
#     mydb.commit()



def detect():
    sid='ACb5dc80eb75200cf68f6b4be52dd7f79e'
    authToken='a1fa42e08325891afcf851a5a5880f3e'
    sec=0
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
    deepsort = DeepSort('osnet_x0_25', max_dist=cfg.DEEPSORT.MAX_DIST,max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)

    # Initialize
    half=False
    # half &= device.type != 'cpu' # half precision only supported on CUDA

    # Load model
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    weightPath = "./yolov5s-face.pt"

    model = detect_face.load_model(weightPath, device)

    # model.classes=[0]
    # stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    # imgsz = check_img_size(416, s=stride)  # check image size

    # Half
    # half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    # if p/t:
        # model.model.half() if half else model.model.float()

    # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names

    # cap=WebcamVideoStream(src='rtsp://admin:pass1105@192.168.1.108:554/live').start()
    cap=cv2.VideoCapture('rtsp://admin:OMJANM@192.168.1.128:554/stream1')
    
    # cap =cv2.VideoCapture(0)
    object_id_list = {}
    sus_id_list={}
    mov={}
    # dtime[id] =[time.time(),0]
    dwell_time = dict()
    dtime = dict()
    res_list=[]
    
    fps_start_time = datetime.datetime.now()
    fps = 0
    
    frameCount=1

    shelfs_roi={0:[(500,200),(1000,615)],1:[(1100,600),(1600,1200)]}
    # 
    # shelfs_roi={ 2 :[(540,160),(1270,420)] }

    for shelf_id in shelfs_roi:
        object_id_list[shelf_id]=[]
        sus_id_list[shelf_id]=[]
        dtime[shelf_id]={}
        
        
    

    while True:
        frameCount +=1
        ret ,im0s = cap.read()
        # im0s=cv2.resize(im0s,(1920,1080))
        if ret is None:
            # cap=WebcamVideoStream(src='rtsp://admin:pass1105@192.168.1.108:554/live').start()
            cap=cv2.VideoCapture('rtsp://admin:pass1105@192.168.1.108:554/stream1')
            
            # cap =cv2.VideoCapture(0)
            print('Camera Reloaded')
            break
     
        # img = letterbox(im0s, 160 , stride=32 , auto=True)[0]
        # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # img = np.ascontiguousarray(img)
        # img = torch.from_numpy(img).to(device)
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
   
        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)
        # live_id_list =[]
        
        # Inference1q
        img=im0s
        pred = model(img)

        # Apply NMS
        pred = non_max_suppression_face(pred, 0.7, 0.1)
        # count_customer_Shelf_1 = 1

        print("\n\n")
    
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 =im0s.copy()
            annotator = Annotator(im0, line_width=2, pil=not ascii)

            w , h =im0.shape[1] ,im0.shape[0]

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                # draw boxes for visualization
                if len(outputs) > 0:
                    
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        object_id = output[4]
                        cls = output[5]
                        
                        
                        
                        # count_obj(bboxes,w,h,id)
                       
                        # label =f'{id} {names[c]} {conf:.2f}'
                        x1 = output[0]
                        y1 = output[1]
                        x2 = output[2]
                        y2 = output[3]  
                        center=(int(x1 + (x2 - x1) / 2), int(y1 + (y2 - y1) / 2))
                        cv2.circle(im0,center,2,(0,0,255),1)
                        img2=(im0[int(y1):int(y2),int(x1):int(x2)])
                
                        # live_id_list.append(id)
                        c = int(cls)  # integer class
                        for shelf_id in shelfs_roi.keys():
                            # object_id_list[shelf_id]=[]
                            shelf_roi=shelfs_roi[shelf_id]
                            x1=shelf_roi[0][0]
                            y1=shelf_roi[0][1]
                            x2=shelf_roi[1][0]
                            y2=shelf_roi[1][1]
                            
  #---------------------------------------------Restricted Area---------------------------------------#
                            
                            if center[0]>x1 and center[0] < x2 and center[1] > y1 and center[1] < y2:
                                
                                if shelf_id ==0:
                                    if object_id not in res_list:
                                        # print("some_one")
                                        res_list.append(object_id)              
                            #             res_area=object_id
                                        today = datetime.datetime.now()
                                        t = time.strftime("%I-%M-%S%p")
                                        file_name = today.strftime('%d-%m-%Y') + '__' + t
                                        save_path=('restricted_area/person.jpg')
                                        cv2.imwrite(save_path,img2)
                                        url='https://twillio.thedisruptlabs.com/upload_file'
                                        files={'files[]':open(save_path,'rb')}
                                        r=requests.post(url,files=files)
                                        print(r.text)
                                        client=Client(sid,authToken)
                                        waurl='https://twillio.thedisruptlabs.com/upload_file/restricted_area/person.jpg'
                                        
                                        client.messages.create(body=f'This person in restricted area  {file_name}  ',
                                                                media_url= 'https://twillio.thedisruptlabs.com/api/person.jpg',
                                                                from_='whatsapp:+14155238886',
                                                                to='whatsapp:+923003538083')
                                                                            
                                # print(f'Someoone {object_id} in Restricted Area ')
                                # insert_db(my_db,cam_id,out_id,ctime=1,shelf_no=0,c_id=1,res_area=res_area)
                            
#-------------------------------------------suspicious person--------------------------------------------------------------#
                            if center[0]>x1 and center[0] < x2 and center[1] > y1 and center[1] < y2:
                                # if shelf_id not in object_id_list[object_id]:
                                if object_id not in sus_id_list[shelf_id]:                                                                   
                                    dtime[shelf_id][object_id] =[time.time(),sec]
                                    sus_id_list[shelf_id].append(object_id)
                                
                                else:                                    
                                    dtime[shelf_id][object_id][1] = time.time() - dtime[shelf_id][object_id][0] 
                                if shelf_id == 1:
                                    
                                    if dtime[shelf_id][object_id][1] >10 and dtime[shelf_id][object_id][1] <10.5 :
                                        annotator.box_label(bboxes, label="id:"+ str(object_id)+" Customer:"+str(shelf_id)+" time:"+str(int(dtime[shelf_id][object_id][1])),color=colors(c, True))
                                        
                                        save_path=('C:/xampp/htdocs/suspicious/person{}.jpg'.format(object_id))
                                        # cloud_path=('G:/My Drive/suspicious/person{}.jpg'.format(object_id))
                                        # cv2.imwrite(cloud_path,img2)
                                        cv2.imwrite(save_path,img2)
                                        client=Client(sid,authToken)
                                        waurl='https://02ae-192-140-145-194.in.ngrok.io/suspicious/person{}.jpg'.format(object_id)
                                 
                                        client.messages.create(body=f' This suspicious person sitting more than 10 second here ',
                                                                media_url= waurl,#'https://fcdb-118-103-235-199.jp.ngrok.io/tmp/person3.jpg',
                                                                from_='whatsapp:+14155238886',
                                                                to='whatsapp:+923003538083')
                                                                           
                            
                            #--------------------NO Movemnt ------------------#
                        
                            if center[0]>x1 and center[0] < x2 and center[1] > y1 and center[1] < y2:
                                # if shelf_id not in object_id_list[object_id]:
                                if object_id not in object_id_list[shelf_id]:                                                                   
                                    dtime[shelf_id][object_id] =[time.time(),sec]
                                    object_id_list[shelf_id].append(object_id) 
                                else:                                    
                                    dtime[shelf_id][object_id][1] = time.time() - dtime[shelf_id][object_id][0] 
                                if shelf_id == 1:
                                    if object_id not in  mov.keys():
                                        
                                        mov[object_id]=bboxes
                                    if dtime[shelf_id][object_id][1] >14.5 and dtime[shelf_id][object_id][1] <16.5:
                                        mov==mov
                                          
                                        save_path=('C:/xampp/htdocs/movent/person{}.jpg'.format(object_id))
                                        # cloud_path=('G:/My Drive/suspicious/person{}.jpg'.format(object_id))
                                        # cv2.imwrite(cloud_path,img2)
                                        cv2.imwrite(save_path,img2)
                                        client=Client(sid,authToken)
                                        waurl='https://02ae-192-140-145-194.in.ngrok.io/movent/person{}.jpg'.format(object_id)
                                 
                                        client.messages.create(body=f' This person not moving last 15 seconds ',
                                                                media_url= waurl,#'https://fcdb-118-103-235-199.jp.ngrok.io/tmp/person3.jpg',
                                                                from_='whatsapp:+14155238886',
                                                                to='whatsapp:+923003538083')
                                                                           
                                        
                                
                                    
                                        
                            #         annotator.box_label(bboxes, label="id:"+ str(object_id)+" Customer:"+str(shelf_id)+" time:"+str(int(dtime[shelf_id][object_id][1])),color=colors(c, True))
                            #         save_path=('C:/xampp/htdocs/tmp/person{}.jpg'.format(object_id))
                                            
                                        
                                    
                    
                            else:
                                if object_id  in object_id_list[shelf_id]:
                                    dtime[shelf_id][object_id][0]= time.time() - dtime[shelf_id][object_id][1]
                                
                    try:        
                        annotator.box_label(bboxes, label="Id: " +str(object_id)+" Time: "+str(int(dtime[shelf_id][object_id][1]))+" Second",color=colors(c, True)) 
                    except:
                        annotator.box_label(bboxes, label="Id: " +str(object_id)+" Time: "+str(int(0))+" Second",color=colors(c, True)) 
                                               
                    

        # else:
            #     deepsort.increment_ages()
                     
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (frameCount / time_diff.seconds)

        fps_text = "FPS: {}".format(int(fps))
        for shelf_id in shelfs_roi.keys():
            shelf_roi=shelfs_roi[shelf_id]
            cv2.rectangle(im0,shelf_roi[0],shelf_roi[1],(0,0,255),3)
            cv2.putText(im0,'Restricted Area',(500,194),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),2)
            cv2.putText(im0,'ROI Area',(1100,570),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),2)

        # cv2.rectangle(im0,(1000,1000),(1300,1200),(0,255,0),1)
        
        cv2.putText(im0, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                    
        cv2.namedWindow("finalImg",cv2.WINDOW_NORMAL)     
        cv2.imshow('finalImg', im0)
       
        if cv2.waitKey(1) == ord('q'):  # q to quit
            break
    cv2.destroyAllWindows()        

if __name__ == '__main__':
    # try:
    #     conn = pyodbc.connect('Driver={SQL Server};'
    #                             'Server=DESKTOP-TIF11FM;'
    #                             'Database=Disrupt;'
    #                             'Trusted_Connection=yes;')
    #     cursor = conn.cursor()
    #     print("Connection succeeded")
    # except Exception as e:
    #     print("Exception: ",e)


    # cam_id=1
    # out_id=5
    # commpany_id=9

    
    
    detect()
    # conn,cam_id,out_id
   
   
    
>>>>>>> 1fef490 (add files2)
