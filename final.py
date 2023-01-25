<<<<<<< HEAD
from deepface import DeepFace
import cv2
import numpy as np
import os
import torch
import datetime
import time
import requests
from twilio.rest import Client
import detect_face
from PIL import Image
from imutils.video import WebcamVideoStream
# import glob
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from math import atan2, degrees, radians
from tensorflow.keras.preprocessing import image
from utils.general import xyxy2xywh, get_angle
import tensorflow as tf
from threading import Thread

gpus = tf.config.list_physical_devices('GPU')
if gpus:
# Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib","SFace"]

model = DeepFace.build_model("Facenet512")


def get_angle(point_1, point_2): #These can also be four parameters instead of two arrays
    angle = atan2(point_1[1] - point_2[1], point_1[0] - point_2[0])

    angle = degrees(angle)
    return angle


def preprocess_face(img, target_size=(224, 224), grayscale = False, enforce_detection = True, detector_backend = 'opencv', return_region = False, align = False):

	#img might be path, base64 or numpy array. Convert it to numpy whatever it is.
	# img = load_image(img)
	base_img = img.copy()


	#--------------------------

	if img.shape[0] == 0 or img.shape[1] == 0:
		if enforce_detection == True:
			raise ValueError("Detected face shape is ", img.shape,". Consider to set enforce_detection argument to False.")
		else: #restore base image
			img = base_img.copy()

	#--------------------------

	#post-processing
	if grayscale == True:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#---------------------------------------------------
	#resize image to expected shape

	img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image

	if img.shape[0] > 0 and img.shape[1] > 0:
		factor_0 = target_size[0] / img.shape[0]
		factor_1 = target_size[1] / img.shape[1]
		factor = min(factor_0, factor_1)

		dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        # print(img)
		img = cv2.resize(img, dsize)
        
		# Then pad the other side to the target size by adding black pixels
		diff_0 = target_size[0] - img.shape[0]
		diff_1 = target_size[1] - img.shape[1]
		if grayscale == False:
			# Put the base image in the middle of the padded image
			img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
		else:
			img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

	#------------------------------------------
	try:
	#double check: if target image is not still the same size with target.
		if img.shape[0:2] != target_size:
			img = cv2.resize(img, target_size)
	except:
		pass
	#---------------------------------------------------

	#normalizing the image pixels

	img_pixels = image.img_to_array(img) #what this line doing? must?
	# print("1",img_pixels.shape)
	img_pixels = np.expand_dims(img_pixels, axis = 0)
	# print("2",img_pixels.shape)
	img_pixels /= 255 #normalize input in [0, 1]

	#---------------------------------------------------

	if return_region == True:
		return img_pixels
	else:
		return img_pixels




# # Vectors
# vec_a = [1, 2, 3, 4, 5]
# vec_b = [1, 3, 5, 7, 9]
# def findCosineDistance(source_representation, test_representation):
#     dot = sum(a*b for a, b in zip(source_representation, test_representation))
#     norm_a = sum(a*a for a in source_representation) ** 0.5
#     norm_b = sum(b*b for b in test_representation) ** 0.5
#     cos_sim = dot / (norm_a*norm_b)
#     return cos_sim




def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def process(img):
    width = height = 160
    # img=cv2.resize(img,(width,height))
    # img=img.reshape(1,width,height,3)
    img=preprocess_face(img, target_size=(width, height), grayscale = False, enforce_detection = True, detector_backend = 'opencv', return_region = False, align = True)

    img_representation = model.predict(img)[0,:]
    return(img_representation)


def processDataset(dataset):
    known={}
    dir_list = os.listdir(dataset)
    
    for image in dir_list:
        # print("imgae",image)
        img=cv2.imread(dataset+image)
        # print(img.shape)
        boxes, landmarks,confs = detect_face.detect_one(model1, img, device)
        # print(boxes)
        if len(boxes)!=0:
            x1,y1,x2,y2=boxes[0]
            img=img[int(y1):int(y2),int(x1):int(x2)]
        # w,h,_=img.shape
        # r=h/w
        
        # img=cv2.resize(img,(500,int(500*r)))
        
        encoding=process(img)
        name=image.split(".")
        # print(name)
        known[name[0]] = encoding
    return known

def trackFaces(deepsort, confs, xywhs, img):
    # pass detections to deepsort
    clss = torch.as_tensor([0]*len(xywhs))
    xywhs = torch.as_tensor(xywhs)
    confs = torch.as_tensor(confs)
    
    if len(xywhs)>0:
        outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), img)
    else:
        deepsort.increment_ages()
    return outputs


def findDistances(img):

    distances={}
    # w,h,_=img.shape
    # r=h/w
    # img=cv2.resize(img,(500,int(500*r)))
    unknown_encoding=process(img)
    for known_encoding in known_encodings:
        # print("koooo",known_encoding)
        distance=findCosineDistance(unknown_encoding,known_encodings[known_encoding])
        # distances.append(distance)
        distances[known_encoding] = distance
    # print(distances)
    return distances

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weightPath = "./yolov5s-face.pt"

model1 = detect_face.load_model(weightPath, device)

known_encodings=processDataset(dataset="approved_faces/")
# cap=cv2.VideoCapture(0)
cap=WebcamVideoStream(src='rtsp://admin:pass1105@192.168.1.108:554/live').start()

def whatsapp_notification(sid,authToken,save_path,Name,file_name):
    url='https://twillio.thedisruptlabs.com/upload_file'
    files={'files[]':open(save_path,'rb')}
    r=requests.post(url,files=files)
    print(r.text)
    client=Client(sid,authToken)
    # waurl='https://twillio.thedisruptlabs.com/upload_file/restricted_area/person.jpg'

    client.messages.create(body=f'This suspicious person sitting more than 10 second here Name:{Name} Date: {file_name}  ',
                            media_url= 'https://twillio.thedisruptlabs.com/api/person.jpg',
                            from_='whatsapp:+14155238886',
                            to='whatsapp:+923003538083')

cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort('osnet_x0_25', max_dist=cfg.DEEPSORT.MAX_DIST,max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)

sid='ACb5dc80eb75200cf68f6b4be52dd7f79e'
authToken='a1fa42e08325891afcf851a5a5880f3e'
sec=0
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
    # img=cv2.imread("2.jpg")
    img=cap.read()
    # img=cv2.resize(img,(700,500))
    # img = cv2.flip(img, 1)
 
    # try:
    boxes, landmarks, confs = detect_face.detect_one(model1, img, device)
    # except:
    #     continue
    angles = []
    for ind, landmark in enumerate(landmarks):
        box = boxes[ind]
        X1 = int(box[0])
        Y1 = int(box[1])
        X2 = int(box[2])
        Y2 = int(box[3])
        q1, r1, q2, r2 = landmark[:4]
        angle=get_angle((q2, r2),(q1, r1))
        angles.append(angle)            

    if len(boxes)>0:
        xywhs = xyxy2xywh(torch.as_tensor(np.array(boxes))) 
        tracked = trackFaces(deepsort, np.array(confs), np.array(xywhs), img)
        img_angles = {}
        for X1, Y1, X2, Y2, id, _ in tracked:
            if [X1, Y1, X2, Y2] in boxes:
                index = boxes.index([X1, Y1, X2, Y2])
                img_angles[id]= angles[index]
                # cv2.imwrite(str(list_name)+".jpg".format(list_name),img[y1:y2,x1:x2])
                # cv2.imwrite("temp.jpg".format(list_name),img[y1:y2,x1:x2])
            cv2.rectangle(img,(X1, Y1), (X2, Y2), (255, 0, 0), thickness=2)
            cv2.putText(img, str(id), (X1, Y1), cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0, 255,0))
            center=(int(X1 + (X2 - X1) / 2), int(Y1 + (Y2 - Y1) / 2))
            cv2.circle(img,center,2,(0,0,255),1)
            img2=(img[int(Y1):int(Y2),int(X1):int(X2)])
            cv2.rectangle(img, (X1,Y1), (X2,Y2), (0,200,29), 3)
            
            # result=findDistances(img)
      
              
            # print(min(result, key=result.get))
            
           
                # print(person,">>>",result)
                # cv2.putText(img, person, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
            
                # cv2.putText(img, person, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
            
            for shelf_id in shelfs_roi.keys():
                        # object_id_list[shelf_id]=[]
                        shelf_roi=shelfs_roi[shelf_id]
                        x1=shelf_roi[0][0]
                        y1=shelf_roi[0][1]
                        x2=shelf_roi[1][0]
                        y2=shelf_roi[1][1]
                        
                        y2=shelf_roi[1][1]
                        
#---------------------------------------------Restricted Area---------------------------------------#
                        
                        if center[0]>x1 and center[0] < x2 and center[1] > y1 and center[1] < y2:
                            
                            if shelf_id ==0:
                                if id not in res_list:
                                    # print("some_one")
                                    res_list.append(id)              
                        #             res_area=object_id
                                    today = datetime.datetime.now()
                                    t = time.strftime("%I-%M-%S%p")
                                    file_name = today.strftime('%d-%m-%Y') + '__' + t
                                    save_path=('restricted_area/person.jpg')
                                    cv2.imwrite(save_path,img2)
                                    result=findDistances(img2)
                                    person=min(result, key=result.get)
                #                 
                                    if result[person]<=0.35:
                                        Name=person
                                    else:
                                        Name = 'unknown'
                                        
                                    # whatsapp_notification(sid,authToken,save_path,Name,file_name)
                                    Thread(target=whatsapp_notification, args=(sid,authToken,save_path,Name,file_name,)).start()   
                                    # Thread(whatsapp_notification,args=(sid,authToken,save_path,Name,file_name,)) 
                                    
                # # print(person," >>> ",result)
                #                      person="unknown"
                #                      url='https://twillio.thedisruptlabs.com/upload_file'
                #                     files={'files[]':open(save_path,'rb')}
                #                     r=requests.post(url,files=files)
                #                     print(r.text)
                #                     client=Client(sid,authToken)
                #                     waurl='https://twillio.thedisruptlabs.com/upload_file/restricted_area/person.jpg'
                                    
                #                     client.messages.create(body=f'This person in restricted area {person}  {file_name}  ',
                #                                             media_url= 'https://twillio.thedisruptlabs.com/api/person.jpg',
                #                                             from_='whatsapp:+14155238886',
                #                                             to='whatsapp:+923003538083')
                                    
                #                     cv2.putText(img, label="id:"+ str(id)+" Customer:"+str(shelf_id)+" time:"+str(int(dtime[shelf_id][id][1]), (x1+5, y1+5), cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0, 255,0)))
                                                 
#-------------------------------------------suspicious person--------------------------------------------------------------#
                        if center[0]>x1 and center[0] < x2 and center[1] > y1 and center[1] < y2:
                            # if shelf_id not in object_id_list[object_id]:
                            if id not in sus_id_list[shelf_id]:                                                                   
                                dtime[shelf_id][id] =[time.time(),sec]
                                sus_id_list[shelf_id].append(id)
                            
                            else:                                    
                                dtime[shelf_id][id][1] = time.time() - dtime[shelf_id][id][0] 
                            if shelf_id == 1:
                                cv2.putText(img,str(int(dtime[shelf_id][id][1])),(X1+15,Y1),cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255,0),thickness=1)
                                
                                if dtime[shelf_id][id][1] >10 and dtime[shelf_id][id][1] <10.5 :
                                    today = datetime.datetime.now()
                                    t = time.strftime("%I-%M-%S%p")
                                    file_name = today.strftime('%d-%m-%Y') + '__' + t
                                    save_path=('suspicious/person.jpg')
                                    cv2.imwrite(save_path,img2)
                                   
                                    result=findDistances(img2)
                                
                                    person=min(result, key=result.get)
        
                                    if result[person]<=0.35:
                                        Name=person
                                    else:
                                        Name = 'unknown'
                                        
                                    # whatsapp_notification(sid,authToken,save_path,Name,file_name)
                                    Thread(target=whatsapp_notification, args=(sid,authToken,save_path,Name,file_name,)).start()   
                                    # Thread(whatsapp_notification,args=(sid,authToken,save_path,Name,file_name,)) 
                                    
                                  
                                
                                    
                                # else:
                                        
            
                                    #     person="unknown"
                                    #     url='https://twillio.thedisruptlabs.com/upload_file'
                                    #     files={'files[]':open(save_path,'rb')}
                                    #     r=requests.post(url,files=files)
                                    #     print(r.text)
                                    #     client=Client(sid,authToken)
                                    #     waurl='https://twillio.thedisruptlabs.com/upload_file/restricted_area/person.jpg'
                                        
                                    #     client.messages.create(body=f'This suspicious person sitting more than 10 second here {person}  {file_name}  ',
                                    #                             media_url= 'https://twillio.thedisruptlabs.com/api/person.jpg',
                                    #                             from_='whatsapp:+14155238886',
                                    #                             to='whatsapp:+923003538083')
                                                                           
                        else:
                            if id  in object_id_list[shelf_id]:
                                dtime[shelf_id][id][0]= time.time() - dtime[shelf_id][id][1]
                 
                
                
            # try:        
            #     annotator.box_label(boxes, label="Id: " +str(id)+" Time: "+str(int(dtime[shelf_id][id][1]))+" Second",color=colors(c, True)) 
            # except:
            #     annotator.box_label(boxes, label="Id: " +str(id)+" Time: "+str(int(0))+" Second",color=colors(c, True))    
            # person="unknown"
            # print(result)
          
            # print(person,">>>",result)
            # print(person,">>>")
            
                          
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (frameCount / time_diff.seconds)

    fps_text = "FPS: {}".format(int(fps))
    for shelf_id in shelfs_roi.keys():
        shelf_roi=shelfs_roi[shelf_id]
        cv2.rectangle(img,shelf_roi[0],shelf_roi[1],(0,0,255),3)
        cv2.putText(img,'Restricted Area',(500,194),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),2)
        cv2.putText(img,'ROI Area',(1100,570),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),2)
        
    cv2.putText(img, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

          

    
    cv2.namedWindow("finalImg",cv2.WINDOW_NORMAL)     
    cv2.imshow('finalImg', img)
    # cv2.imshow("img",img)
    k=cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
       


=======
from deepface import DeepFace
import cv2
import numpy as np
import os
import torch
import datetime
import time
import requests
from twilio.rest import Client
import detect_face
from PIL import Image
from imutils.video import WebcamVideoStream
# import glob
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from math import atan2, degrees, radians
from tensorflow.keras.preprocessing import image
from utils.general import xyxy2xywh, get_angle
import tensorflow as tf
from threading import Thread

gpus = tf.config.list_physical_devices('GPU')
if gpus:
# Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib","SFace"]

model = DeepFace.build_model("Facenet512")


def get_angle(point_1, point_2): #These can also be four parameters instead of two arrays
    angle = atan2(point_1[1] - point_2[1], point_1[0] - point_2[0])

    angle = degrees(angle)
    return angle


def preprocess_face(img, target_size=(224, 224), grayscale = False, enforce_detection = True, detector_backend = 'opencv', return_region = False, align = False):

	#img might be path, base64 or numpy array. Convert it to numpy whatever it is.
	# img = load_image(img)
	base_img = img.copy()


	#--------------------------

	if img.shape[0] == 0 or img.shape[1] == 0:
		if enforce_detection == True:
			raise ValueError("Detected face shape is ", img.shape,". Consider to set enforce_detection argument to False.")
		else: #restore base image
			img = base_img.copy()

	#--------------------------

	#post-processing
	if grayscale == True:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#---------------------------------------------------
	#resize image to expected shape

	img = cv2.resize(img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image

	if img.shape[0] > 0 and img.shape[1] > 0:
		factor_0 = target_size[0] / img.shape[0]
		factor_1 = target_size[1] / img.shape[1]
		factor = min(factor_0, factor_1)

		dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        # print(img)
		img = cv2.resize(img, dsize)
        
		# Then pad the other side to the target size by adding black pixels
		diff_0 = target_size[0] - img.shape[0]
		diff_1 = target_size[1] - img.shape[1]
		if grayscale == False:
			# Put the base image in the middle of the padded image
			img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
		else:
			img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

	#------------------------------------------
	try:
	#double check: if target image is not still the same size with target.
		if img.shape[0:2] != target_size:
			img = cv2.resize(img, target_size)
	except:
		pass
	#---------------------------------------------------

	#normalizing the image pixels

	img_pixels = image.img_to_array(img) #what this line doing? must?
	# print("1",img_pixels.shape)
	img_pixels = np.expand_dims(img_pixels, axis = 0)
	# print("2",img_pixels.shape)
	img_pixels /= 255 #normalize input in [0, 1]

	#---------------------------------------------------

	if return_region == True:
		return img_pixels
	else:
		return img_pixels




# # Vectors
# vec_a = [1, 2, 3, 4, 5]
# vec_b = [1, 3, 5, 7, 9]
# def findCosineDistance(source_representation, test_representation):
#     dot = sum(a*b for a, b in zip(source_representation, test_representation))
#     norm_a = sum(a*a for a in source_representation) ** 0.5
#     norm_b = sum(b*b for b in test_representation) ** 0.5
#     cos_sim = dot / (norm_a*norm_b)
#     return cos_sim




def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def process(img):
    width = height = 160
    # img=cv2.resize(img,(width,height))
    # img=img.reshape(1,width,height,3)
    img=preprocess_face(img, target_size=(width, height), grayscale = False, enforce_detection = True, detector_backend = 'opencv', return_region = False, align = True)

    img_representation = model.predict(img)[0,:]
    return(img_representation)


def processDataset(dataset):
    known={}
    dir_list = os.listdir(dataset)
    
    for image in dir_list:
        # print("imgae",image)
        img=cv2.imread(dataset+image)
        # print(img.shape)
        boxes, landmarks,confs = detect_face.detect_one(model1, img, device)
        # print(boxes)
        if len(boxes)!=0:
            x1,y1,x2,y2=boxes[0]
            img=img[int(y1):int(y2),int(x1):int(x2)]
        # w,h,_=img.shape
        # r=h/w
        
        # img=cv2.resize(img,(500,int(500*r)))
        
        encoding=process(img)
        name=image.split(".")
        # print(name)
        known[name[0]] = encoding
    return known

def trackFaces(deepsort, confs, xywhs, img):
    # pass detections to deepsort
    clss = torch.as_tensor([0]*len(xywhs))
    xywhs = torch.as_tensor(xywhs)
    confs = torch.as_tensor(confs)
    
    if len(xywhs)>0:
        outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), img)
    else:
        deepsort.increment_ages()
    return outputs


def findDistances(img):

    distances={}
    # w,h,_=img.shape
    # r=h/w
    # img=cv2.resize(img,(500,int(500*r)))
    unknown_encoding=process(img)
    for known_encoding in known_encodings:
        # print("koooo",known_encoding)
        distance=findCosineDistance(unknown_encoding,known_encodings[known_encoding])
        # distances.append(distance)
        distances[known_encoding] = distance
    # print(distances)
    return distances

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weightPath = "./yolov5s-face.pt"

model1 = detect_face.load_model(weightPath, device)

known_encodings=processDataset(dataset="approved_faces/")
# cap=cv2.VideoCapture(0)
cap=WebcamVideoStream(src='rtsp://admin:pass1105@192.168.1.108:554/live').start()

def whatsapp_notification(sid,authToken,save_path,Name,file_name):
    url='https://twillio.thedisruptlabs.com/upload_file'
    files={'files[]':open(save_path,'rb')}
    r=requests.post(url,files=files)
    print(r.text)
    client=Client(sid,authToken)
    # waurl='https://twillio.thedisruptlabs.com/upload_file/restricted_area/person.jpg'

    client.messages.create(body=f'This suspicious person sitting more than 10 second here Name:{Name} Date: {file_name}  ',
                            media_url= 'https://twillio.thedisruptlabs.com/api/person.jpg',
                            from_='whatsapp:+14155238886',
                            to='whatsapp:+923003538083')

cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort('osnet_x0_25', max_dist=cfg.DEEPSORT.MAX_DIST,max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)

sid='ACb5dc80eb75200cf68f6b4be52dd7f79e'
authToken='a1fa42e08325891afcf851a5a5880f3e'
sec=0
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
    # img=cv2.imread("2.jpg")
    img=cap.read()
    # img=cv2.resize(img,(700,500))
    # img = cv2.flip(img, 1)
 
    # try:
    boxes, landmarks, confs = detect_face.detect_one(model1, img, device)
    # except:
    #     continue
    angles = []
    for ind, landmark in enumerate(landmarks):
        box = boxes[ind]
        X1 = int(box[0])
        Y1 = int(box[1])
        X2 = int(box[2])
        Y2 = int(box[3])
        q1, r1, q2, r2 = landmark[:4]
        angle=get_angle((q2, r2),(q1, r1))
        angles.append(angle)            

    if len(boxes)>0:
        xywhs = xyxy2xywh(torch.as_tensor(np.array(boxes))) 
        tracked = trackFaces(deepsort, np.array(confs), np.array(xywhs), img)
        img_angles = {}
        for X1, Y1, X2, Y2, id, _ in tracked:
            if [X1, Y1, X2, Y2] in boxes:
                index = boxes.index([X1, Y1, X2, Y2])
                img_angles[id]= angles[index]
                # cv2.imwrite(str(list_name)+".jpg".format(list_name),img[y1:y2,x1:x2])
                # cv2.imwrite("temp.jpg".format(list_name),img[y1:y2,x1:x2])
            cv2.rectangle(img,(X1, Y1), (X2, Y2), (255, 0, 0), thickness=2)
            cv2.putText(img, str(id), (X1, Y1), cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0, 255,0))
            center=(int(X1 + (X2 - X1) / 2), int(Y1 + (Y2 - Y1) / 2))
            cv2.circle(img,center,2,(0,0,255),1)
            img2=(img[int(Y1):int(Y2),int(X1):int(X2)])
            cv2.rectangle(img, (X1,Y1), (X2,Y2), (0,200,29), 3)
            
            # result=findDistances(img)
      
              
            # print(min(result, key=result.get))
            
           
                # print(person,">>>",result)
                # cv2.putText(img, person, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
            
                # cv2.putText(img, person, (x1,y1), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA)
            
            for shelf_id in shelfs_roi.keys():
                        # object_id_list[shelf_id]=[]
                        shelf_roi=shelfs_roi[shelf_id]
                        x1=shelf_roi[0][0]
                        y1=shelf_roi[0][1]
                        x2=shelf_roi[1][0]
                        y2=shelf_roi[1][1]
                        
                        y2=shelf_roi[1][1]
                        
#---------------------------------------------Restricted Area---------------------------------------#
                        
                        if center[0]>x1 and center[0] < x2 and center[1] > y1 and center[1] < y2:
                            
                            if shelf_id ==0:
                                if id not in res_list:
                                    # print("some_one")
                                    res_list.append(id)              
                        #             res_area=object_id
                                    today = datetime.datetime.now()
                                    t = time.strftime("%I-%M-%S%p")
                                    file_name = today.strftime('%d-%m-%Y') + '__' + t
                                    save_path=('restricted_area/person.jpg')
                                    cv2.imwrite(save_path,img2)
                                    result=findDistances(img2)
                                    person=min(result, key=result.get)
                #                 
                                    if result[person]<=0.35:
                                        Name=person
                                    else:
                                        Name = 'unknown'
                                        
                                    # whatsapp_notification(sid,authToken,save_path,Name,file_name)
                                    Thread(target=whatsapp_notification, args=(sid,authToken,save_path,Name,file_name,)).start()   
                                    # Thread(whatsapp_notification,args=(sid,authToken,save_path,Name,file_name,)) 
                                    
                # # print(person," >>> ",result)
                #                      person="unknown"
                #                      url='https://twillio.thedisruptlabs.com/upload_file'
                #                     files={'files[]':open(save_path,'rb')}
                #                     r=requests.post(url,files=files)
                #                     print(r.text)
                #                     client=Client(sid,authToken)
                #                     waurl='https://twillio.thedisruptlabs.com/upload_file/restricted_area/person.jpg'
                                    
                #                     client.messages.create(body=f'This person in restricted area {person}  {file_name}  ',
                #                                             media_url= 'https://twillio.thedisruptlabs.com/api/person.jpg',
                #                                             from_='whatsapp:+14155238886',
                #                                             to='whatsapp:+923003538083')
                                    
                #                     cv2.putText(img, label="id:"+ str(id)+" Customer:"+str(shelf_id)+" time:"+str(int(dtime[shelf_id][id][1]), (x1+5, y1+5), cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0, 255,0)))
                                                 
#-------------------------------------------suspicious person--------------------------------------------------------------#
                        if center[0]>x1 and center[0] < x2 and center[1] > y1 and center[1] < y2:
                            # if shelf_id not in object_id_list[object_id]:
                            if id not in sus_id_list[shelf_id]:                                                                   
                                dtime[shelf_id][id] =[time.time(),sec]
                                sus_id_list[shelf_id].append(id)
                            
                            else:                                    
                                dtime[shelf_id][id][1] = time.time() - dtime[shelf_id][id][0] 
                            if shelf_id == 1:
                                cv2.putText(img,str(int(dtime[shelf_id][id][1])),(X1+15,Y1),cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255,0),thickness=1)
                                
                                if dtime[shelf_id][id][1] >10 and dtime[shelf_id][id][1] <10.5 :
                                    today = datetime.datetime.now()
                                    t = time.strftime("%I-%M-%S%p")
                                    file_name = today.strftime('%d-%m-%Y') + '__' + t
                                    save_path=('suspicious/person.jpg')
                                    cv2.imwrite(save_path,img2)
                                   
                                    result=findDistances(img2)
                                
                                    person=min(result, key=result.get)
        
                                    if result[person]<=0.35:
                                        Name=person
                                    else:
                                        Name = 'unknown'
                                        
                                    # whatsapp_notification(sid,authToken,save_path,Name,file_name)
                                    Thread(target=whatsapp_notification, args=(sid,authToken,save_path,Name,file_name,)).start()   
                                    # Thread(whatsapp_notification,args=(sid,authToken,save_path,Name,file_name,)) 
                                    
                                  
                                
                                    
                                # else:
                                        
            
                                    #     person="unknown"
                                    #     url='https://twillio.thedisruptlabs.com/upload_file'
                                    #     files={'files[]':open(save_path,'rb')}
                                    #     r=requests.post(url,files=files)
                                    #     print(r.text)
                                    #     client=Client(sid,authToken)
                                    #     waurl='https://twillio.thedisruptlabs.com/upload_file/restricted_area/person.jpg'
                                        
                                    #     client.messages.create(body=f'This suspicious person sitting more than 10 second here {person}  {file_name}  ',
                                    #                             media_url= 'https://twillio.thedisruptlabs.com/api/person.jpg',
                                    #                             from_='whatsapp:+14155238886',
                                    #                             to='whatsapp:+923003538083')
                                                                           
                        else:
                            if id  in object_id_list[shelf_id]:
                                dtime[shelf_id][id][0]= time.time() - dtime[shelf_id][id][1]
                 
                
                
            # try:        
            #     annotator.box_label(boxes, label="Id: " +str(id)+" Time: "+str(int(dtime[shelf_id][id][1]))+" Second",color=colors(c, True)) 
            # except:
            #     annotator.box_label(boxes, label="Id: " +str(id)+" Time: "+str(int(0))+" Second",color=colors(c, True))    
            # person="unknown"
            # print(result)
          
            # print(person,">>>",result)
            # print(person,">>>")
            
                          
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (frameCount / time_diff.seconds)

    fps_text = "FPS: {}".format(int(fps))
    for shelf_id in shelfs_roi.keys():
        shelf_roi=shelfs_roi[shelf_id]
        cv2.rectangle(img,shelf_roi[0],shelf_roi[1],(0,0,255),3)
        cv2.putText(img,'Restricted Area',(500,194),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),2)
        cv2.putText(img,'ROI Area',(1100,570),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,255,0),2)
        
    cv2.putText(img, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

          

    
    cv2.namedWindow("finalImg",cv2.WINDOW_NORMAL)     
    cv2.imshow('finalImg', img)
    # cv2.imshow("img",img)
    k=cv2.waitKey(1)
    if k==27:
        break
cv2.destroyAllWindows()
       


>>>>>>> 1fef490 (add files2)
