import cv2 as cv
import numpy as np
import time
import math
import i2c
# Distance constants
KNOWN_DISTANCE = 35.0  # CM
TH_WIDTH = 22.5  # CM


# Object detector constant
CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.3

# colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
# defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#  setttng up opencv net
yoloNet = cv.dnn.readNet('yolov4-tiny-custom_final96.weights', 'yolov4-tiny-custom.cfg')

#yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
#yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(96, 96), scale=1 / 255, swapRB=True)

#khai bao trong tam bounding box
def get_center_point(point1, point2):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]

    return [int((x1+x2)/2), int((y1+y2)/2)]
def calculate_angle(p1,p2):
    p1 = p1[0]
    p2 = p2[0]
    return math.atan((20 + ((320 - p1) * 22.5) / (p2 - p1)) / 43.5)

# object detector funciton /method
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

#Xuat ra trong tam bounding boxes
    if boxes is not None:
        try:
            p1 = boxes[0][:2]
            p2 = boxes[0][2:]
            cent_point = get_center_point(p1,p2)
            #print(cent_point)
            #print(p1)
            #print(p2)
            #cv.circle(frame, cent_point,2, GREEN, 2)
            #alpha = atan((15 + ((320 - x1) * 22.5) / (x2 - x1)) / 35)
        except:
            pass

    # creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        # define color of each, object based on its class id
        color = COLORS[int(0) % len(COLORS)]

        label = "%s : %f" % (class_names[0], score)

        # draw rectangle on and label on object
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1] - 14), FONTS, 0.5, color, 2)

        # getting the data
        # 1: class name  2: object width in pixels, 3: position where have to draw text(distance)
        if classid == 0:  # class id
            data_list.append([class_names[classid[0]], box[2], (box[0], box[1] - 2),(box[0],box[1]),(box[0]+box[2],box[1]+box[3])])

        # if you want inclulde more classes then you have to simply add more [elif] statements here
        # returning list containing the object data.
    return data_list


def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width

    return focal_length


# distance finder function
def distance_finder(focal_length, real_object_width, width_in_frmae):
    distance = (real_object_width * focal_length) / width_in_frmae
    return distance


# reading the reference image from dir
ref_TH = cv.imread('ReferenceImages/th7.jpg')

TH_data = object_detector(ref_TH)
TH_width_in_rf = TH_data[0][1]

#TH_width_in_rf = 450
TH_width_in_rf = 315

print(
    f"thunghang width in pixels : {TH_width_in_rf}")

# finding focal length
focal_TH = focal_length_finder(KNOWN_DISTANCE, TH_WIDTH, TH_width_in_rf)

cap = cv.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0

while True:
    a = 0
    ret, frame = cap.read()
    #frame = cv.resize(frame, (214,214))
    #print(frame.shape)

    data = object_detector(frame)
    for d in data:
        if d[0] == 'thunghang':
            distance = distance_finder(focal_TH, TH_WIDTH, d[1])
            x, y = d[2]

            cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
            cv.putText(frame, f'Dis: {round(distance, 2)} cm', (x + 5, y + 13), FONTS, 0.48, GREEN, 2)

            if distance<50:
                a=1
                p1 = d[3]
                p2 = d[4]
                angle = calculate_angle(p1,p2)
                angle = round(angle*180/3.14,2)
                if 50<angle:
                    angle = 52
                elif 40<angle<=50:
                    angle = 45
                elif 30<angle<=40:
                    angle = 35
                elif 20<angle<=30:
                    angle = 25
                elif 10<angle<=20:
                    angle = 15
                elif angle<10:
                    angle = 0
                #print(angle)
                i2c.i2c(a, angle)

#FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    cv.putText(frame, "fps =  ", (20, 50), FONTS, 0.7, (0, 0, 255), 2)
    cv.putText(frame, str(int(fps)), (95, 50), FONTS, 0.7, (0, 0, 255), 2)
    cv.imshow('frame',frame)

    key = cv.waitKey(1)
    if key ==ord('q'):
        break
cv.destroyAllWindows()
cap.release()

