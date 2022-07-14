#khai bao trong tam bounding box
def get_center_point(point1, point2):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]

    return [int((x1+x2)/2), int((y1+y2)/2)]
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
        except:
            pass
