cap = cv.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0
while True:
    ret, frame = cap.read()

    data = object_detector(frame)
    for d in data:
        if d[0] == 'thunghang':
            distance = distance_finder(focal_TH, TH_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'robotcar':
            distance = distance_finder(focal_RC, RC_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'steering':
            distance = distance_finder(focal_ST, ST_WIDTH, d[1])
            x, y = d[2]
        cv.rectangle(frame, (x, y - 3), (x + 150, y + 23), BLACK, -1)
        cv.putText(frame, f'Dis: {round(distance, 2)} cm', (x + 5, y + 13), FONTS, 0.48, GREEN, 2)

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
