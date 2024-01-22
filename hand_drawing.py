import cv2
import pyautogui
import time
import math
import hand_detector as hd
import numpy as np

wCam, hCam = 1280, 960
frameR = 100
smoothening = 10
threshold_speed = 300
prev_x, prev_y, prev_time = 0, 0, time.time()
curr_x, curr_y = 0, 0
prev_index_y = 0

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
detector = hd.handDetector(detectionCon = 0.7)
wScr, hScr = pyautogui.size()

while True:
    tipIds = [4,8,12,16,20]
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    output = img.copy()
    if len(lmList) != 0:
        x1, y1, z1 = lmList[8][1:]
        curr_time = time.time()

        #손가락 펴짐, 구부러짐 판단
        fingers = detector.fingersUp()
        print(fingers)
        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            #TODO 마우스 커서 이동
            cv2.circle(img, (x1, y1), 10, (255, 28, 0), cv2.FILLED)
            print("마우스 커서 이동")

            #화면에 따른 위치 계산
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0,hScr))
            curr_x = prev_x + (x3 - prev_x) / smoothening
            curr_y = prev_y + (y3 - prev_y) / smoothening

            #커서 이동
            pyautogui.moveTo(wScr - curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y
            prev_time = curr_time
            
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            #TODO 클릭 앤 드래그
            length, img, lineInfo = detector.findDistance(8, 12, img)
            cx, cy = lineInfo[4:6]
            xmin, ymin, xmax, ymax = bbox[0:]
            standard_lenth = math.hypot(xmax - xmin, ymax - ymin)    
            length = standard_lenth / length
            print(f'length: {length}')
            if length > 6:
                print("클릭")
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0,255,0), cv2.FILLED)
                print(f"cx: {cx}, cy: {cy}")
                pyautogui.click()
            if length > 2.5 and length < 4.5:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (255,0,255), cv2.FILLED)
                x0, y0 = lmList[0][1:3]
                if cy > y0:
                    print("스크롤 다운")
                    pyautogui.scroll(-2)
                else:
                    print("스크롤 업")                   
                    pyautogui.scroll(2)
        
        if fingers[0] == 1  and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
            print("스페이스바")
            # cv2.circle(img, ((lmList[0][1] + lmList[4][1]) / 2, (lmList[0][2] + lmList[4][2]) / 2), 10, (255,255,255), cv2.FILLED)
            cv2.circle(img, (lmList[0][1], lmList[0][2]), 10, (28,255,255), cv2.FILLED)
            pyautogui.press('space')
    
        #TODO 스크린샷
        if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            angle = detector.findAngle(8, 0, 4, img)
            print(f'angle: {angle}')
            if angle > 40 and angle < 50:
                print("스크린샷")
                screenshot = pyautogui.screenshot()
                screenshot.save('screenshot.png')
            
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.imshow("Hand Drawing monitor", cv2.flip(img, 1))
    cv2.setWindowProperty("Hand Drawing monitor", cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(1)