import cv2
import time
import numpy as np
import hand_detector as hd
import pyautogui
import math

wCam, hCam = 1280, 960
# wCam, hCam = 640, 480
frameR = 200
smoothening = 10

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = hd.handDetector(detectionCon = 0.7)
wScr, hScr = pyautogui.size()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    output = img.copy()
    
    if len(lmList) != 0:
        x0, y0, z0 = lmList[0][1:]
        x1, y1, z1 = lmList[8][1:]
        x2, y2, z2 = lmList[12][1:]
        

        # fingers 펴져 있는지 판단 (0,1,2,3,4)
        fingers = detector.fingersfold()
        print(fingers)
        num_zero = 0
        num_one =  5 - num_zero
        for i in fingers:
            if i==0:
                num_zero+=1
        # img = cv2.addWeighted(img, 0.5, output, 1-.5, 0, output)
                
        if num_zero==5:
                print("정지")
        #검지는 펴져 있고 중지는 구부러져 있는 경우
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            pyautogui.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 6, (255, 28, 0), cv2.FILLED)
            plocX, plocY = clocX, clocY
        
        #엄지, 검지, 중지 다 펴져있는 경우
        if fingers[0] ==1 and fingers[1]==1 and fingers[2]==1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            cx, cy, cz = lineInfo[4:]
            # print(f"length: {length}")
            if length < 500:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 6, (0,255,0), cv2.FILLED)
                print(f"cx: {cx}, cy: {cy}, cz: {cz}")
                print(f"wrist_x: {x0}, wrist_y: {y0}, xrist_z: {z0}")
                
                #표준 길이로 변환(검지 손가락 한마디 기준)
                standard_x, standard_y = abs(lmList[8][1]-lmList[7][1]), abs(lmList[8][2]-lmList[7][2])
                dist_x, dist_y = 0, 0
                if standard_x != 0 and standard_y != 0:
                    dist_y = abs(cy - y0) / standard_y
                    dist_x = abs(cx - x0) / standard_x
                print(f"dist_y: {dist_y}")
                print(f"dist_x: {dist_x}")

                if cz < z0:
                    pyautogui.keyUp('down')
                    pyautogui.keyUp('left')
                    pyautogui.keyUp('right')
                    print("앞으로")
                    pyautogui.keyDown('up')
                    # time.sleep(0.01)
                    # pyautogui.keyUp('up')
                    # if dist_y > 7 and dist_y < 10 and cy < y0:
                    #     print("위로")
                    #     pyautogui.press('up')
                    # if dist_y > 7 and dist_y < 10 and cy > y0:
                    #     print("아래로")
                    #     pyautogui.keyDown('down')
                    if dist_x > 7 and dist_x < 10 and cx > x0:
                        # pyautogui.keyUp('left')
                        # pyautogui.keyUp('right')
                        print("왼쪽으로")
                        # pyautogui.keyDown('left')
                        pyautogui.press('left')
                    if dist_x > 7 and dist_x < 10 and cx < x0:
                        # pyautogui.keyUp('left')
                        # pyautogui.keyUp('right')
                        print("오른쪽으로")
                        # pyautogui.keyDown('right')
                        pyautogui.press('right')
                elif cz > z0:
                    pyautogui.keyUp('up')
                    pyautogui.keyUp('left')
                    pyautogui.keyUp('right')
                    print("뒤로")
                    pyautogui.keyDown('down')
        


                # if cz < z0:
                #     print("앞으로!!!")
                #     if cy < y0:
                #         print("위로")
                #     if cx > x0:
                #         print("왼쪽으로")
                #     if cx < x0:
                #         print("오른쪽으로")
                #     if cy > y0:
                #         print("아래로")
                
                # dist = math.hypot(cx-x0, cy-y0, cz-z0)
                # dist_y = abs(cy-y0)
                # print(f"dist: {dist}")
                # print(f"dist_y: {dist_y}")
                # if dist_y < 160:
                #     pyautogui.scroll(-1)

        # 엄지는 펴져 있고 검지는 구부러져 있는 경우
        # elif fingers[0]==1 and fingers[1]==0:
        #     cap.release()
        #     cv2.destroyAllWindows()
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.imshow("Virtual mouse monitor", cv2.flip(img, 1))
    cv2.setWindowProperty("Virtual mouse monitor", cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(1)