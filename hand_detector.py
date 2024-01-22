# hand detect using mediapipe
import math
import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.lmList = []
        self.results = None
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_img)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        zList = []
        bbox = []
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z*1000
                xList.append(cx)
                yList.append(cy)
                zList.append(cz)
                # print(id, cx, cy, cz)
                self.lmList.append([id, cx, cy, cz])

                if draw:
                    cv2.circle(img, (cx, cy), 6, (0, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0]-20, bbox[1]-20), (bbox[2]+20, bbox[3]+20), (0, 255, 0), 2)

        return self.lmList, bbox
    
    #TODO 손가락 앞으로 펼쳐지더라도 펼쳐진거 인식 + 손가락 아래로
    # def fingersUp(self):
    #     fingers = []
    #     # print(f"lmList[4]의 x좌표: {self.lmList[self.tipIds[0]][1]}, lmList[3]의 x좌표: {self.lmList[self.tipIds[0] - 1][1]}")
    #     print(f"lmList[8]의 z좌표: {self.lmList[self.tipIds[1]][3]}, lmList[6]의 z좌표: {self.lmList[self.tipIds[1] - 2][3]}, lmList[5]의 z좌표: {self.lmList[self.tipIds[1] - 3][3]}")
    #     # Thumb
    #     if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
    #         fingers.append(1)
    #     else:
    #         fingers.append(0)
    #     # 4 Fingers
    #     #8: 검지, 12: 중지, 16: 약지, 20: 새끼
    #     for id in range(1, 5):
    #         if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
    #             fingers.append(1)
    #         else:
    #             fingers.append(0)
    #     print(fingers)
    #     return fingers
    
    def fingersfold(self):
        fingers = []
        x0, y0, z0 = self.lmList[0][1:]
        x1, y1, z1 = self.lmList[4][1:]
        x1p, y1p, z1p = self.lmList[2][1:]
        # Thumb
        # if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
        dist_thumb = math.hypot(x1-x0, y1-y0, z1-z0)
        dist_thumb_palm = math.hypot(x1-x1p, y1-y1p, z1-z1p)
        dist_thumb_absolute = dist_thumb / dist_thumb_palm
        # print(f"엄지끝과 손목 까지의 거리: {dist_thumb}")
        # print(f"엄지 절대비율: {dist_thumb_absolute}")
        if dist_thumb_absolute > 2.3:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Index
        xi, yi, zi = self.lmList[8][1:]
        xip, yip, zip = self.lmList[6][1:]
        dist_index = math.hypot(xi-x0, yi-y0,zi-z0)
        # print(f"검지 손목: {dist_index}")
        dist_index_palm = math.hypot(xi-xip, yi-yip, zi-zip)
        # print(f"검지 손바닥: {dist_index_palm}")
        dist_index_absolute = dist_index / dist_index_palm
        # print(f"검지 손가락 절대 비율: {dist_index_absolute}")
        if dist_index_absolute > 3.7:
            fingers.append(1)
        else:
            fingers.append(0)
        # Three fingers
        for id in range(2, 5):
            x, y, z = self.lmList[self.tipIds[id]][1:]
            xp, yp, zp = self.lmList[self.tipIds[id]-2][1:]
            # if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
            dist_four = math.hypot(x-x0, y-y0, z-z0)
            # print(f"손가락 끝과 손목까지의 거리 id[{id}]: {dist_four}")
            dist_palm = math.hypot(x-xp, y-yp, z-zp)
            # print(f"손가락 끝과 손바닥 까지의 거리 id[{id}]: {dist_palm}")
            dist_absolute = dist_four / dist_palm
            # print(f"절대 비율 [{id}]: {dist_absolute}")
            if dist_absolute > 3.5:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1, z1 = self.lmList[p1][1:]
        x2, y2, z2 = self.lmList[p2][1:]
        cx, cy, cz = (x1 + x2) // 2, (y1 + y2) // 2, (z1 + z2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.circle(img, (x1, y1), 6, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 6, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 6, (0, 255, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy, cz]

    def findAngle(self, p1, p2, p3, img, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # print(angle)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), cv2.FILLED)
            # cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        print(f"angle: {angle}")
        return angle

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
