import cv2
import mediapipe as mp
import time
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=0, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.complexity = complexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return lmList

def get_volume_control():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    return volume

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    volume = get_volume_control()
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            continue 
        
        img = detector.findHands(img, draw=True)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            thumb_tip = lmList[4][1], lmList[4][2] 
            index_tip = lmList[8][1], lmList[8][2]  
            distance = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
            cv2.line(img, thumb_tip, index_tip, (255, 0, 0), 2)
            if distance < 40:  
                cv2.putText(img, "Pinching!", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                volume.SetMasterVolumeLevelScalar(0.1, None)
            elif distance < 50: 
                cv2.putText(img, "Pinching!", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                volume.SetMasterVolumeLevelScalar(0.2, None) 
            elif distance < 60:
                cv2.putText(img, "Pinching!", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                volume.SetMasterVolumeLevelScalar(0.3, None)
            elif distance < 70:
                cv2.putText(img, "Pinching!", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                volume.SetMasterVolumeLevelScalar(0.4, None)
            elif distance < 80:
                cv2.putText(img, "Pinching!", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                volume.SetMasterVolumeLevelScalar(0.5, None)
            elif distance < 90:
                cv2.putText(img, "Pinching!", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                volume.SetMasterVolumeLevelScalar(0.6, None)
            elif distance < 100:
                cv2.putText(img, "Pinching!", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                volume.SetMasterVolumeLevelScalar(0.7, None)
            elif distance < 110:
                cv2.putText(img, "Pinching!", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                volume.SetMasterVolumeLevelScalar(0.8, None)
            elif distance < 120:
                cv2.putText(img, "Pinching!", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                volume.SetMasterVolumeLevelScalar(0.9, None)
            elif distance < 130:
                cv2.putText(img, "Pinching!", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                volume.SetMasterVolumeLevelScalar(1, None)
            else:
                cv2.putText(img, "Not Pinching", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
