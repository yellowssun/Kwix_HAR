import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from time import sleep
import pandas as pd
import mediapipe as mp
from keyboard import buttonList1, buttonList2, buttonList3, buttonList4

import remove_landmarks.redefine_pose_connection
from mediapipe.framework.formats import landmark_pb2

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def detect_mediapipe(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
detector = HandDetector(detectionCon=0.8, maxHands=2)
cap = cv2.VideoCapture(0)
    

def click_one_hand(hands):      ## one hand
    hand1 = hands[0]
    handType1 = hand1["type"]
    lmList1 = hand1["lmList"]
    if handType1 == "Right":     ## right hand
        fingers1 = detector.fingersUp(hand1)
        return lmList1, fingers1
    else:
        return None, None
            
            
def click_two_hand(hands):      ## two hands
    hand1, hand2 = hands[0], hands[1]
    lmList1, lmList2 = hand1["lmList"], hand2["lmList"]
    handType1, handType2 = hand1["type"], hand2["type"]
    fingers1, fingers2 = detector.fingersUp(hand1), detector.fingersUp(hand2)
    
    if handType1 == "Right":
        return lmList1, fingers1
    else:
        return lmList2, fingers2

cap.set(3, 1200)
cap.set(4, 720)
data = {}       # save User data
case = 0

col_list = ['Name', 'Height', 'Weight', 'Old', 'Sex']
data = []
Name = ""
Height = ""
Weight = ""
Old = ""

"""Bounding box의 사이즈를 계산하여 캠과의 거리를 조절할 수 있음"""

def drawAll(img, buttonList):               ## 버튼을 투명하게 만든다. 
    imgNew = np.zeros_like(img, np.uint8)   ## img와 같은 크기의 zeros를 만든다.
    for button in buttonList:
        x, y = button.pos
        cv2.rectangle(imgNew, button.pos, (x+button.size[0], y+button.size[1]), (100, 150, 255), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x+10, y+40), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
        
    out = img.copy()
    alpha = 0.3
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
        
    return out


lmList1 = None

while True:
    _, img = cap.read()
    hands, img = detector.findHands(img)

    # Name
    if case == 0:
        img = drawAll(img, buttonList1)
        
        if len(hands)==2:
            lmList1, fingers1 = click_two_hand(hands)

        if len(hands)==1:
            lmList1, fingers1 = click_one_hand(hands)

        if lmList1:
            for button in buttonList1:
                x, y = button.pos
                w, h = button.size
                
                if x < lmList1[8][0] < x + w and y < lmList1[8][1] < y +h:
                    cv2.rectangle(img, (x-5, y-5), (x + w +5 , y + h + 5), (140, 190, 255), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 10, y +40), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
                    
                    if fingers1[2] == 1:
                        cv2.rectangle(img, button.pos, (x +w, y + h), (190, 170, 255), cv2.FILLED)
                        cv2.putText(img, button.text, (x +10, y +40), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)
                        Name += button.text
                        sleep(0.25)
                    
        cv2.rectangle(img, (50, 350), (700, 450), (100, 150, 255), cv2.FILLED)
        cv2.putText(img, Name, (60, 430), cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255), 5)
        cv2.imshow('1', img)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        
        
        if Name[-1:] == "/":
            data.append(Name[:-1])
            print(data)
            case = 1

    
    # Height
    if case == 1:
        img = drawAll(img, buttonList2)
        
        if len(hands)==2:
            lmList1, fingers1 = click_two_hand(hands)

        if len(hands)==1:
            lmList1, fingers1 = click_one_hand(hands)

        if lmList1:
            for button in buttonList2:
                x, y = button.pos
                w, h = button.size
                
                if x < lmList1[8][0] < x + w and y < lmList1[8][1] < y +h:
                    cv2.rectangle(img, (x-5, y-5), (x + w +5 , y + h + 5), (140, 190, 255), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 10, y +40), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
                    
                    if fingers1[2] == 1:
                        cv2.rectangle(img, button.pos, (x +w, y + h), (190, 170, 255), cv2.FILLED)
                        cv2.putText(img, button.text, (x +10, y +40), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)
                        Height += button.text
                        sleep(0.25)
                    
        cv2.rectangle(img, (50, 500), (700, 600), (100, 150, 255), cv2.FILLED)
        cv2.putText(img, Height, (60, 560), cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255), 5)
        
        cv2.rectangle(img, (1075, 70), (1500, 0),(100, 150, 255), cv2.FILLED)
        cv2.putText(img, "Height", (1100, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
        
        cv2.imshow('1', img)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        
        if Height[-1:] == "*":
            data.append(Height[:-1])
            case = 2
    
    # Weight        
    if case == 2:
        img = drawAll(img, buttonList2)
        
        if len(hands)==2:
            lmList1, fingers1 = click_two_hand(hands)

        if len(hands)==1:
            lmList1, fingers1 = click_one_hand(hands)

        if lmList1:
            for button in buttonList2:
                x, y = button.pos
                w, h = button.size
                
                if x < lmList1[8][0] < x + w and y < lmList1[8][1] < y +h:
                    cv2.rectangle(img, (x-5, y-5), (x + w +5 , y + h + 5), (140, 190, 255), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 10, y +40), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
                    
                    if fingers1[2] == 1:
                        cv2.rectangle(img, button.pos, (x +w, y + h), (190, 170, 255), cv2.FILLED)
                        cv2.putText(img, button.text, (x +10, y +40), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)
                        Weight += button.text
                        sleep(0.25)
                    
        cv2.rectangle(img, (50, 500), (700, 600), (100, 150, 255), cv2.FILLED)
        cv2.putText(img, Weight, (60, 560), cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255), 5)
        
        cv2.rectangle(img, (1075, 70), (1500, 0),(100, 150, 255), cv2.FILLED)
        cv2.putText(img, "Weight", (1100, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
        
        cv2.imshow('1', img)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        
        if Weight[-1:] == "#":
            data.append(Weight[:-1])
            case = 3
    
    # Old
    if case == 3:
        img = drawAll(img, buttonList2)
        
        if len(hands)==2:
            lmList1, fingers1 = click_two_hand(hands)

        if len(hands)==1:
            lmList1, fingers1 = click_one_hand(hands)

        if lmList1:
            for button in buttonList2:
                x, y = button.pos
                w, h = button.size
                
                if x < lmList1[8][0] < x + w and y < lmList1[8][1] < y +h:
                    cv2.rectangle(img, (x-5, y-5), (x + w +5 , y + h + 5), (140, 190, 255), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 10, y +40), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
                    
                    if fingers1[2] == 1:
                        cv2.rectangle(img, button.pos, (x +w, y + h), (190, 170, 255), cv2.FILLED)
                        cv2.putText(img, button.text, (x +10, y +40), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)
                        Old += button.text
                        sleep(0.25)
                    
        cv2.rectangle(img, (50, 500), (700, 600), (100, 150, 255), cv2.FILLED)
        cv2.putText(img, Old, (60, 560), cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255), 5)
        
        cv2.rectangle(img, (1075, 70), (1500, 0),(100, 150, 255), cv2.FILLED)
        cv2.putText(img, "Old", (1140, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
        
        cv2.imshow('1', img)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        
        if Old[-1:] == "*":
            data.append(Old[:-1])
            case = 4

    # Sex
    if case == 4:
        img = drawAll(img, buttonList3)
        
        if len(hands)==2:
            lmList1, fingers1 = click_two_hand(hands)

        if len(hands)==1:
            lmList1, fingers1 = click_one_hand(hands)

        if lmList1:
            for button in buttonList3:
                x, y = button.pos
                w, h = button.size
                
                if x < lmList1[8][0] < x + w and y < lmList1[8][1] < y +h:
                    cv2.rectangle(img, (x-5, y-5), (x + w +5 , y + h + 5), (140, 190, 255), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 10, y +40), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
                    
                    if fingers1[2] == 1:
                        cv2.rectangle(img, button.pos, (x +w, y + h), (190, 170, 255), cv2.FILLED)
                        cv2.putText(img, button.text, (x +10, y +40), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)
                        sleep(0.25)
                        Sex = button.text
                        data.append(Sex)
                        case = 5
                        
        cv2.rectangle(img, (1075, 70), (1500, 0),(100, 150, 255), cv2.FILLED)
        cv2.putText(img, "Sex", (1140, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)

        cv2.imshow('1', img)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    
    # Ex_mode
    if case == 5:
        img = drawAll(img, buttonList4)
        
        if len(hands)==2:
            lmList1, fingers1 = click_two_hand(hands)

        if len(hands)==1:
            lmList1, fingers1 = click_one_hand(hands)

        if lmList1:
            for button in buttonList4:
                x, y = button.pos
                w, h = button.size
                
                if x < lmList1[8][0] < x + w and y < lmList1[8][1] < y +h:
                    cv2.rectangle(img, (x-5, y-5), (x + w +5 , y + h + 5), (140, 190, 255), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 10, y +40), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)
                    
                    if fingers1[2] == 1:
                        cv2.rectangle(img, button.pos, (x +w, y + h), (190, 170, 255), cv2.FILLED)
                        cv2.putText(img, button.text, (x +10, y +40), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4)
                        sleep(0.25)
                        case = 6
                        
        cv2.imshow('1', img)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
            
    if case == 6:
        _, img = cap.read()
        img, results = detect_mediapipe(img, holistic) 
        try:
            landmark_subset = landmark_pb2.NormalizedLandmarkList(
                                    landmark = [
                                                results.pose_landmarks.landmark[0],
                                                results.pose_landmarks.landmark[11],
                                                results.pose_landmarks.landmark[12],
                                                results.pose_landmarks.landmark[13],
                                                results.pose_landmarks.landmark[14],
                                                results.pose_landmarks.landmark[15],
                                                results.pose_landmarks.landmark[16],
                                                results.pose_landmarks.landmark[23],
                                                results.pose_landmarks.landmark[24],
                                                results.pose_landmarks.landmark[25],
                                                results.pose_landmarks.landmark[26],
                                                results.pose_landmarks.landmark[27],
                                                results.pose_landmarks.landmark[28],
                                                results.pose_landmarks.landmark[29],
                                                results.pose_landmarks.landmark[30],
                                                results.pose_landmarks.landmark[31],
                                                results.pose_landmarks.landmark[32]]
                                    )
            
        except:
            cv2.putText(img, 'step back', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 125, 60),2, cv2.LINE_AA)
            pass
        
        mp_drawing.draw_landmarks(img, landmark_subset,  remove_landmarks.redefine_pose_connection.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2,circle_radius=4),
                        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2,circle_radius=4)
                        )
        
        cv2.imshow('1', img)
        if cv2.waitKey(10) & 0xFF == ord('q'): # break video
            break
        
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data, col_list)
print(df)

"""가지고 있는 정보
    Hand - Dict(type)
    lmList  ex) lmList = hand1["lmList"]       ## 21 landmarks, index, list type, pixel value(not normalized)
    bbox    ex) bbox1 = hand1["bbox"]          ## Bounding box info x, y, w, h
    center  ex) centerPoint1 = hand1["center"] ## center of the hand cx, cy
    type    ex) handType1 = hand1["type"]      ## Hand Type Left or Right
    fingers1 = detector.fingersUp(hand1)       ## List type, finger up -> 1, finger down -> 0
"""