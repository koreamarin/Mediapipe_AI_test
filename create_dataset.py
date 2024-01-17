import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['come', 'away', 'spin']
seq_length = 30         # 시퀀스 길이
secs_for_action = 30    # 학습 시간 (오래 학습시키고 싶다면 늘려도 됨)

# MediaPipe hands model
mp_hands = mp.solutions.hands           # mediapipe의 hand모델
mp_drawing = mp.solutions.drawing_utils # mediapipe의 그려주는 모델?
hands = mp_hands.Hands(
    max_num_hands=1,        # 손1개만 인식, 2로 하면 두손 인식함
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)       # OpenCV를 이용해서 웹캠을 사용

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)   # 데이터셋을 저장할 폴더를 만듦. 폴더이름은 dataset

while cap.isOpened():   # 캠이 켜져있다면
    for idx, action in enumerate(actions):  # action(제스쳐)마다 녹화, (화면상에 나타남)
        data = []

        ret, img = cap.read()   # 이미지를 읽음

        img = cv2.flip(img, 1)  # flip, 이미지를 반전시킴

        # 어떤 액션을 학습할 것인지 표시해줌
        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)   # 3초동안 대기탐. 준비해야함

        start_time = time.time()

        while time.time() - start_time < secs_for_action:       # 30초동안 반복
            ret, img = cap.read()       # 프레임을 읽음

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)     # 프레임을 읽어 만든 변수 img를 mediapipe에 넣음
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]    # 좌표와 보이는지 안보이는지에 대한 변수들을 뽑아서 joint 리스트에 넣음

                    # Compute angles between joints (joint에 있는 좌표를 이용하여 손가락 관절사이의 각도 구하기)
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    d = np.concatenate([joint.flatten(), angle_label])

                    data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)       # data를 30초동안 모았으면 numpy array형태로 변환시킴
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)      # 데이터를 파일로 저장

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data) # 시퀀싱한 데이터로 파일을 다시 저장
    break
