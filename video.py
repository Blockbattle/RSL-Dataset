import cv2
import os
import copy
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

folder = 'J'
number = 32
if not os.path.exists('F:/Russian_alphabet/skeletons/' + folder):
  os.mkdir('F:/Russian_alphabet/skeletons/' + folder)

fileX = open("landmarks_x.txt", "a")
fileY = open("landmarks_y.txt", "a")
fileLabel = open("labels.txt", "a")

idx = 1001
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #imageR = cv2.flip(image, 1)
    results = hands.process(image)
    
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
      hand_landmarks = results.multi_hand_landmarks[0]
      mp_drawing.draw_landmarks(
          image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    
      for i in range(len(hand_landmarks.landmark)):
        hand_landmarks.landmark[i].x = hand_landmarks.landmark[i].x * 4 / 3 #меняем разрешение с 4*3 на квадратное
      # Растягиваем координаты по размеру фона для скелета
      margin = 10
      minX = min(hand_landmarks.landmark, key=lambda i: i.x).x
      maxX = max(hand_landmarks.landmark, key=lambda i: i.x).x
      minY = min(hand_landmarks.landmark, key=lambda i: i.y).y
      maxY = max(hand_landmarks.landmark, key=lambda i: i.y).y

      w = maxX - minX
      h = maxY - minY

      maxLength = max(w, h)
      
      annotated_image = cv2.imread('background192.jpg')
      new_hand_landmarks = copy.deepcopy(hand_landmarks)

      for i in range(len(new_hand_landmarks.landmark)):
        new_hand_landmarks.landmark[i].x = ((new_hand_landmarks.landmark[i].x - minX) * (192 - 2 * margin) / maxLength + margin) / 192
        new_hand_landmarks.landmark[i].y = ((new_hand_landmarks.landmark[i].y - minY) * (192 - 2 * margin) / maxLength + margin) / 192
        fileX.write(f' {new_hand_landmarks.landmark[i].x:.7f}')
        fileY.write(f' {new_hand_landmarks.landmark[i].y:.7f}')
      
      fileX.write('\n')
      fileY.write('\n')
      fileLabel.write(str(number) + '\n')
      
      mp_drawing.draw_landmarks(
          annotated_image,
          new_hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
      cv2.imwrite('F:/Russian_alphabet/skeletons/' + folder + '/' + folder + str(idx) + '.jpg', annotated_image)
      
      idx = idx + 1
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

fileX.close()
fileY.close()
fileLabel.close()