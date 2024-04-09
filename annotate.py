import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utilities import draw_landmarks_on_image

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
vid = cv2.VideoCapture('./video_1.mp4')
n = 0
while(vid.isOpened()): 
    ret, frame = vid.read()
    if ret:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # STEP 4: Detect hand landmarks from the input image.
        detection_result = detector.detect(image)
        if len(detection_result.hand_landmarks):
            cv2.imwrite('{}_{}.{}'.format('./dataset/hands/', str(n).zfill(6), '.jpg'), frame)
            n += 1
        else:
            cv2.imwrite('{}_{}.{}'.format('./dataset/not_hands/', str(n).zfill(6), '.jpg'), frame)
            n += 1
    else:
        break
vid.release() 
cv2.destroyAllWindows() 