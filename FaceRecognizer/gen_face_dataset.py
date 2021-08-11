import cv2
import os
import uuid
import argparse
from utils import httpreq
from constants import constants

parser = argparse.ArgumentParser()
parser.add_argument("--name")
parser.add_argument("--lastName")
parser.add_argument("--phoneNumber")
args = parser.parse_args()
index = httpreq.send_staff_credentials(args.name,args.lastName,args.phoneNumber)
credentials = f"{args.name}_{args.lastName}_{index}"

cap = cv2.VideoCapture(0)
path = constants.FACE_DATASET_PATH
person_path = os.path.join(path, credentials)

if not os.path.exists(path):
    os.mkdir(path)

if not os.path.exists(person_path):
    os.mkdir(person_path)


total_photos = 0
while True:
    check, frame = cap.read()

    if check is not True:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if cv2.waitKey(10) & 0xFF == ord('s'):
        cv2.imwrite(os.path.join(person_path, f'{str(uuid.uuid4().int)[:6]}.jpg'), frame)
        total_photos += 1

    cv2.putText(frame, f'{total_photos} photos have saved', (250, 460), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    cv2.imshow('Gather Image', frame)

cap.release()
cv2.destroyAllWindows()
