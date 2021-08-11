from imutils import paths
from constants import constants
import face_recognition
import pickle
import cv2
import os

imagePaths = list(paths.list_images(constants.FACE_DATASET_PATH))
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print(f"Processing {i + 1} of {len(imagePaths)}")
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model=constants.DETECTION_METHOD)
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

print("Saving encodings to encodings.pickle ...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(constants.ENCODINGS_PATH, "wb")
f.write(pickle.dumps(data))
f.close()
print("Encodings have been saved successfully.")
