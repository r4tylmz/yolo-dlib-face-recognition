from imutils import paths
import face_recognition
import pickle
import cv2
import os

# define the constants
ENCODINGS_PATH = 'encodings.pickle'
DETECTION_METHOD = 'cnn'
DATASET_PATH = '/home/ylmz/PycharmProjects/face_dataset'

imagePaths = list(paths.list_images(DATASET_PATH))

knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print(f"Processing {i + 1} of {len(imagePaths)}")
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model=DETECTION_METHOD)
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

print("Saving encodings to encodings.pickle ...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(ENCODINGS_PATH, "wb")
f.write(pickle.dumps(data))
f.close()
print("Encodings have been saved successfully.")
