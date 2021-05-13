from collections import OrderedDict
import pickle
from Helpers.person import PersonTracker
from Helpers.centroid_tracker import CentroidTracker

CLASSESFILE = '/media/ylmz/ARSIV/ubuntu/weights/class.names'
MODELCONFIG = '/media/ylmz/ARSIV/ubuntu/weights/yolo.cfg'
MODELWEIGHTS = '/media/ylmz/ARSIV/ubuntu/weights/yolov4-obj_final.weights'
ENCODINGS_PATH = '/home/ylmz/PycharmProjects/encodings.pickle'
DETECTION_METHOD = 'cnn'
DATA = pickle.loads(open(ENCODINGS_PATH, "rb").read())
with open(CLASSESFILE, 'rt') as f: CLASSNAMES = f.read().rstrip('\n').split('\n')

TARGET_WH = 320
THRESHOLD = 0.85
NMS_THRESHOLD = 0.3

cams_in_use = []

yolo_points = OrderedDict()
recognizer_points = OrderedDict()
person_ids = OrderedDict()
missing_staffs = OrderedDict()
pt = OrderedDict()
ct = CentroidTracker()
dicts = {"yolo_points": yolo_points,
        "recognizer_points": recognizer_points,
        "person_ids": person_ids}

def initialize():
    for i in cams_in_use:
        yolo_points[i] = []
        recognizer_points[i] = []
        missing_staffs[i] = OrderedDict()
        person_ids[i] = []
        pt[i] = PersonTracker()

# generalized function to add to dicts
def append(dict_name, to_add, cam_idx):
    dicts[dict_name][cam_idx].append(to_add)

def clear_ordered_dicts():
    for i in cams_in_use:
        yolo_points[i].clear()
        recognizer_points[i].clear()
        person_ids[i].clear()