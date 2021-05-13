import datetime
from collections import OrderedDict

import cv2
import face_recognition
import imutils
import numpy as np
import requests
from Constants import constants

net = cv2.dnn.readNetFromDarknet(constants.MODELCONFIG, constants.MODELWEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
class Helpers:
    
    def __init__(self, width, height):
        self.scale = 0.0
        self.cams = constants.cams_in_use
        self.width = width
        self.height = height
        self.drawing_frames = OrderedDict()

    def scale_box(self, box):
        return (int(x * self.scale) for x in box)

    def get_box_center(self, box):
        cx, cy = int((box[1] + box[3])/2), int((box[0] + box[2])/2)
        return cx, cy

    def get_name_id(self, name):
        # I add the id number which is related to name 
        # so we need to split the name by underscore character.
        split_name = name.split('_')
        return f"{split_name[0]}_{split_name[2]}"

    def get_id(self, name):
        split_name = name.split('_')
        return int(split_name[2])

    def check_staff_missing(self, center, name, cam_id):
        if 50 < center < (self.width - 50):
            constants.missing_staffs[cam_id][self.get_id(name)] = False

    def release_cams(self, cams):
        for cam in cams:
            if cam is not None:
                cam.release()

    def add_staff(self, names, cam_id):
        for current_name in names:
            if current_name not in constants.pt[cam_id].persons:
                constants.pt[cam_id].register(current_name, datetime.datetime.now())

    def get_recognized_face_names(self, rgb, data):
        boxes = face_recognition.face_locations(rgb, model=constants.DETECTION_METHOD)
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        for (index, encoding) in enumerate(encodings):
            name = ""
            matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.6)
            if True in matches:
                matched_indexes = [index for (index, match) in enumerate(matches) if match]
                name_counts = {}

                for i in matched_indexes:
                    name = data["names"][i]
                    name_counts[name] = name_counts.get(name) + 1 if name in name_counts.keys() else 1

                name = max(name_counts, key=name_counts.get)
            if name != "":
                names.append(name)
        return boxes, names

    def show_recognized_faces(self, frames):
        for cam_id in self.cams:
            frame = frames[cam_id]
            self.drawing_frames[cam_id] = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = imutils.resize(rgb, width=750)
            self.scale = frame.shape[1] / float(rgb.shape[1])
            (boxes, names) = self.get_recognized_face_names(rgb, constants.DATA)
            self.add_staff(names, cam_id)
            for (box, name) in zip(boxes, names):
                (top, right, bottom, left) = self.scale_box(box)
                cx, cy = self.get_box_center(box)
                y = top - 15 if top - 15 > 15 else top + 15
                self.check_staff_missing(cx, name, cam_id)
                constants.append("recognizer_points", (cx, cy), cam_id)
                constants.append("person_ids", self.get_id(name), cam_id)
                cv2.rectangle(self.drawing_frames[cam_id], (left, top), (right, bottom), (0, 0, 255), 1)
                cv2.rectangle(self.drawing_frames[cam_id], (left, y), (right, y + 15), (0, 0, 255), -1)
                cv2.putText(self.drawing_frames[cam_id], self.get_name_id(name).upper(), (left, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def yolo_find_objects(self, outputs, frame, drawing_frame, camIndex):
        hT, wT, cT = frame.shape
        bbox = []
        class_ids = []
        confs = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]  # remove first five elements.
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > constants.THRESHOLD:
                    box = detection[0:4] * np.array([wT, hT, wT, hT])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    bbox.append([x, y, int(width), int(height)])
                    class_ids.append(class_id)
                    confs.append(float(confidence))
        indices = cv2.dnn.NMSBoxes(bbox, confs, constants.THRESHOLD, constants.NMS_THRESHOLD)

        for index in indices:
            index = index[0]  # the index has extra brackets so we have to remove them.
            class_name_index = class_ids[index]

            box = bbox[index]
            (x, y, w, h) = box[0:4]
            label = constants.CLASSNAMES[class_name_index]
            conf = confs[index]
            cv2.rectangle(drawing_frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            constants.append("yolo_points",(cx, cy),camIndex)
            cv2.putText(drawing_frame, f'{label} {int(conf * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 0, 0), 1)

    def show_yolo_bboxes(self, frames):

        for cam_id in constants.cams_in_use:
            blob = cv2.dnn.blobFromImage(frames[cam_id], 1 / 255.0, (constants.TARGET_WH, constants.TARGET_WH),
                                        [0, 0, 0], 1, crop=False)
            net.setInput(blob)

            layer_names = net.getLayerNames()
            output_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            outputs = net.forward(output_names)
            self.yolo_find_objects(outputs, frames[cam_id], self.drawing_frames[cam_id], cam_id)

    def track(self):
        for cam_id in constants.cams_in_use:
            frame = self.drawing_frames[cam_id]
            if len(constants.yolo_points[cam_id]) != 0:
                objects = constants.ct.update(constants.person_ids[cam_id],
                                                constants.yolo_points[cam_id],
                                                constants.recognizer_points[cam_id],
                                                cam_id)

                for (objectID, centroid) in objects.items():
                    cv2.putText(frame, str(objectID), (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 4, (0, 255, 0), -1)


                    if 50 < int(centroid[0]) < (self.width - 50):
                        constants.missing_staffs[objectID] = False

                    if constants.missing_staffs[objectID]:
                        continue

                    if int(centroid[0]) < 50 or int(centroid[0]) > (self.width - 50):
                        response = requests.get(f'https://localhost:5001/api/Staff/{objectID}', verify=False).json()
                        fullname = f"{response['name']}_{response['lastName']}_{response['id']}"
                        constants.pt[cam_id].mark_person_disappeared(fullname, datetime.datetime.now())
                        constants.pt[cam_id].send_server(fullname)
                        constants.missing_staffs[objectID] = True
    
    def draw_to_screen(self):
        for cam_id in constants.cams_in_use:
            cv2.putText(self.drawing_frames[cam_id], f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}",(int(self.width / 2), self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(self.drawing_frames[cam_id], f"Room Id:{cam_id}", (int(self.width / 2) - 150, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.line(self.drawing_frames[cam_id], (self.width - 50, 0), (self.width - 50, self.height), (255, 255, 255), 2)
            cv2.line(self.drawing_frames[cam_id], (50, 0), (50, self.height), (255, 255, 255), 2)

    def get_concatenated_frames(self):
        return np.concatenate([value for key,value in self.drawing_frames.items()], axis=1)
