from collections import OrderedDict

import numpy as np
from constants import constants
from scipy.spatial import distance as dist


class CentroidTracker():
    def __init__(self):
        self.person_id_centroids = OrderedDict()

    def initialize(self):
        for i in constants.cams_in_use:
            self.person_id_centroids[i] = OrderedDict()

    def update(self, person_ids, yolo_centroids, recognizer_centroids, cam_index):
        detected_faces_greater_than_centroids = len(recognizer_centroids) >= len(self.person_id_centroids[cam_index].values())
        is_first_detection = len(self.person_id_centroids[cam_index].values()) == 0
        recognized_faces_equals_one = len(recognizer_centroids) == 1
        if is_first_detection or detected_faces_greater_than_centroids or recognized_faces_equals_one:
            if len(person_ids) != 0 and len(recognizer_centroids) != 0:
                y_centroids = np.array(yolo_centroids)
                r_centroids = np.array(recognizer_centroids)
                D = dist.cdist(r_centroids, y_centroids, metric="euclidean")

                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]
                usedRows = set()
                usedCols = set()

                for (row, col) in zip(rows, cols):
                    if row in usedRows or col in usedCols:
                        continue

                    self.person_id_centroids[cam_index][person_ids[row]] = y_centroids[col]

                    usedRows.add(row)
                    usedCols.add(col)
                """
                unusedRows = set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[1])).difference(usedCols)
                for row, col in zip(unusedRows, unusedCols):
                    self.person_id_centroids[cam_index] = OrderedDict(
                        (person_ids[row] if k == self.person_id_centroids[cam_index][row] else k, v) for k, v in
                        self.person_id_centroids[cam_index])"""

        else:
            y_centroids = np.array(yolo_centroids)
            pi_centroids = np.array(list(self.person_id_centroids[cam_index].values()))
            person_indexes = list(self.person_id_centroids[cam_index].keys())
            D = dist.cdist(pi_centroids, y_centroids, metric="euclidean")

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                self.person_id_centroids[cam_index][person_indexes[row]] = y_centroids[col]
                usedRows.add(row)
                usedCols.add(col)

        for staff_id, value in constants.missing_staffs[cam_index].items():
            if value == True and staff_id in self.person_id_centroids[cam_index].keys():
                del self.person_id_centroids[cam_index][staff_id]

        return self.person_id_centroids[cam_index]
