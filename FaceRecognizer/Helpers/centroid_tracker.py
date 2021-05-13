from collections import OrderedDict

import numpy as np
from Constants import constants
from scipy.spatial import distance as dist


class CentroidTracker():
    def __init__(self):
        self.person_id_centroids = OrderedDict()
        self.initialized = False

    def initialize(self):
        if self.initialized == False:
            for i in constants.cams_in_use:
                self.person_id_centroids[i] = OrderedDict()
            self.initialized = True
        else:
            return

    def update(self, person_ids, yolo_centroids, recognizer_centroids, camIndex):
        if len(self.person_id_centroids[camIndex].values()) == 0 or len(recognizer_centroids) >= len(self.person_id_centroids[camIndex].values()):
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
                    
                    self.person_id_centroids[camIndex][person_ids[row]] = y_centroids[col]

                    usedRows.add(row)
                    usedCols.add(col)

        else:
            y_centroids = np.array(yolo_centroids)
            pi_centroids = np.array(list(self.person_id_centroids[camIndex].values()))
            person_indexes = list(self.person_id_centroids[camIndex].keys())
            D = dist.cdist(pi_centroids, y_centroids, metric="euclidean")

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                self.person_id_centroids[camIndex][person_indexes[row]] = y_centroids[col]
                usedRows.add(row)
                usedCols.add(col)

        for staff_id in constants.missing_staffs[camIndex].keys():
            if constants.missing_staffs[camIndex][staff_id] and staff_id in self.person_id_centroids[camIndex].keys():
                del self.person_id_centroids[camIndex][staff_id]

        return self.person_id_centroids[camIndex]
