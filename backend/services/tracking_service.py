import numpy as np
from scipy.spatial import distance
class TrackingService:
    def __init__(self):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = 50
    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
    def update(self, detections):
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects
        input_centroids = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cX = int((x1 + x2) / 2)
            cY = int((y1 + y2) / 2)
            input_centroids.append((cX, cY))
        input_centroids = np.array(input_centroids)
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = distance.cdist(object_centroids, input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows = set()
            used_cols = set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)
            unused_rows = set(range(0, D.shape[0])) - used_rows
            unused_cols = set(range(0, D.shape[1])) - used_cols
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            for col in unused_cols:
                self.register(input_centroids[col])
        return self.objects