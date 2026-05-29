import math
import numpy as np
import joblib
import pandas as pd
from backend.config import (
    SPEED_LIMIT,
    LINE_Y,
    LINE_OFFSET,
    FRAME_WIDTH,
    FRAME_HEIGHT
)
from backend.logger import logger
class AnalyticsService:
    def __init__(self):
        self.counted_ids = set()
        self.total_vehicles = 0
        self.line_y = LINE_Y
        self.offset = LINE_OFFSET
        self.in_count = 0
        self.out_count = 0
        self.previous_positions = {}
        self.vehicle_speeds = {}
        self.stationary_frames = {}
        self.anomaly_vehicles = set()
        self.speed_limit = SPEED_LIMIT
        self.heatmap = None
        self.lane_counts = {1:0,2:0,3:0}
        self.density_model = joblib.load("backend/ml/density_model.pkl")
        self.anomaly_model = joblib.load("backend/ml/anomaly_model.pkl")
        self.accident_zones = []
        self.accident_alert = set()
    def analyze(self, detections, objects):
        vehicle_count = 0
        if self.heatmap is None:
            self.heatmap = np.zeros( (FRAME_HEIGHT, FRAME_WIDTH), dtype=np.float32)
        self.lane_counts = {1: 0,2: 0,3: 0}
        vehicle_classes = [
            "car",
            "bus",
            "truck",
            "motorcycle",
            "bicycle"
        ]
        for det in detections:
            if det["label"] in vehicle_classes:
                vehicle_count += 1
        for object_id, centroid in objects.items():
            cx, cy = centroid
            previous_position = self.previous_positions.get( object_id, centroid)
            px, py = previous_position
            distance = math.sqrt((cx - px) ** 2 +(cy - py) ** 2)
            speed = distance * 2
            if speed < 1.5 and self.vehicle_speeds.get(object_id, 0) > 5:
                self.stationary_frames[object_id] = self.stationary_frames.get(object_id, 0) + 1
                if self.stationary_frames[object_id] > 20:
                    self.accident_zones.append((cx, cy))
                    logger.warning(f" POSSIBLE ACCIDENT: sudden stop at {cx},{cy}")
            if len(self.accident_zones) >= 2:
                cluster_center_x = sum([p[0] for p in self.accident_zones]) / len(self.accident_zones)
                cluster_center_y = sum([p[1] for p in self.accident_zones]) / len(self.accident_zones)
                self.accident_alert.add("accident_detected")
                logger.critical(f" ACCIDENT ZONE DETECTED at ({cluster_center_x:.1f}, {cluster_center_y:.1f})")
            self.vehicle_speeds[object_id] = int(speed)
            frame_height, frame_width = self.heatmap.shape
            lane_width = frame_width // 3
            if cx < lane_width:
                lane = 1
            elif cx < lane_width * 2:
                lane = 2
            else:
                lane = 3
            self.lane_counts[lane] += 1
            logger.info(f"Vehicle {object_id} in lane {lane}")
            if 0 <= cy < FRAME_HEIGHT and 0 <= cx < FRAME_WIDTH:
                self.heatmap[cy, cx] += 1
            if speed > self.speed_limit:
                logger.warning(f"Overspeed Vehicle ID {object_id}")
            if (self.line_y - self.offset < cy < self.line_y + self.offset):
                if object_id not in self.counted_ids:
                    self.counted_ids.add(object_id)
                    logger.info(f"Vehicle {object_id} counted")
                    if cy > py:
                       self.in_count += 1
                    elif cy < py:
                       self.out_count += 1
                self.total_vehicles = self.in_count + self.out_count
            self.previous_positions[object_id] = (cx, cy)
        avg_speed = 0
        traffic_density = "Low"
        speeds = list(self.vehicle_speeds.values())
        if len(speeds) > 0:
            avg_speed = sum(speeds) / len(speeds)
            features = pd.DataFrame([{"vehicle_count": vehicle_count,"avg_speed": avg_speed,"vehicles_in": self.in_count,"vehicles_out": self.out_count}])
            traffic_density = self.density_model.predict(features)[0]
            prediction = self.anomaly_model.predict(features)[0]
            if prediction == -1:
                self.anomaly_vehicles.add("traffic_anomaly")
                logger.warning("ML TRAFFIC ANOMALY DETECTED")    
        analytics = {
            "current_vehicle_count": vehicle_count,
            "total_vehicles_seen": self.total_vehicles,
            "traffic_density": traffic_density,
            "vehicles_in": self.in_count,
            "vehicles_out": self.out_count,
            "vehicle_speeds": self.vehicle_speeds,
            "lane_counts": self.lane_counts,
            "heatmap": self.heatmap,
            "anomaly_vehicles": list(self.anomaly_vehicles),
            "accident_alert": list(self.accident_alert)
        }
        return analytics