from ultralytics import YOLO
class DetectionService:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.allowed_classes = [
            "car",
            "bus",
            "truck",
            "motorcycle",
            "bicycle",
            "person"
        ]
    def detect(self, frame):
        results = self.model(frame)
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = self.model.names[class_id]
                if label not in self.allowed_classes:
                    continue
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })
        return detections