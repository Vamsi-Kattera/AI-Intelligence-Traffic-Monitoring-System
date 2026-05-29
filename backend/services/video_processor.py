import cv2
from ultralytics import YOLO
from backend.services.analytics_service import AnalyticsService
from backend.database.analytics_data import analytics_data
from backend.config import USE_WEBCAM, VIDEO_PATH
import json
from backend.config import (
    USE_WEBCAM,
    VIDEO_PATH,
    WEBCAM_INDEX,
    MODEL_PATH
)
analytics_service = AnalyticsService()
model = YOLO(MODEL_PATH)
if USE_WEBCAM:
    cap = cv2.VideoCapture(WEBCAM_INDEX)
else:
    cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error opening video file")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # YOLOv8 + BYTE TRACK
    results = model.track( frame, persist=True, tracker="bytetrack.yaml" )
    result = results[0]
    detections = []
    objects = {}
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            detections.append({ "bbox": (x1, y1, x2, y2), "label": label, "confidence": conf })
            # TRACK ID
            if box.id is not None:
                track_id = int(box.id[0])
                centroid = ( int((x1 + x2) / 2), int((y1 + y2) / 2) )
                objects[track_id] = centroid
    analytics = analytics_service.analyze(detections, objects)
    with open(
        "backend/database/live_analytics.json", "w") as file:
        json.dump({
            "current_vehicle_count":
                analytics["current_vehicle_count"],
            "total_vehicles_seen":
                analytics["total_vehicles_seen"],
            "traffic_density":
                analytics["traffic_density"],
            "vehicles_in":
                analytics["vehicles_in"],
            "vehicles_out":
                analytics["vehicles_out"]}, file)
    annotated_frame = result.plot()
    # HEATMAP VISUALIZATION
    heatmap = analytics["heatmap"]
    heatmap_normalized = cv2.normalize( heatmap, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    heatmap_colored = cv2.applyColorMap( heatmap_normalized,cv2.COLORMAP_JET)
    heatmap_colored = cv2.resize(heatmap_colored,( annotated_frame.shape[1], annotated_frame.shape[0]))
    annotated_frame = cv2.addWeighted(annotated_frame, 0.7, heatmap_colored, 0.3, 0)
    frame_width = annotated_frame.shape[1]
    lane_width = frame_width // 3
    cv2.line(annotated_frame,(lane_width, 0),(lane_width, annotated_frame.shape[0]),(255, 255, 255),1)
    cv2.line(annotated_frame,(lane_width * 2, 0),(lane_width * 2, annotated_frame.shape[0]),(255, 255, 255),1)
    for track_id, centroid in objects.items():
        speed = analytics["vehicle_speeds"].get(track_id, 0)
        if speed > analytics_service.speed_limit:
            cv2.putText( annotated_frame, "OVERSPEED", (centroid[0], centroid[1] + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.circle(annotated_frame,centroid,4,(0, 0, 255),-1)
        cv2.putText(annotated_frame,f"{speed} km/h",(centroid[0], centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 0, 255),1)
    cv2.line( annotated_frame, (0, analytics_service.line_y), (annotated_frame.shape[1], analytics_service.line_y), (0, 255, 0), 1 )
    cv2.putText( annotated_frame, f"Vehicles: {analytics['current_vehicle_count']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText( annotated_frame, f"Total Seen: {analytics['total_vehicles_seen']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText( annotated_frame, f"Traffic Density: {analytics['traffic_density']}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText( annotated_frame, f"Vehicles In: {analytics['vehicles_in']}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText( annotated_frame, f"Vehicles Out: {analytics['vehicles_out']}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText( annotated_frame, f"Lane 1: {analytics['lane_counts'][1]}", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    cv2.putText( annotated_frame, f"Lane 2: {analytics['lane_counts'][2]}", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    cv2.putText( annotated_frame, f"Lane 3: {analytics['lane_counts'][3]}", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    cv2.imshow("Traffic Intelligence System", annotated_frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()