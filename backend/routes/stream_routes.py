import cv2
import json
import csv
import os
from ultralytics import YOLO

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from backend.services.analytics_service import AnalyticsService
from backend.config import USE_WEBCAM, VIDEO_PATH

router = APIRouter()

from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

print("Loading model from:", MODEL_PATH)

model = YOLO(str(MODEL_PATH))

analytics_service = AnalyticsService()


def generate_frames():

    if USE_WEBCAM:

        cap = cv2.VideoCapture(0)

    else:

        cap = cv2.VideoCapture(VIDEO_PATH)

    while True:

        success, frame = cap.read()

        if not success:
            break

        # YOLO TRACKING
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml"
        )

        result = results[0]

        detections = []

        objects = {}

        if result.boxes is not None:

            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                cls = int(box.cls[0])

                label = model.names[cls]

                conf = float(box.conf[0])

                detections.append({

                    "bbox": (x1, y1, x2, y2),

                    "label": label,

                    "confidence": conf
                })

                # TRACKING ID
                if box.id is not None:

                    track_id = int(box.id[0])

                    centroid = (

                        int((x1 + x2) / 2),

                        int((y1 + y2) / 2)
                    )

                    objects[track_id] = centroid

        # ANALYTICS
        analytics = analytics_service.analyze(
            detections,
            objects
        )

        # DRAW YOLO RESULTS
        annotated_frame = result.plot()

        # DRAW COUNT LINE
        cv2.line(

            annotated_frame,

            (0, analytics_service.line_y),

            (
                annotated_frame.shape[1],
                analytics_service.line_y
            ),

            (0, 255, 0),

            2
        )

        # DRAW SPEEDS
        for track_id, centroid in objects.items():
            if(track_id in analytics["anomaly_vehicles"]):
                cv2.putText( annotated_frame, "ANOMALY",( centroid[0], centroid[1] - 40 ), cv2.FONT_HERSHEY_SIMPLEX, 0.7(0, 0, 255), 2 )

            speed = analytics[
                "vehicle_speeds"
            ].get(track_id, 0)

            cv2.putText(

                annotated_frame,

                f"{speed} km/h",

                (
                    centroid[0],
                    centroid[1] - 20
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (0, 0, 255),

                2
            )

        # OVERLAY TEXT
        cv2.putText(

            annotated_frame,

            f"Vehicles: {analytics['current_vehicle_count']}",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 0),

            2
        )

        cv2.putText(

            annotated_frame,

            f"Vehicles In: {analytics['vehicles_in']}",

            (20, 80),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 0),

            2
        )

        cv2.putText(

            annotated_frame,

            f"Vehicles Out: {analytics['vehicles_out']}",

            (20, 120),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 0, 255),

            2
        )

        # SAVE LIVE ANALYTICS
        with open(
            "backend/database/live_analytics.json",
            "w"
        ) as file:

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
                    analytics["vehicles_out"]

            }, file)
        csv_file = ("backend/database/traffic_data.csv")
        avg_speed = 0
        speeds = list(analytics["vehicle_speeds"].values())
        if len(speeds) > 0:
            avg_speed = (sum(speeds) / len(speeds))
        with open(csv_file, mode="a", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow([analytics["current_vehicle_count"], round(avg_speed, 2), analytics["vehicles_in"], analytics["vehicles_out"], analytics["traffic_density"]])

        # CONVERT FRAME
        _, buffer = cv2.imencode(
            ".jpg",
            annotated_frame
        )

        frame_bytes = buffer.tobytes()

        yield (

            b'--frame\r\n'

            b'Content-Type: image/jpeg\r\n\r\n'

            + frame_bytes +

            b'\r\n'
        )

    cap.release()


@router.get("/video_feed")
def video_feed():

    return StreamingResponse(

        generate_frames(),

        media_type=
        "multipart/x-mixed-replace; boundary=frame"
    )
