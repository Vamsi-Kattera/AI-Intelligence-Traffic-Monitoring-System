import cv2

def draw_detections(frame, detections):

    for det in detections:

        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]

        text = f"{label} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return frame