import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_pytorch.deep_sort import DeepSort

# Load the YOLO model
model = YOLO("D://Python//ML//Vehicle Detection//best.pt")

# Initialize Deep SORT
deepsort = DeepSort("D:/Python/ML/Vehicle Detection/stats/ckpt.t7/ckpt.t7", 
                    nms_max_overlap=0.5,  
                    min_confidence=0.4,  
                    max_iou_distance=0.6,  
                    max_age=30,  
                    n_init=3)

# Set up video capture
cap = cv2.VideoCapture("./test6.mp4")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
output_file = "outputSort_Refined.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Helper to convert bounding box format
def xyxy_to_xywh(bbox):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2
    yc = y1 + h / 2
    return [xc, yc, w, h]

# Class names
class_names = model.names

# Initialize storage
prev_position = {}
velocity_threshold = 3.0
meters_per_pixel = 0.1
lstm_input_buffer = {}
TIME_STEPS = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detections
    results = model(frame, stream=True)
    detections, confidences, classes = [], [], []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  
            confidence = box.conf[0]
            class_id = int(box.cls[0])

            if confidence > 0.5:
                [x1, y1, x2, y2] = xyxy_to_xywh([x1, y1, x2, y2])
                detections.append([x1, y1, x2, y2])
                confidences.append(confidence)
                classes.append(class_id)

    if detections:
        detections = np.array(detections)
        confidences = np.array(confidences)
        classes = np.array(classes)
        outputs, _ = deepsort.update(detections, confidences, classes, frame)

        if outputs is not None:
            for track in outputs:
                if len(track) == 6:
                    x1, y1, x2, y2, track_cls, track_id = track
                    vehicle_name = class_names[track_cls]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    current_position = (center_x, center_y)

                    if track_id in prev_position:
                        prev_pos = prev_position[track_id]
                        delta_x = current_position[0] - prev_pos[0]
                        delta_y = current_position[1] - prev_pos[1]
                        displacement = np.sqrt(delta_x**2 + delta_y**2)

                        if displacement > velocity_threshold:
                            displacement_meters = displacement * meters_per_pixel
                            velocity_m_per_s = displacement_meters * fps
                            velocity_km_per_h = velocity_m_per_s * 3.6
                        else:
                            velocity_km_per_h = 0
                    else:
                        velocity_km_per_h = 0

                    prev_position[track_id] = current_position
                    if track_id not in lstm_input_buffer:
                        lstm_input_buffer[track_id] = []
                    
                    lstm_input_buffer[track_id].append([x1, y1, x2, y2, velocity_km_per_h])
                    if len(lstm_input_buffer[track_id]) > TIME_STEPS:
                        lstm_input_buffer[track_id].pop(0)

                    # Display
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'ID: {int(track_id)} {vehicle_name}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f'Velocity: {velocity_km_per_h:.2f} km/h', (int(x1), int(y2) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    out.write(frame)
    cv2.imshow("YOLOv8 with Deep SORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved as: {output_file}")
