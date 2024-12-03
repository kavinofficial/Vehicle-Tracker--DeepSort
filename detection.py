from ultralytics import YOLO

model = YOLO("D:/Accident_detection/final train/best.pt")  # Replace with the path to your best model file

model.predict(source="./test1.mp4" , show = True , save = True )