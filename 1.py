from ultralytics import YOLO

model = YOLO("best(new).pt")

result = model.predict("1.mp4", show = True, save = True)