from ultralytics import YOLO

model = YOLO("models/best.pt")
# results = model.predict("inputs/image.png" , save = True)
results = model.predict("inputs/input_video.mp4" ,conf=0.2, save = True)
print(results)
print("Boxes : ")
for box in results[0].boxes:
    print(box)

