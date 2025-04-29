from ultralytics import YOLO
## for
model = YOLO("models/best.pt")

# results = model.predict("inputs/image.png" , save = True)
results = model.predict("inputs/input_video.mp4" ,conf=0.2, save = True)
print(results)
print("Boxes : ")
# display boxes of the first frame in video
for box in results[0].boxes:
    print(box)

