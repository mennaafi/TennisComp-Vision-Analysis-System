## here : i am playing with ultralytics and Yolo


from ultralytics import YOLO

model = YOLO(" yolov8x ")
# detect person
# results = model.predict("inputs/image.png" , save = True)
# results = model.predict("inputs/input_video.mp4" , save = True)
# tracking person through frames
results = model.track('inputs/input_video.mp4',conf=0.2, save=True )

print(results)
print("Boxes : ")
# display boxes of the first frame in video
for box in results[0].boxes:
   print(box)

