

<div align="center">  

# 👁️Tennis Analysis System👁️   

[![Watch the video](https://github.com/user-attachments/assets/25cfceab-2b42-46b4-b533-3f6c3baf505e)](https://github.com/user-attachments/assets/25cfceab-2b42-46b4-b533-3f6c3baf505e)  
  
</div>



#  🏀 Stage |  

 
- Player Detection using ulralytics and YOLOv8.
- Ball Detection , i will Fine tune and train my own YOLO on my own custom dataset.
- Tracking Objects.
- Court Key Points Detection , i will train a CNN with pytorch to extract keypoints


### 1️⃣ Let`s start with my inputs file , it contains input video and img of video first frame :


| 📷 Video[0]                                            | 🎥 Video                                    |  
|------------------------------------------------------|---------------------------------------------|  
| ![Input Image](https://github.com/user-attachments/assets/3f6e0887-53b0-4b8a-8320-e256ac6709cc) | [Watch Video](https://github.com/mennaafi/TennisComp-Vision-Analysis-System/blob/main/inputs/input_video.mp4) |


## 2️⃣ Player Detction 🏃‍♂️  
 - from ulttralytics I used YOLOv8 model to detect players.
   

To use the YOLO model for predictions, you can use the following code:  

```python  
from ultralytics import YOLO  

# Load the YOLO model  
model = YOLO("yolov8x")

# Predict on a video  
results = model.predict("inputs/input_video.mp4", save=True

```
here in this files , I played with YOLOv8 to detect players and ball 🤸‍♀️
- inferance_YOLO.py
- inferance_YOLO_ ball.py

### this is my output video : 

[![Watch the video](https://github.com/user-attachments/assets/25cfceab-2b42-46b4-b533-3f6c3baf505e)](https://github.com/user-attachments/assets/78fe1c22-1709-4505-b236-90081f7e7046)


## 📌 Note :

 > - ### After reviewing the results, I noticed that while the model detects players very well, the detection of the ball is significantly lower across the frames. Therefore, I will use another method to address this issue.




## 3️⃣ Ball Detction 🏀
- I will fine tune a detector model to detect ball better to utilize its output a little bit better.
- Using dataset from Robowflow : https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection
- Fine tuning YOLOv5 from ultralytics , YOLOv5 it gives me best output.

### here is my output :)

[![Watch the image](https://github.com/user-attachments/assets/760b034c-05b2-4a7a-91ec-101fcc786c64)](https://github.com/user-attachments/assets/760b034c-05b2-4a7a-91ec-101fcc786c64)


## 4️⃣ Tracking objects 🔎
- Using ultrlytics , I will track players through video , and since we have only one ball , so we don`t have to track it.

To track persons through frames using the YOLOv8 model, use the following code:  

```python  
from ultralytics import YOLO  

model = YOLO("yolov8x")  

# Tracking person through frames  
results = model.track('inputs/input_video.mp4', conf=0.2, save=True)
```
### here is my output :)

[Watch the video](https://github.com/user-attachments/assets/c613ef69-d955-439a-b485-4fdddd778e53)

## 5️⃣ Court KeyPoints Detection 🏟️
- Using  this dataset : https://drive.google.com/file/d/1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta/view?usp=sharing
- I used feature extraction technique, model modification, and training process for keypoint extraction using a CNN in PyTorch

To adapt the ResNet-50 model for keypoint detection, we modify the final fully connected layer as follows:

```python
import torch
from torchvision import models

# Load the pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)

# Modify the final layer to output keypoints (14 keypoints, 2 coordinates each)
model.fc = torch.nn.Linear(model.fc.in_features, 14 * 2)
```






#  🏀 Stage ||


###  ✅ In video_utils.py file : we start to read video frames and save them .


###  ✅ In player_tracker.py file : 

- The PlayerTracker class helps track players in video frames using a YOLO model.
- It starts by loading the YOLOv8 model to quickly detect players.
- The class picks and filters player detections based on how close they are to important court positions.
- It finds the two players closest to specific court keypoints by looking at the centers of their bounding boxes.
- The system efficiently processes multiple video frames and can read existing detections from a saved file.
- For each individual frame, it accurately identifies players and connects their track IDs to their bounding box positions.
- Finally, the system adds labels and draws boxes around players in the video frames, making it easy to see their locations.



### ✅ In ball_tracker.py  , court_line_detector.py files :  I used same steps for ball tracker but my model was YOLOv5 , and for court keypoints i used model keypoints_model.pth
 > - ### You can find my models in models file , there is a link for my Google drive.


###  ✅ Now let`s draw mini court it`s vey good in visualization but also will help us when we want to determine the distance between ny two things.


| Tennis Court Dimensions | Screenshot |  
|------------------------|------------|  
| ![doubles-tennis-court-dimensions-598x381](https://github.com/user-attachments/assets/c1f391d9-f1d9-471a-ae94-c54e237e3c4a) | ![Screenshot (87)](https://github.com/user-attachments/assets/e330f186-147b-4937-956f-90778f1c1794) |




