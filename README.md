

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


- PlayerTracker Class  is designed to facilitate the detection and tracking of players in video frames. It utilizes a YOLOv8  model for object detection, filtering out thw closest two detected players based on their proximity to specified court keypoints.
  
 ### Functions Summary ☑️ :

##### `__init__(self, model_path)`
Initializes the PlayerTracker by loading the YOLO model from the specified path.

##### `choose_and_filter_players(self, court_keypoints, player_detections)`
Selects and filters player detections based on their proximity to key court positions, returning only the closest players.

##### `choose_players(self, court_keypoints, player_dict)`
Identifies the two closest players to the specified court keypoints based on the distance from their bounding box centers.

##### `detect_frames(self, frames, read_from_stub=False, stub_path=None)`
Processes a list of video frames to detect players, with an option to read detections from a stub file for efficiency.

##### `detect_frame(self, frame)`
Detects players in a single video frame and returns a mapping of track IDs to their bounding box coordinates.

##### `draw_bboxes(self, video_frames, player_detections)`
Annotates video frames by drawing bounding boxes and player IDs based on detected players.



### ✅ In ball_tracker.py  , court_line_detector.py files :  I used same functions for ball tracker but my model was YOLOv5 , and for court keypoints i used model keypoints_model.pth
 > - ### You can find my models in models file , there is a link for my Google drive.



