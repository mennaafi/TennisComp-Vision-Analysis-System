

<div align="center">  

# 👁️Tennis Analysis System👁️   

[![Watch the video](https://github.com/user-attachments/assets/25cfceab-2b42-46b4-b533-3f6c3baf505e)](https://github.com/user-attachments/assets/25cfceab-2b42-46b4-b533-3f6c3baf505e)  
  
</div>



#  🏀 Stage |  

 
- Player Detection using ulralytics and YOLOv8.
- Ball Detection , i will Fine tune and train my own YOLO on my own custom dataset.
- Tracking  
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
here in this files :
- inferance_YOLO.py
- inferance_YOLO_ ball.py
I played with YOLOv8 to detect players and ball , but unfortunately


| Video                                   |   
|-----------------------------------------|  
| <video width="385" height="250" controls> <source src="https://github.com/user-attachments/assets/78fe1c22-1709-4505-b236-90081f7e7046" type="video/mp4"> Your browser does not support the video tag. </video> |
