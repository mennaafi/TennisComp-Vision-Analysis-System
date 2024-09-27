

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
- Using ultrlytics , I will track player through video , and since we have only one ball , so we don`t have to track it.

To track persons through frames using the YOLOv8 model, use the following code:  

```python  
from ultralytics import YOLO  

model = YOLO("yolov8x")  

# Tracking person through frames  
results = model.track('inputs/input_video.mp4', conf=0.2, save=True)
```
### here is my output :)

  [![Watch the video](https://github.com/user-attachments/assets/e4c7afeb-e6aa-456c-8e64-a9a7987a2043)](https://github.com/user-attachments/assets/78fe1c22-1709-4505-b236-90081f7e7046)  



