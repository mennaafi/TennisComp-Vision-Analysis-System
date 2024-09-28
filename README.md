

<div align="center">  

# 👁️Tennis Analysis System👁️   

Welcome to My Tennis Analysis System project! In this repository, I will leverage computer vision, and deep learning techniques to create a robust tennis analysis system. This project employs YOLO, a state-of-the-art object detector, to accurately identify players and tennis balls within video frames. Additionally, I implement advanced tracking methods to follow these objects seamlessly across frames. A custom Convolutional Neural Network (CNN) is developed to detect key points on the tennis court, enhancing the depth of analysis.

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






###  ✅ Now let`s draw mini court it`s vey good in visualization but also will help us when we want to determine the distance between any two things.


| Tennis Court Dimensions | output |  
|------------------------|------------|  
| ![doubles-tennis-court-dimensions-598x381](https://github.com/user-attachments/assets/c1f391d9-f1d9-471a-ae94-c54e237e3c4a) | ![Screenshot (87)](https://github.com/user-attachments/assets/e330f186-147b-4937-956f-90778f1c1794) |

with some research ,
Here are the dimensions relevant to the tennis court and players:  

```python  
SINGLE_LINE_WIDTH = 8.23  
DOUBLE_LINE_WIDTH = 10.97  
HALF_COURT_LINE_HEIGHT = 11.88  
SERVICE_LINE_WIDTH = 6.4  
DOUBLE_ALLY_DIFFERENCE = 1.37  
NO_MANS_LAND_HEIGHT = 5.48  

PLAYER_1_HEIGHT_METERS = 1.88  
PLAYER_2_HEIGHT_METERS = 1.91  
```

In mini_court.py file : The MiniCourt class is designed for visual analysis and representation of a tennis court within video frames. It provides functionalities to:

- Initialize a Mini Court: Sets up the dimensions and key positions for drawing the mini court.
- Coordinate Conversion: Converts real-world measurements (in meters) to pixel values, enabling accurate placement on the mini court.
- Drawing Capabilities: Draws various elements, including court lines, a net, and player positions using OpenCV, enhancing visual clarity.
- Player Position Tracking: Computes player coordinates relative to key points on the court and updates their positions based on changes in video frames.
- Frame Processing: Processes multiple frames to track player and ball movements, ensuring dynamic visualization.
- Visual Highlights: Marks player positions and movements through graphical annotations, making it easier to analyze gameplay.




















###  ✅ Detect Ball Shots 

- I noticed that ball detections in some frames are missing , so we should address this problem.
- im my ball_analysis_ipynb file :
   > - Loads ball position data from a pickle file (ball_detections.pkl).
   > - Extracts ball positions corresponding to a specific key (1 in this case) and converts the list into a pandas DataFrame with columns ['x1', 'y1', 'x2', 'y2'].
   > - Interpolates missing values in the DataFrame using linear interpolation , Fills any remaining missing values with the next valid observation (backward fill).
   > - Computes the midpoint of the ball's y-coordinates (mid_y) by averaging y1 and y2.
   > - Calculates a rolling mean of the midpoint over a window of 5 frames to smooth out the variations.
   > - Computes the difference in consecutive rolling means to detect changes in the ball's position (delta_y).
   > - Initializes a ball_hit column to track detected hits.
   > - Looks for significant changes in the position (delta_y) to identify ball hits, requiring a minimum frame change to confirm a hit.
   > - Collects the frame numbers where ball hits are detected into a list (frame_nums_with_ball_hits).













###  ✅ Convert Bounding boxes to mini court coodinates.

###  ✅ Convert Objects positions to mini court positions 





  




#  🏀 Stage ||| 

### Finally , we want to display a  statistical DataFrame for match analysis.

| Description | Screenshot |  
|-------------|------------|  
| Match Analysis Statistical DataFrame | ![Screenshot (89)](https://github.com/user-attachments/assets/05ce675d-3a98-480e-8327-8a55b55f05fb) |

In main.py file :

By analyzing and tracking player statistics during a match by focusing on ball shot events. It initializes a data structure to hold statistics for two players, iterating through the frames where the ball is shot to calculate the duration between shots, as well as the distance and speed of the ball. By determining which player is closest to the ball at the time of the shot, the code records that player's performance, including the total number of shots taken and their average shot speed. Additionally, it measures the opponent's speed during each shooting event. into a DataFrame, This comprehensive analysis provides valuable insights into player performance, enhancing the understanding of their contributions throughout the match.

## THE END ..
### Thank you for keep  reading :) 🌷






