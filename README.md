# Code-Scene-aware-VAD
Scene aware VAD using Reconstruction-based OCC method

## Model Architecture
<img src="https://github.com/user-attachments/assets/2acf5983-ea46-4615-b451-77e641a9975f" width="700"/>

## Training method
<img src="https://github.com/user-attachments/assets/b7c18b8c-eafc-4c2a-afce-37f5b7090677" width="700"/>

## Results
|     AUC                  |Training method   |Shanghai-SD    |
|:------------------------:|:-----------:|:-----------:|
| Scene-Independent method  |reconstruction        |57.9%        |
| **Scene-Aware method**   |**reconstruction + contrastive**        |**77.5%**        |

## Qualitative Evaluation

- **Scene 1 (normal: walking, standing, sitting, bicycle, motorcycle)**
  
| Aware     | Status                                                                | frame (160th) |Anomaly Score |
|-----------|------------------------------------------------------------------------|-------|-------|
| ❌ | **bicycle: normal**  | <img src="https://github.com/user-attachments/assets/bf046ade-09e0-4320-b53e-7946200526cf" width="400"/>  |<img src="https://github.com/user-attachments/assets/0c9566fb-205c-41d7-a5d0-4384b87bca8a" width="500"/>|
| ✅ | **bicycle: normal**| <img src="https://github.com/user-attachments/assets/bf046ade-09e0-4320-b53e-7946200526cf" width="400"/>  |<img src="https://github.com/user-attachments/assets/d48304bb-7b72-4ebb-a5a4-6aa8004449c7" width="500"/>|

- **Scene 2 (normal: walking, standing, sitting)**
  
| Aware     | Status                                                                | frame (130th) |Anomaly Score |
|-----------|------------------------------------------------------------------------|-------|-------|
| ❌ | **bicycle: abnormal**  | <img src="https://github.com/user-attachments/assets/69abefff-0712-4e10-848c-8266e3a38348" width="400"/>  |<img src="https://github.com/user-attachments/assets/0108f066-402d-4eda-b0fe-c3815bd86ddc" width="500"/>|
| ✅ | **bicycle: abnormal**| <img src="https://github.com/user-attachments/assets/69abefff-0712-4e10-848c-8266e3a38348" width="400"/>  |<img src="https://github.com/user-attachments/assets/dd634e8f-6ad8-4db8-bc3d-1b09ee435cc9" width="500"/>|

- **Scene 3 (normal: walking, standing, sitting)**
  
| Aware     | Status                                                                | frame (160th) |Anomaly Score |
|-----------|------------------------------------------------------------------------|-------|-------|
| ❌ | **motorcycle: abnormal**  | <img src="https://github.com/user-attachments/assets/794894f8-a80d-49cb-b474-f3c22215e0ee" width="400"/>  |<img src="https://github.com/user-attachments/assets/9237603e-06a2-480e-9ee8-01098a26ed94" width="500"/>|
| ✅ | **motorcycle: abnormal**| <img src="https://github.com/user-attachments/assets/794894f8-a80d-49cb-b474-f3c22215e0ee" width="400"/>  |<img src="https://github.com/user-attachments/assets/7a8c49ef-bf43-4051-8577-ddd7ea3e71ec" width="500"/>|

