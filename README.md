# Code-Scene-aware-VAD
Scene aware VAD using Reconstruction-based OCC method

## Qualitative Evaluation

- **Scene 1 (normal: walking, standing, sitting, bicycle, motorcycle)**
  
| Scene-aware     | Status                                                                | frame (160th) |Anomaly Score |
|-----------|------------------------------------------------------------------------|-------|-------|
| ❌ | **bicycle: normal**  | <img src="https://github.com/user-attachments/assets/bf046ade-09e0-4320-b53e-7946200526cf" width="400"/>  |<img src="https://github.com/user-attachments/assets/0c9566fb-205c-41d7-a5d0-4384b87bca8a" width="500"/>|
| ✅ | **bicycle: normal**| <img src="https://github.com/user-attachments/assets/bf046ade-09e0-4320-b53e-7946200526cf" width="400"/>  |<img src="https://github.com/user-attachments/assets/d48304bb-7b72-4ebb-a5a4-6aa8004449c7" width="500"/>|

- **Scene 2 (normal: walking, standing)**
  
| Scene-aware     | Status                                                                | frame (130th) |Anomaly Score |
|-----------|------------------------------------------------------------------------|-------|-------|
| ❌ | **bicycle: abnormal**  | <img src="https://github.com/user-attachments/assets/69abefff-0712-4e10-848c-8266e3a38348" width="400"/>  |<img src="https://github.com/user-attachments/assets/0108f066-402d-4eda-b0fe-c3815bd86ddc" width="500"/>|
| ✅ | **bicycle: abnormal**| <img src="https://github.com/user-attachments/assets/69abefff-0712-4e10-848c-8266e3a38348" width="400"/>  |<img src="https://github.com/user-attachments/assets/dd634e8f-6ad8-4db8-bc3d-1b09ee435cc9" width="500"/>|

- **Scene 3 (normal: walking, standing)**
  
| Scene-aware     | Status                                                                | frame (160th) |Anomaly Score |
|-----------|------------------------------------------------------------------------|-------|-------|
| ❌ | **motorcycle: abnormal**  | <img src="https://github.com/user-attachments/assets/794894f8-a80d-49cb-b474-f3c22215e0ee" width="400"/>  |<img src="https://github.com/user-attachments/assets/9237603e-06a2-480e-9ee8-01098a26ed94" width="500"/>|
| ✅ | **motorcycle: abnormal**| <img src="https://github.com/user-attachments/assets/794894f8-a80d-49cb-b474-f3c22215e0ee" width="400"/>  |<img src="https://github.com/user-attachments/assets/7a8c49ef-bf43-4051-8577-ddd7ea3e71ec" width="500"/>|

