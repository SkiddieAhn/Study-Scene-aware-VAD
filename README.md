# Scene-aware Video Anomaly Detection
Since the **definition of normal and abnormal events in video anomaly detection (VAD) can vary depending on the scene, it is crucial to design a model that is aware of scene information**. For instance, a car is considered normal on a road but abnormal on a pedestrian walkway. If a model is trained solely on normal data without distinguishing between different scenes, it may incorrectly classify ```scene-dependent abnormal objects``` (e.g., a car on a pedestrian) as normal during testing. To address this issue, I propose a simple yet effective **scene-aware VAD method**.

## Overview
To perform anomaly detection at the video segment level, segments from various scenes are utilized during training.
To enable fast training with a lightweight model, I operate at the feature level instead of using raw frames.  
For this purpose, I employ the **CLIP Image Encoder** based on ```ViT-L/14```, which extracts fine-grained semantic information from individual frames.
Subsequently, I perform reconstruction-based training using a **Transformer-based AutoEncoder** to model temporal dynamics.

<img src="https://github.com/user-attachments/assets/2acf5983-ea46-4615-b451-77e641a9975f" width="750"/>

## Method
In this study, I adopt a **reconstruction-based approach**, one of the commonly used **One-Class Classification (OCC)** methods in video anomaly detection. This approach enables the computation of anomaly scores in a straightforward manner by measuring the **reconstruction error** between the input and the output.
To equip the model with the ability to distinguish between different scenes, I additionally incorporate **contrastive learning**.
The objective is to learn **scene-specific normal manifolds** such that, during testing, features from abnormal frames that are inconsistent with the current scene deviate from the learned manifolds. This design makes it more difficult for the decoder to reconstruct those abnormal inputs, thereby resulting in higher reconstruction errors.

<img src="https://github.com/user-attachments/assets/b7c18b8c-eafc-4c2a-afce-37f5b7090677" width="750"/>

## Results
To evaluate whether the model can effectively handle anomalies that vary depending on the scene, I utilize ```ShanghaiTech-SD```, a **scene-dependent dataset**. Details of the dataset can be found in the following paper [[Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_A_New_Comprehensive_Benchmark_for_Semi-Supervised_Video_Anomaly_Detection_and_CVPR_2023_paper.pdf)].  
Experimental results show that our proposed model, which learns to distinguish between different scenes, achieves a **19.6% higher AUC**. This improvement demonstrates that **scene-specific normal manifolds are appropriately constructed**, allowing the model to effectively detect **abnormal frames that violate scene semantics**—such as a bicycle on a pedestrian walkway.


|     AUC                  |Training method   |ShanghaiTech-SD    |
|:------------------------:|:-----------:|:-----------:|
| Scene-agnostic method  |reconstruction        |57.9%        |
| **Scene-aware method**   |**reconstruction + contrastive**        |**77.5%**        |


## Qualitative Evaluation
Scene1 is a general-purpose road where bicycles and motorcycles are allowed, while Scene2 and Scene3 are pedestrian-only areas.
A **scene-agnostic model**, which does not take scene context into account, tends to assign **low anomaly scores** to scene-dependent anomalies such as a ```bicycle appearing in Scene2```.
In contrast, the proposed **scene-aware model** gives **higher anomaly scores** in such cases, effectively detecting situations that don't fit the scene.

- **Scene 1 (normal: walking, standing, sitting, bicycle, motorcycle)**
  
| Aware     | Status                                                                | frame (160th) |Anomaly Score |
|-----------|------------------------------------------------------------------------|-------|-------|
| ❌ | **bicycle: normal**  | <img src="https://github.com/user-attachments/assets/bf046ade-09e0-4320-b53e-7946200526cf" width="400"/>  |<img src="https://github.com/user-attachments/assets/0c9566fb-205c-41d7-a5d0-4384b87bca8a" width="600"/>|
| ✅ | **bicycle: normal**| <img src="https://github.com/user-attachments/assets/bf046ade-09e0-4320-b53e-7946200526cf" width="400"/>  |<img src="https://github.com/user-attachments/assets/d48304bb-7b72-4ebb-a5a4-6aa8004449c7" width="600"/>|

- **Scene 2 (normal: walking, standing, sitting)**
  
| Aware     | Status                                                                | frame (130th) |Anomaly Score |
|-----------|------------------------------------------------------------------------|-------|-------|
| ❌ | **bicycle: abnormal**  | <img src="https://github.com/user-attachments/assets/69abefff-0712-4e10-848c-8266e3a38348" width="400"/>  |<img src="https://github.com/user-attachments/assets/0108f066-402d-4eda-b0fe-c3815bd86ddc" width="600"/>|
| ✅ | **bicycle: abnormal**| <img src="https://github.com/user-attachments/assets/69abefff-0712-4e10-848c-8266e3a38348" width="400"/>  |<img src="https://github.com/user-attachments/assets/dd634e8f-6ad8-4db8-bc3d-1b09ee435cc9" width="600"/>|

- **Scene 3 (normal: walking, standing, sitting)**
  
| Aware     | Status                                                                | frame (160th) |Anomaly Score |
|-----------|------------------------------------------------------------------------|-------|-------|
| ❌ | **motorcycle: abnormal**  | <img src="https://github.com/user-attachments/assets/794894f8-a80d-49cb-b474-f3c22215e0ee" width="400"/>  |<img src="https://github.com/user-attachments/assets/9237603e-06a2-480e-9ee8-01098a26ed94" width="600"/>|
| ✅ | **motorcycle: abnormal**| <img src="https://github.com/user-attachments/assets/794894f8-a80d-49cb-b474-f3c22215e0ee" width="400"/>  |<img src="https://github.com/user-attachments/assets/7a8c49ef-bf43-4051-8577-ddd7ea3e71ec" width="600"/>|


## Visualization
To effectively visualize the **normal manifolds** of the **scene-agnostic** and **scene-aware** methods, I trained both approaches using a ```Variational AutoEncoder (VAE)```.
While both methods produce features that follow a **standard normal distribution**, the proposed scene-aware method additionally shows clear **separation by scene**.
Incidentally, the proposed method also achieved better performance even with the VAE. However, for more stable training, I used an AutoEncoder (AE), which resulted in even higher performance.

|     Scene-agnostic manifold                |Scene-aware manifold  |
|:------------------------:|:-----------:|
| <img src="https://github.com/user-attachments/assets/75a1fef8-4683-4d2a-bea6-1c01209e814d" width="450"/>| <img src="https://github.com/user-attachments/assets/94c34176-e198-4efb-9c4a-6ac576f4baf2" width="450"/>|


