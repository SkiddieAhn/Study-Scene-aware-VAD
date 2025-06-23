# Scene-aware Video Anomaly Detection
Since the **definition of normal and abnormal events in video anomaly detection (VAD) can vary depending on the scene, it is crucial to design a model that is aware of scene information**. For instance, a car is considered normal on a road but abnormal on a pedestrian walkway. If a model is trained solely on normal data without distinguishing between different scenes, it may incorrectly classify ```scene-dependent abnormal objects``` (e.g., a car on a pedestrian) as normal during testing. To address this issue, I propose a simple yet effective **scene-aware VAD method**.

## Overview
To perform anomaly detection at the video segment level, segments from various scenes are utilized during training.  
**To enable fast training with a lightweight model, I operate at the feature level instead of using raw frames.**  
For this purpose, I employ the **CLIP Image Encoder** based on ```ViT-L/14```, which extracts fine-grained semantic information from individual frames.
Subsequently, I perform reconstruction-based training using a **Transformer-based AutoEncoder** to model temporal dynamics.

<img src="https://github.com/user-attachments/assets/2acf5983-ea46-4615-b451-77e641a9975f" width="750"/>

## Training method
In this study, I adopt a **reconstruction-based approach**, one of the commonly used **One-Class Classification (OCC)** methods in VAD. This approach enables the computation of anomaly scores in a straightforward manner by measuring the **reconstruction error** between the input and the output.
To equip the model with the ability to distinguish between different scenes, I additionally incorporate **contrastive learning**.
The objective is to learn **scene-specific normal manifolds** such that, during testing, features from **abnormal frames-particularly those containing scene-dependent abnormal objects—** deviate from the learned manifolds.
This design makes it more difficult for the decoder to reconstruct those abnormal inputs, thereby resulting in higher reconstruction errors.

<img src="https://github.com/user-attachments/assets/f9910f79-21d5-49d0-addf-c59ca3dca98f" width="750"/>

## Results
<details>
<summary><b>Quantitative comparison</b></summary>
  
## Quantitative  comparison
To evaluate whether the model can effectively handle anomalies that vary depending on the scene, I utilize ```ShanghaiTech-SD```, a **scene-dependent dataset**. Details of the dataset can be found in the following paper [[Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_A_New_Comprehensive_Benchmark_for_Semi-Supervised_Video_Anomaly_Detection_and_CVPR_2023_paper.pdf)].  
Experimental results show that our proposed model, which learns to distinguish between different scenes, achieves a **17.7% higher AUC**. This improvement demonstrates that **scene-specific normal manifolds are appropriately constructed**, allowing the model to effectively detect **abnormal frames that violate scene semantics**—such as a bicycle on a pedestrian walkway.


|     Method                  |Training  |AUC    |
|:------------------------:|:-----------:|:-----------:|
| Scene-agnostic  |reconstruction        |61.3%        |
| **Scene-aware**   |**reconstruction + contrastive**        |**79.0%**        |
</details>

<details>
<summary><b>Qualitative  comparison</b></summary>
  
## Qualitative  comparison
```Scene1``` is a **general-purpose road where bicycles and motorcycles are allowed**, while ```Scene2``` and ```Scene3``` are **pedestrian-only areas**.
A **scene-agnostic model**, which does not take scene context into account, tends to assign **low anomaly scores** to scene-dependent anomalies such as a ```bicycle appearing in Scene2```.
In contrast, the proposed **scene-aware model** gives **higher anomaly scores** in such cases, effectively detecting situations that don't fit the scene.

- **Scene 1 (normal: walking, standing, sitting, bicycle, motorcycle)**
  
| Aware     | Status                                                                | frame (160th) |Anomaly Score |
|-----------|------------------------------------------------------------------------|-------|-------|
| ❌ | **bicycle: normal**  | <img src="https://github.com/user-attachments/assets/bf046ade-09e0-4320-b53e-7946200526cf" width="400"/>  |<img src="https://github.com/user-attachments/assets/7dfe9dd6-0bec-479a-b2cb-5219e347d04d" width="600"/>|
| ✅ | **bicycle: normal**| <img src="https://github.com/user-attachments/assets/bf046ade-09e0-4320-b53e-7946200526cf" width="400"/>  |<img src="https://github.com/user-attachments/assets/d6dddf75-b130-40ef-bb15-e7f07fa90ebf" width="600"/>|


- **Scene 2 (normal: walking, standing, sitting)**
  
| Aware     | Status                                                                | frame (130th) |Anomaly Score |
|-----------|------------------------------------------------------------------------|-------|-------|
| ❌ | **bicycle: abnormal**  | <img src="https://github.com/user-attachments/assets/69abefff-0712-4e10-848c-8266e3a38348" width="400"/>  |<img src="https://github.com/user-attachments/assets/b8ba7763-2cff-466f-bceb-5fbd9a652e8d" width="600"/>|
| ✅ | **bicycle: abnormal**| <img src="https://github.com/user-attachments/assets/69abefff-0712-4e10-848c-8266e3a38348" width="400"/>  |<img src="https://github.com/user-attachments/assets/036a281c-c6e4-4691-b792-cef480aa2504" width="600"/>|


- **Scene 3 (normal: walking, standing, sitting)**
  
| Aware     | Status                                                                | frame (160th) |Anomaly Score |
|-----------|------------------------------------------------------------------------|-------|-------|
| ❌ | **motorcycle: abnormal**  | <img src="https://github.com/user-attachments/assets/794894f8-a80d-49cb-b474-f3c22215e0ee" width="400"/>  |<img src="https://github.com/user-attachments/assets/511e2229-45e0-4788-9f4e-c61fe146bb03" width="600"/>|
| ✅ | **motorcycle: abnormal**| <img src="https://github.com/user-attachments/assets/794894f8-a80d-49cb-b474-f3c22215e0ee" width="400"/>  |<img src="https://github.com/user-attachments/assets/058b659b-1e3d-45e7-aacb-e633adfb84a5" width="600"/>|
</details>

<details>
<summary><b>Visualization</b></summary>

## Visualization
To effectively visualize the **normal manifolds** of the **scene-agnostic** and **scene-aware** methods, I trained both approaches using a ```Variational AutoEncoder (VAE)```.
While both methods produce features that follow a **normal distribution**, the proposed scene-aware method additionally shows clear **separation by scene**.  
Incidentally, the proposed method also achieved better performance even with the VAE. However, for more stable training, I used an AutoEncoder, which resulted in even higher performance.

|     Scene-agnostic manifold                |Scene-aware manifold  |
|:------------------------:|:-----------:|
| <img src="https://github.com/user-attachments/assets/75a1fef8-4683-4d2a-bea6-1c01209e814d" width="450"/>| <img src="https://github.com/user-attachments/assets/94c34176-e198-4efb-9c4a-6ac576f4baf2" width="450"/>|
</details>

<details>
<summary><b>VAD Performance Benchmarking</b></summary>
  
## VAD Performance Benchmarking
Compared to existing state-of-the-art VAD methods, my approach achieves **superior performance** in most cases.  
This is attributed to the incorporation of a **scene-aware mechanism**, which allows the model to learn **scene-specific normal patterns** (e.g., a bicycle is normal only in scene 1).         
Although it shows slightly lower performance than Cao et al., it still delivers **competitive results without relying on object detection**.  
The combination of a ```powerful feature extractor``` and a ```lightweight AutoEncoder``` enables **efficient training and inference**, making the method well-suited for real-time applications or deployment in resource-constrained environments.  

| Method                 | Feature | Scene-aware | AUC  |
|------------------------|---------|-------------|------|
| MemAE (ICCV'19)        | Image   | ❌          | 67.4 |
| MNAD (CVPR'20)         | Image   | ❌          | 68.2 |
| OG-Net (CVPR'20)       | Image   | ❌          | 69.6 |
| AMMC-Net (AAAI'21)     | Image   | ❌          | 64.9 |
| HF²-VAD (ICCV'21)      | Object  | ❌          | 70.8 |
| MPN (CVPR'21)          | Image   | ❌          | 76.9 |
| Cao et al. (CVPR'23)   | Object  | ✅          | 82.7 |
| **Proposed**               | **Image**  | ✅          | **79.0** |

</details>

## Execution
<details>
<summary><b>Environment</b></summary>

## Environment
PyTorch >= 1.13.1  
Python >= 3.8  
sklearn  
opencv  
torchvision  
wandb  
h5py  
fastprogress  
git+https://github.com/openai/CLIP.git  
Other common packages.
</details>

<details>
<summary><b>Download</b></summary>
  
## Download
Please move the **dataset (shanghai-sd)** into the ```data_root``` directory specified in ```config.py```.  
The **features** and **weights** directories should be moved to the ```working directory```.
|     Dataset    |  CLIP features    |   Weights    |  Train Log    | 
|:------------------------:|:------------------------:|:------------------------:|:------------------------:|
|[Google Drive](https://drive.google.com/file/d/1H5i-rBBlPZpk7Ix3TOR66rRY_2BtTMuD/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1R-wggWDWKF4usG6SoToUEretUTETba8r/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1fW-OrFLqaZHIa2lb6d0QacSV9KM0uUnA/view?usp=sharing)|[Google Colab](https://colab.research.google.com/drive/1bLsA-coc6WOXPZsPm3wUyIEIl54DqzJp?usp=sharing)|
</details>

<details>
<summary><b>Command</b></summary>
  
## Command
For training, the ```segment length``` was set to **16**, and **30 video segments were used per scene** for contrastive learning.  
The total number of segments used in contrastive learning is calculated as ```scenes × segments```.  
For example, if there are **4 scenes and 30 segments**, a total of **120 segments** are used.  
For each anchor, **29 segments** are used as **positives** and **90 segments** are used as **negatives**.
- feature extraction
```bash
python featuring.py --dataset=shanghai-sd --save_mode=training  # train data -> train features
python featuring.py --dataset=shanghai-sd --save_mode=testing   # test data -> test features
```
- training
```bash
python train.py --dataset=shanghai-sd --training_mode=0  # reconstruction 
python train.py --dataset=shanghai-sd --training_mode=1  # reconstruction + contrastive
python train.py --dataset=shanghai-sd --training_mode=2  # reconstruction + contrastive + classifiaction 
```
- testing
```bash
python eval.py --dataset=shanghai-sd --trained_model={weight_file_name} # micro auc calculation 
python eval.py --dataset=shanghai-sd --trained_model={weight_file_name} --visualization=True  # + (anomaly score, t-sne) visualization
```
</details>

## Acknowledgement
I conducted this study with reference to ```SupCon(NIPS'2020)``` and ```HSC(CVPR'2023)```, and was also inspired by ```Cao et al.(CVPR'2023)```. I sincerely appreciate the contributions of the authors of these three papers.

- **SupCon (NIPS'2020)** [[paper](https://arxiv.org/pdf/2004.11362)]
- **HSC (CVPR'2023)** [[paper](https://arxiv.org/pdf/2303.13051)]
- **Cao et al. (CVPR'2023)** [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_A_New_Comprehensive_Benchmark_for_Semi-Supervised_Video_Anomaly_Detection_and_CVPR_2023_paper.pdf)]
