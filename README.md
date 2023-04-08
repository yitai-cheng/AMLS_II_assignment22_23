# AMLS_II_assignment22_23

A malignant tumor in the brain is a life-threatening condition. 
The presence of Methylguanine methyltransferase (MGMT) promoter methylation has been shown to be a favorable prognostic factor and a strong predictor of responsiveness to chemotherapy. 
In this project, machine learning based models are proposed to deal with the detection of MGMT promoter methylation instead of invasive surgeries taking brain tissues out of patients’ body. 
Taking advantage of multiple modalities of MRI images, both Single Modality Model (SMM) and Multi-Modality Model (MMM) are proposed. 
The main building block of SMM is the feature extractor and that of MMM is feature extractor with the attention layer.


The program will be started by main.py. It will first fetch the dataset from kaggle. Then exploratory data analysis is conducted.
Then we preprocess the dataset. The machine learning models are defined in models.py. We train and validate our model in train.py. Finally we test our model in test.py.
Hyper-parameters are set in constants.py.

This project uses the following packages:
- matplotlib==3.6.1
- numpy==1.22.3
- pandas==1.5.0
- pydicom==2.3.1
- requests==2.26.0
- scikit_learn==1.2.2
- seaborn==0.12.2
- timm==0.6.13
- torch==2.0.0
- torchvision==0.15.1