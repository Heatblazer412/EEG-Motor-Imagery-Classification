# EEG-Motor-Imagery-Classification

| Run           | Task                            |               |
| ------------- | ------------------------------- | ------------- |
| R01           | Baseline, eyes open             |               |
| R02           | Baseline, eyes closed           |               |
| R03, R07, R11 | Execute left vs right fist      |               |
| R04, R08, R12 | Imagine left vs right fist      |               |
| R05, R09, R13 | Execute both fists vs both feet |               |
| R06, R10, R14 | Imagine both fists vs both feet | ([Kaggle][1]) |

[1]: https://www.kaggle.com/datasets/gamalasran/physionet-eeg-motor-movement-imagery/data?utm_source=chatgpt.com "PhysioNet EEG Motor Movement / Imagery"

Current status:
#Training progress:
    #With a dropout rate of .1 and a single convolutional layer, this CNN achieves a peak
    #testing accuracy of roughly 61%. To achieve a higher testing accuracy we need 
    #to build more convolutional layers as to better identify relevant information for the
    #model to work with.
