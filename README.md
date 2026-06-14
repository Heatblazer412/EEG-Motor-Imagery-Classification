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
Finished Preprocessing.py implementation to serve the functionality of the now finished baseline_model.py. 
Documentation of baseline model effectiveness: 
    Model immediately prioritizes memorization over understanding after just a couple passes 
    Random chance baseline for 3 classes: 33.33%
    At 1 pass a typical result will indicate a training accuracy of 54.84% & a test accuracy of 43.90%
    At roughly 3-4 passes, we yield our optimal model resulting in, Training accuracy: 64.71%, Test accuracy: 44.97%
    From there on, training accuracy goes up and and testing accuracy goes down
    This overall performance indicates that a more complicated model is necessary to achieve better results, 
    though a testing accuracy of ~45% is not insignificant, demonstrating real success in interpreting the data.
