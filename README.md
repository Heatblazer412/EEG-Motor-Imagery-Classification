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
Goal of project: Implement a neural network that can accurately examine an EEG motor imagery graph and correctly relate segmented epochs to their associated event labels
Documentation of progress: 
    - Began development with an initial MLP model:
            - This model yielded a maximum accuracy of 42%, which indicated a learnable signal as random chance yields 33%
            - Despite this, the model suffered from severe over-fitting, as training accuracy would quickly reach 100% despite the weak testing accuracy
    - Continued development with a more sophisticated aaproach, using a Convolutional Neural Network (CNN):
            - One convolutional layer immediately yielded a superior maximum testing accuracy of ~61%
            - Two convolutional layers yielded an even greater accuracy of 64.5%
            - This double layer CNN achieves this result with the following configuration:
                    Architecture:
                        - Conv2D(1 -> 32), kernel=(64,10)
                        - Batch Normalization
                        - Max Pooling
                        - Conv2D(32 -> 64), kernel=(1,10)
                        - Batch Normalization
                        - Max Pooling
                        - Dropout (0.3)
                        - Fully Connected Output Layer
                    Training Configuration:
                        - Optimizer: Adam
                        - Learning Rate: 0.0001
                        - Batch Size: 35
                        - Passes: 50
            - A diagnostic of this model reveals:
                    Best Test Accuracy: 64.5%
                    Classification Report:
                        -Class 0:
                            -Precision: 0.67
                            -Recall: 0.77
                            -F1: 0.72
                        -Class 1:
                            -Precision: 0.60
                            -Recall: 0.47
                            -F1: 0.53
                        -Class 2:
                            -Precision: 0.54
                            -Recall: 0.50
                            -F1: 0.52
                    Interpretation:
                        - The CNN substantially outperformed the initial MLP baseline,
                        which barely achiveved a ~40% test accuracy due to severe overfitting.
                        - The model identifies resting-state EEG (Class 0) more reliably
                        than motor imagery classes (Classes 1 and 2).
                        - Confusion matrix analysis indicates that the dominant source
                        of error is the misclassification of motor imagery epochs as
                        resting-state epochs.
                        - This suggests the network has learned to distinguish rest from
                        motor imagery, but still struggles to consistently identify
                        which motor imagery task is being performed.
    - To be done:
            - Deeper CNN architectures 
            - Accuracy and loss visualizations
            - Parameter tuning
            - Additional EEG feature extraction

            

                    -
            
