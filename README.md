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

## Goal of the Project

Implement a neural network capable of analyzing EEG motor imagery recordings and correctly associating segmented EEG epochs with their corresponding event labels.

---

## Documentation of Progress

### Initial Baseline: Multi-Layer Perceptron (MLP)

* Began development with a simple MLP architecture.
* Achieved a maximum testing accuracy of approximately **42%**.
* Since random chance performance for a three-class problem is roughly **33%**, this indicated the presence of a learnable signal within the EEG data.
* However, the model suffered from severe overfitting:

  * Training accuracy rapidly approached **100%**.
  * Testing accuracy remained relatively low.

### Convolutional Neural Network (CNN)

To better preserve the spatial and temporal structure of EEG data, development continued with a Convolutional Neural Network (CNN).

#### Performance Improvements

* A single convolutional layer increased maximum testing accuracy to approximately **61%**.
* Expanding the architecture to two convolutional layers further improved performance to approximately **64.5%**.

#### Architecture

* Conv2D(1 → 32), kernel=(64,10)
* Batch Normalization
* Max Pooling
* Conv2D(32 → 64), kernel=(1,10)
* Batch Normalization
* Max Pooling
* Dropout (0.3)
* Fully Connected Output Layer

#### Training Configuration

* Optimizer: Adam
* Learning Rate: 0.0001
* Batch Size: 35
* Passes: 50

---

## Model Diagnostics

### Best Test Accuracy

**64.5%**

### Classification Report

| Class | Precision | Recall | F1-Score |
| ----- | --------- | ------ | -------- |
| 0     | 0.67      | 0.77   | 0.72     |
| 1     | 0.60      | 0.47   | 0.53     |
| 2     | 0.54      | 0.50   | 0.52     |

### Interpretation

* The CNN substantially outperformed the initial MLP baseline.
* The model identifies resting-state EEG activity (**Class 0**) more reliably than motor imagery tasks (**Classes 1 and 2**).
* Confusion matrix analysis indicates that the dominant source of error is the misclassification of motor imagery epochs as resting-state epochs.
* These results suggest that the network has successfully learned to distinguish resting-state activity from motor imagery activity.
* However, the model still struggles to consistently differentiate between the two motor imagery tasks.

---

## Future Work

* Develop deeper CNN architectures.
* Add training accuracy and loss visualizations.
* Perform hyperparameter tuning.
* Investigate additional EEG feature extraction techniques.
* Explore alternative architectures for improved motor imagery discrimination.

            

                    -
            
