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

### Model Progression

| Task                             | Runs          | Test Accuracy  |               |
| -------------------------------- | ------------- | -------------- | ------------- |
| Imagined left vs right fist      | R04, R08, R12 | 64.5%          |               |
| Executed left vs right fist      | R03, R07, R11 | 69.2%          |               |
| Executed both fists vs both feet | R05, R09, R13 | 68.3%          |               |
| Imagined both fists vs both feet | R06, R10, R14 | 56.4%          |               |

### Architecture Iteration 
| Changes Tested                   | Result        | Interpretation |               |
|--------------------------------- | ------------- | -------------- | ------------- |
| Kernel size (1,10) → (64,10)     | Initially observed improvement from 52.59% accuracy to 60.42% |  This change offered the largest single improvement as shifting the kernel size expanded the model's view from 1 channel to all 64 simultaneously per time window | |
|


---

## Future Work

* Develop deeper CNN architectures.
* Add training accuracy and loss visualizations.
* Perform hyperparameter tuning.
* Investigate additional EEG feature extraction techniques.
* Explore alternative architectures for improved motor imagery discrimination.

            

                    
            
