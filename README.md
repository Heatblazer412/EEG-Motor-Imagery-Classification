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

* Random chance baseline (3 classes): 33.33%
* Pass 1: Training accuracy 54.84%, Test accuracy 43.90%
* Optimal (~pass 3): Training accuracy 64.71%, Test accuracy 44.97%
* Beyond pass 3: Training accuracy continued climbing; test accuracy declined, as the model began overfitting

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
| Stage | Architecture                                  | Best Test Accuracy | Notes                                |
| ----- | ---------------------------------------------- | ------------------- | ------------------------------------- |
| 1     | Baseline MLP (1 hidden layer, manual weights)  | ~45%                | Single subject, 90 epochs             |
| 2     | CNN (1 conv layer)                             | ~61%                | Full dataset (104+ subjects)          |
| 3     | CNN (2 conv layers + dropout + tuned kernel)   | ~65-69%             | Task-dependent, see below             |
| 4     | EEGNet (planned)                               | TBD                 | Purpose-built depthwise/separable convs |

### Architecture Iteration 

| Changes Tested                   | Result        | Interpretation |               |
|--------------------------------- | ------------- | -------------- | ------------- |
| Kernel size (1,10) → (64,10)     | Initially observed improvement from 52.59% accuracy to 60.42% |  This change offered the largest single improvement as shifting the kernel size expanded the model's view from 1 channel to all 64 simultaneously per time window | |
| Dropout (p=0.5 → 0.25 → 0.1)     | Stabilized noisy test accuracy, peak ~61% | Reduced the vairance in test accuracy results, despite maintaining the same peak. Also regulated the divergence in training accuracy and testing accuracy as the model learned |
| Doubling filter count (32→64, 64→128) | ~64.58% (no improvement) | Ruled out width as the bottleneck. Wider network reached similar accuracy but required more passes and had a worse train/test gap (94% train vs 65% test), indicating pure overfitting with no real gain. | |
| Weight decay (L2, 1e-4)          | ~63.82% (no improvement) | Ruled out additional regularization as the crux | |
| 3rd conv layer                   | ~57.6% (worse)| Ruled out depth (at this scale). Results indicated increased capacity to memorize (94.89% train) without improving generalization (57.62% test). With ~9,000 total epochs, a 3-layer CNN likely needs more data to justify the added complexity. | |
| Removing 8-30 Hz bandpass filter | 64.33% (no meaningful change) |The model learns despite a filtered range, indicating that filtering neither supresses nor highlights any useful information for the model. | |
| Class-weighted loss (for imbalance) | 61.89% (worse) | Class imbalance (45/23/22 split in early single-subject testing) wasn't severe enough to justify the precision/recall trade-off. | |

#### Conclusion on Architecture Tuning 
* Manipulating and finetuning architecture will likely not reveal any significant changes within the ~64-69% cwiling.
* Changes in effectivity are likely only possible through significant change in the pipeline or changes in feature representation of the data, neither of which are trivial. 

### Task-type comparison

* Using the same final architecture, performance was compared across four task conditions to see whether the ceiling was universal or task-dependent:

| Task                             | Runs          | Test Accuracy  | Test Samples  |
| -------------------------------- | ------------- | -------------- | ------------- |
| Imagined left vs right fist      | R04, R08, R12 | 64.5%          | 1968          |
| Executed left vs right fist      | R03, R07, R11 | 69.2%          | 1906          |
| Executed both fists vs both feet | R05, R09, R13 | 68.3%          | 1899          |
| Imagined both fists vs both feet | R06, R10, R14 | 56.4%          | 1269          |

#### Findings:
* Execution consistently outperforms imagery for both task types (69.15% vs 64.48% for L/R fist; 68.30% vs 56.42% for hands/feet), which is consistent with literature, since actual movement produces stronger, less noisy motor signal than imagined movement.
* Executed L/R fist slightly outperforms executed hands/feet, and imagine L/R fist significantly outperforms imagined hands/feet, despite hands/feet being the more spatially separated distinction. This goes counter to my hypothesis that hand/feet signals would be more distinct as they're functionality is spacially further apart and would consequently produce more differentiable signals.
* Possible explanations:
      * Elecetrode sensors could be more likely to detect hand motor movements over foot motor movements, as motor regions for foot movement are deeper embedded within the brain consequently may produce noisier quality
      * This contradiction in imagined imagery specifically could be explained by the difference in test samples (1269 vs. ~1900)
---

## Future Work

* Develop deeper CNN architectures.
* Add training accuracy and loss visualizations.
* Perform hyperparameter tuning.
* Investigate additional EEG feature extraction techniques.
* Explore alternative architectures for improved motor imagery discrimination.

            

                    
            
