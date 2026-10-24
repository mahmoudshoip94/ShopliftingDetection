# 🛒 Shoplifting Detection
### CCTV Video Classification — From Scratch vs. Pretrained VideoMAE

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![OpenCV](https://img.shields.io/badge/OpenCV-Video%20Processing-green)
![Google Colab](https://img.shields.io/badge/Google%20Colab-GPU-F9AB00)

A deep learning project for **binary shoplifting activity classification in CCTV surveillance videos**.

The project investigates two approaches to temporal video understanding:

1. **Training from scratch:** A custom CNN + Bidirectional LSTM (BiLSTM).
2. **Transfer learning:** A pretrained VideoMAE Small video Transformer fine-tuned for shoplifting classification.

Both approaches predict one of two classes:

- **Normal Shopping (Class 0)**
- **Shoplifting (Class 1)**

The project includes Exploratory Data Analysis (EDA), dataset cleaning, frame sampling, video augmentation, model training, evaluation, and performance comparison.

> The models perform video-level classification. They do not identify individuals, localize theft events, or establish that theft has occurred.

---

## 🏆 Results Overview

Three completed experiments are included in the repository.

| Metric | CNN + BiLSTM (Scratch) | VideoMAE (Validation) | VideoMAE (No Validation) |
|---|---:|---:|---:|
| Test Accuracy | **95.79%** | 89.47% | 91.58% |
| Macro Precision | **96.08%** | 89.78% | 91.91% |
| Macro Recall | **95.83%** | 89.43% | 91.53% |
| Macro F1-Score | **95.79%** | 89.44% | 91.56% |
| Shoplifting Recall | 91.67% | 93.75% | **95.83%** |
| Correct Predictions | **91** | 85 | 87 |
| Incorrect Predictions | **4** | 10 | 8 |
| Test Videos | 95 | 95 | 95 |

**Key observations:**

- The custom CNN-BiLSTM achieved the highest overall Accuracy and Macro F1.
- VideoMAE with all training videos achieved higher scores than the earlier VideoMAE experiment.
- VideoMAE without validation achieved the highest Shoplifting Recall, detecting 46 out of 48 shoplifting videos.
- The pretrained experiments did not surpass the scratch model's Macro F1.

All scores come from the same previously inspected internal test set. These results are useful for architectural comparison but are not independent estimates of performance on unseen stores.

---

# 📊 1. Exploratory Data Analysis

The dataset was explored before model development to understand its characteristics and identify potential data-quality issues.

Notebook:

**[Shoplifting_EDA.ipynb](FromScratch/Shoplifting_EDA.ipynb)**

## 1.1 Original Dataset

The original dataset contains 855 labeled CCTV videos.

| Class | Number of Videos | Percentage |
|---|---:|---:|
| Normal Shopping | 531 | 62.11% |
| Shoplifting | 324 | 37.89% |
| **Total** | **855** | **100%** |

![Class Distribution](assets/eda/class_distribution.png)

## 1.2 Video Metadata

The EDA successfully retrieved metadata from 854 of the 855 original videos.

| Property | Observation |
|---|---|
| Readable Videos | 854 / 855 |
| Median Duration | 13 seconds |
| Median FPS | 25 |
| Predominant Resolution | 704 × 576 |
| Videos Shorter Than 5 Seconds | 3 |
| Videos Longer Than 60 Seconds | 2 |

### Duration Statistics

| Statistic | Normal | Shoplifting |
|---|---:|---:|
| Readable Videos | 530 | 324 |
| Minimum Duration | 3.00 s | 4.00 s |
| Median Duration | 14.00 s | 12.00 s |
| Mean Duration | 13.93 s | 12.22 s |
| Maximum Duration | 74.00 s | 23.03 s |

These measurements describe the original dataset before duplicate removal.

![Video Duration Distribution](assets/eda/duration_distribution.png)

### FPS and Resolution

Most valid videos have approximately 25 FPS, and all 854 videos with readable metadata have a resolution of 704 × 576.

![FPS and Resolution](assets/eda/fps_resolution.png)

The consistent camera environment may make classification easier on the internal dataset but may also encourage the models to learn scene-specific visual patterns.

## 1.3 Duplicate Detection and Cleaning

SHA-256 hashes were used to identify byte-identical video files.

| Stage | Videos |
|---|---:|
| Original Dataset | 855 |
| Extra Exact Duplicate Copies | 218 |
| Unique Videos | 637 |
| Additional Known Overlaps Excluded | 5 |
| **Final Cleaned Dataset** | **632** |

The exact duplicate analysis identified 218 hash groups containing two copies each and 419 hash groups containing one file.

![Duplicate Summary](assets/eda/duplicate_summary.png)

Exact hashing prevents byte-identical duplicates from being assigned to different subsets. However, it does not identify every possible re-encoded or overlapping recording.

## 1.4 Train/Test Split

The final cleaned dataset uses the following split:

| Split | Normal | Shoplifting | Total |
|---|---:|---:|---:|
| Training | 263 | 274 | 537 |
| Testing | 47 | 48 | 95 |
| **Total** | **310** | **322** | **632** |

The EDA verified:

- 632 matching video hashes in the saved split manifest.
- Zero exact SHA-256 collisions between training and testing.
- Consistent class labels across the saved manifest.

![Train/Test Distribution](assets/eda/split_distribution.png)

The models use the same 537/95 video identities for internal comparison. The validation-based VideoMAE experiment additionally divides the original training portion into training and validation subsets.

## 1.5 Temporal Sampling

Both approaches use a 1 FPS sampling strategy to reduce the computational cost of processing the original 25 FPS recordings.

The scratch training notebook extracted 8,289 sampled frames across 632 cleaned videos.

![Sampled Frame Distribution](assets/eda/sampled_frames_distribution.png)

The two architectures use these sampled frames differently, as described below.

---

# 🧠 2. From-Scratch CNN + BiLSTM

Notebook:

**[Shoplifting_Independent_V6.ipynb](FromScratch/Shoplifting_Independent_V6.ipynb)**

## 2.1 Architecture

The from-scratch model combines CNN spatial feature extraction with BiLSTM temporal sequence modeling.

```text
              Input CCTV Video
                     |
                     v
               Sampling at 1 FPS
                     |
                     v
             Resize to 112 × 112
                     |
                     v
              CNN Feature Extractor
                     |
                     v
             Feature Projection
                     |
                     v
             Bidirectional LSTM
                     |
                     v
          Masked Mean + Max Pooling
                     |
                     v
              Classification Head
                     |
                     v
             Normal / Shoplifting
```

### CNN Backbone

Four convolutional blocks extract spatial features from individual frames.

| Block | Output Channels |
|---|---:|
| CNN Block 1 | 24 |
| CNN Block 2 | 48 |
| CNN Block 3 | 96 |
| CNN Block 4 | 160 |

The CNN uses convolutional layers, Group Normalization, SiLU activation, and adaptive average pooling.

### BiLSTM

The BiLSTM processes the frame feature sequence in both temporal directions.

| Parameter | Value |
|---|---:|
| Input Features | 160 |
| Hidden Size | 128 |
| Bidirectional | Yes |
| Output Features | 256 per timestep |

Masked mean pooling and masked max pooling produce a 512-dimensional video representation.

The final classifier produces two logits.

**Total parameters: 998,090.**

## 2.2 Preprocessing and Augmentation

| Setting | Configuration |
|---|---|
| Input | RGB Video Frames |
| Sampling | 1 FPS |
| Frame Resolution | 112 × 112 |
| Sequence Length | Variable |
| Normalization | [-1, 1] |
| Horizontal Flip | 50% |
| Brightness | ±15% |
| Contrast | ±15% |
| Saturation | ±15% |

Augmentation parameters are consistent across all frames in the same video.

No augmentation is applied during testing.

## 2.3 Training Configuration

| Hyperparameter | Value |
|---|---|
| Weight Initialization | Random |
| Epochs | 30 |
| Batch Size | 8 |
| Optimizer | AdamW |
| Learning Rate, Epochs 1–5 | 1e-3 |
| Learning Rate, Epochs 6–30 | 5e-4 |
| Weight Decay | 1e-4 |
| Loss Function | Cross-Entropy |
| Label Smoothing | 0.02 |
| Gradient Clipping | 1.0 |
| Mixed Precision | CUDA AMP |

The final checkpoint was saved after 30 predetermined epochs.

## 2.4 Test Results

**Accuracy: 95.79%**

**Macro F1: 95.79%**

### Confusion Matrix

| Actual / Predicted | Normal | Shoplifting |
|---|---:|---:|
| Normal | 47 | 0 |
| Shoplifting | 4 | 44 |

The model correctly classified 91 of 95 videos.

---

# 🤗 3. Pretrained VideoMAE Small

The transfer-learning experiments use a pretrained video Transformer available on Hugging Face:

**[MCG-NJU/videomae-small-finetuned-kinetics](https://huggingface.co/MCG-NJU/videomae-small-finetuned-kinetics)**

The checkpoint was originally trained for activity recognition with 400 output classes.

Its classification head was replaced with a new two-class head:

```text
Normal     → Class 0
Shoplifting → Class 1
```

The pretrained backbone is reused, while selected model parameters are fine-tuned for the target dataset.

## 3.1 VideoMAE Architecture

```text
               Input CCTV Video
                      |
                      v
                Sampling at 1 FPS
                      |
                      v
          Select / Repeat to 16 Frames
                      |
                      v
             Resize Full Frames to
                    224 × 224
                      |
                      v
             VideoMAE Transformer
                      |
                      v
             Video Representation
                      |
                      v
           Binary Classification Head
                      |
                      v
              Normal / Shoplifting
```

### Model Configuration

| Parameter | Value |
|---|---|
| Architecture | VideoMAE Small |
| Pretrained Source | Kinetics-400 Checkpoint |
| Input Frames | 16 |
| Input Resolution | 224 × 224 |
| Output Classes | 2 |
| Total Parameters | 21,880,706 |
| Classifier Input Features | 384 |
| Classifier Output Features | 2 |

The pretrained checkpoint was loaded successfully, and a forward-pass sanity test confirmed an output shape of `(1, 2)`.

## 3.2 VideoMAE Input Processing

The notebook samples video frames at 1 FPS, then uniformly selects 16 positions across the sampled sequence.

For videos containing fewer than 16 sampled frames, some frames are repeated.

The notebook's input compatibility check found:

- 462 videos with fewer than 16 sampled frames.
- 170 videos with 16 or more sampled frames.

The final input tensor shape per video is:

`(16, 3, 224, 224)`

### Full-Frame Resize — No Crop

VideoMAE uses direct resizing to 224 × 224.

**No center crop, random crop, or resized crop is applied.**

This preserves the full visible frame, including hands, shelves, and objects near image boundaries, although resizing the original aspect ratio to a square changes image proportions.

### Normalization

The input frames use the pretrained model's image normalization statistics:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### Augmentation

| Technique | Configuration |
|---|---|
| Horizontal Flip | 50% probability |
| Brightness | ±15% |
| Contrast | ±15% |
| Saturation | ±15% |
| Crop | None |

The same randomly selected augmentation factors are used for all frames within each training video.

---

# 🔬 4. VideoMAE Fine-Tuning Experiments

Two fine-tuning configurations are included.

## 4.1 Experiment A — With Validation

Notebook:

**[VideoMAE_Small_Finetuning.ipynb](pretrained/VideoMAE_Small_Finetuning.ipynb)**

The original 537 training videos were divided into training and validation subsets.

| Dataset Portion | Videos |
|---|---:|
| Training | 456 |
| Validation | 81 |
| Test | 95 |

### Training Stages

**Stage A — Classification Head Training**

- Train the new classification head.
- Keep the pretrained backbone frozen.
- Train for up to 2 epochs.

**Stage B — Partial Backbone Fine-Tuning**

- Fine-tune the final two Transformer blocks.
- Train the classification head.
- Train for up to 6 epochs.

The best checkpoint was selected according to validation Macro F1, with validation loss used as a tie-breaker.

### Results

| Metric | Value |
|---|---:|
| Best Validation Macro F1 | 90.11% |
| Test Accuracy | 89.47% |
| Test Macro F1 | 89.44% |
| Shoplifting Recall | 93.75% |

### Test Confusion Matrix

| Actual / Predicted | Normal | Shoplifting |
|---|---:|---:|
| Normal | 40 | 7 |
| Shoplifting | 3 | 45 |

The model correctly classified 85 of 95 test videos.

---

## 4.2 Experiment B — No Validation

Notebook:

**[VideoMAE_Small_No_Validation.ipynb](pretrained/VideoMAE_Small_No_Validation.ipynb)**

This experiment uses all 537 original training videos.

| Dataset Portion | Videos |
|---|---:|
| Training | 537 |
| Validation | 0 |
| Test | 95 |

### Training Strategy

The experiment uses a fixed training schedule, without validation-based checkpoint selection or early stopping.

| Stage | Epochs | Trainable Components |
|---|---:|---|
| Stage A | 2 | Classification Head |
| Stage B | 6 | Final Two Transformer Blocks + Head |

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Batch Size | 2 on CUDA |
| Gradient Accumulation | 4 |
| Head Training LR | 3e-4 |
| Fine-tuned Backbone LR | 1e-5 |
| Fine-tuned Head LR | 1e-4 |
| Weight Decay | 0.01 |
| Gradient Clipping | 1.0 |
| Mixed Precision | CUDA AMP |
| Validation | Disabled |
| Checkpoint Selection | Final Predetermined Epoch |

### Final Results

| Metric | Value |
|---|---:|
| **Test Accuracy** | **91.58%** |
| Macro Precision | 91.91% |
| Macro Recall | 91.53% |
| **Macro F1** | **91.56%** |
| Shoplifting Recall | 95.83% |
| Correct Predictions | 87 |
| Incorrect Predictions | 8 |

### Test Confusion Matrix

| Actual / Predicted | Normal | Shoplifting |
|---|---:|---:|
| Normal | 41 | 6 |
| Shoplifting | 2 | 46 |

The model detected 46 of the 48 shoplifting videos but incorrectly classified 6 normal videos as shoplifting.

Using all 537 training videos produced a higher recorded test score than the validation-based run. However, removing validation also removes the ability to monitor generalization during training.

These results do not establish that excluding validation is generally beneficial.

---

# ⚖️ 5. From-Scratch vs. Pretrained Comparison

The following table compares the scratch baseline with the completed pretrained experiment using all 537 training videos.

| Property | CNN + BiLSTM | VideoMAE Small |
|---|---|---|
| Initialization | Random | Pretrained |
| Architecture | CNN + BiLSTM | Video Transformer |
| Parameters | 998,090 | 21,880,706 |
| Frames per Video | Variable | 16 |
| Frame Resolution | 112 × 112 | 224 × 224 |
| Temporal Sampling | 1 FPS | 1 FPS + Uniform Selection |
| Training Videos | 537 | 537 |
| Validation | No | No |
| Test Videos | 95 | 95 |
| **Test Accuracy** | **95.79%** | 91.58% |
| **Macro F1** | **95.79%** | 91.56% |
| Shoplifting Recall | 91.67% | **95.83%** |

## 5.1 Accuracy and F1

The scratch CNN-BiLSTM achieved a Macro F1 approximately **4.23 percentage points higher** than the VideoMAE no-validation experiment.

However, VideoMAE achieved higher Shoplifting Recall.

The results illustrate that higher recall for one class does not necessarily produce better overall classification performance.

## 5.2 Model Size

The scratch model contains approximately 1 million parameters, while VideoMAE Small contains approximately 21.9 million.

VideoMAE therefore has around 21.9 times as many parameters.

Its original Hugging Face model download is approximately 88 MB. Exact serialized checkpoint sizes for both locally trained models should be measured separately.

## 5.3 Inference Speed

Both architectures have different computational requirements.

The scratch model processes a variable-length sequence of 112 × 112 frames, while VideoMAE processes 16 frames at 224 × 224.

VideoMAE is substantially larger in parameter count, but actual inference latency has not been measured and recorded in the repository notebooks.

A controlled benchmark on the same hardware is required before making numerical speed claims.

### Trade-off Summary

The scratch model provides better overall internal Accuracy and Macro F1 with fewer parameters.

The pretrained VideoMAE model achieves higher Shoplifting Recall but produces more false positives.

Neither result establishes superiority on external surveillance footage.

---

# 🗂️ 6. Repository Structure

```text
ShopliftingDetection/
│
├── README.md
│
├── FromScratch/
│   ├── Shoplifting_EDA.ipynb
│   └── Shoplifting_Independent_V6.ipynb
│
├── pretrained/
│   ├── VideoMAE_Small_Finetuning.ipynb
│   └── VideoMAE_Small_No_Validation.ipynb
│
└── assets/
    └── eda/
        ├── class_distribution.png
        ├── duplicate_summary.png
        ├── duration_distribution.png
        ├── fps_resolution.png
        ├── sampled_frames_distribution.png
        └── split_distribution.png
```

The original video dataset and model checkpoints are stored outside this repository.

---

# 💻 7. Installation and Usage

## Requirements

The experiments use Python, PyTorch, OpenCV, Hugging Face Transformers, and common data-analysis libraries.

Recommended environment: Google Colab with an NVIDIA GPU.

### Clone the Repository

```bash
git clone https://github.com/mahmoudshoip94/ShopliftingDetection.git
cd ShopliftingDetection
```

### Install Dependencies

```bash
pip install torch torchvision "transformers>=4.45,<5" accelerate \
    opencv-python numpy pandas matplotlib scikit-learn Pillow tqdm
```

### Dataset Preparation

The dataset is not distributed in the repository.

The notebooks expect labeled video folders under the configured Google Drive dataset root.

The saved scratch split is located at:

```text
DataSet/
└── shoplifting_independent_v6_no_attention/
    └── video_split.csv
```

The pretrained notebooks reuse this manifest to preserve training and testing video identities.

### Run the EDA

Open:

`FromScratch/Shoplifting_EDA.ipynb`

Configure the dataset root and run the cells to inspect class distribution, video metadata, duplicates, and split consistency.

### Run the Scratch Model

Open:

`FromScratch/Shoplifting_Independent_V6.ipynb`

Run the notebook to prepare 1 FPS frame sequences, train the CNN-BiLSTM model, and evaluate the checkpoint.

### Run the Pretrained Model

Choose one of the two pretrained experiments:

**With Validation**

`pretrained/VideoMAE_Small_Finetuning.ipynb`

**Without Validation**

`pretrained/VideoMAE_Small_No_Validation.ipynb`

The notebooks load VideoMAE Small from Hugging Face, replace the classification head, prepare 16-frame inputs, train the model, and evaluate it.

### Model Outputs

The no-validation experiment saves the following files to Google Drive:

```text
shoplifting_pretrained_videomae_small_no_validation/
├── head_after_2_epochs.pt
├── final_finetuned_after_6_epochs.pt
├── training_history.csv
├── training_curves.png
├── final_test_predictions.csv
├── final_test_metrics.json
└── confusion_matrix.png
```

The pretrained weights are downloaded when the Hugging Face model is loaded.

---

# ⚠️ 8. Limitations

### Dataset Diversity

Most video footage originates from a relatively consistent CCTV environment.

The models may learn background-specific patterns that do not transfer to different stores.

### Dataset Leakage

Exact SHA-256 duplicates were removed before training.

However, byte-level duplicate detection does not guarantee that all visually overlapping recordings have been eliminated.

### Previously Inspected Test Set

The same test split was viewed during earlier experimentation.

The reported results should therefore be interpreted as internal experimental outcomes, not an untouched benchmark.

### No-Validation Experiment

The no-validation VideoMAE experiment does not use a validation set or validation-based early stopping.

Its final epoch was chosen in advance.

### Frame Sampling

Sampling at 1 FPS may miss rapid movements.

VideoMAE additionally uses only 16 frame positions per video, potentially discarding temporal detail from longer recordings or repeating content in shorter clips.

### External Generalization

Neither model has been independently validated on footage from unseen stores and cameras.

### Responsible Use

The system predicts video-level activity labels and does not provide proof of wrongdoing.

Its outputs should not be used as the sole basis for theft accusations or punitive actions.

---

# 🔮 9. Future Work

Future improvements include:

- Evaluating on videos from new cameras and stores.
- Creating recording-aware dataset splits.
- Expanding the training dataset.
- Running repeated experiments with multiple random seeds.
- Investigating alternative sampling strategies.
- Benchmarking end-to-end inference speed and GPU memory consumption.
- Investigating the difference between high recall and high precision.
- Building a Streamlit interface for video classification.

---

# 🏁 Conclusion

This project investigates shoplifting detection using both a custom deep learning architecture trained from scratch and a pretrained video Transformer.

The original dataset contained 855 CCTV videos, reduced to 632 cleaned videos after exact duplicate removal and known overlap exclusions.

The CNN-BiLSTM model achieved **95.79% Accuracy and 95.79% Macro F1**, while the pretrained VideoMAE Small model trained without validation achieved **91.58% Accuracy and 91.56% Macro F1**.

The pretrained model achieved higher Shoplifting Recall, while the smaller scratch model achieved better overall internal classification performance.

These experiments demonstrate the trade-offs between learning video representations from scratch and transferring representations from pretrained video models.

Further evaluation on independent surveillance footage is needed before drawing conclusions about real-world deployment reliability.
