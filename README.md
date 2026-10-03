# Bone-Fracture-Detection
Binary classification of bone X-ray images (**fractured / not fractured**) using transfer learning with MobileNetV2 (TensorFlow/Keras).

## Dataset
[Bone Fracture Dataset](https://www.kaggle.com/datasets/osamajalilhassan/bone-fracture-dataset) (Kaggle)
- Training: 4,480 fractured / 4,383 not fractured
- Testing: 360 fractured / 240 not fractured
- 20% of the training set was held out for validation

## Method
1. Load images at 224×224 with `image_dataset_from_directory`
2. MobileNetV2 (ImageNet weights) as a frozen feature extractor + GlobalAveragePooling + Dropout + Softmax head
3. Stage 1: train the head (10 epochs, lr = 1e-3)
4. Stage 2: fine-tune the last 30 layers (lr = 1e-5)
5. Evaluate on the separate test set

## Results
| Metric | Value |
|---|---|
| Validation accuracy | ~99% |
| Test accuracy | XX% |
| Recall (fractured) | XX% |

![Training curves](images/training_curves.png)
![Confusion matrix](images/confusion_matrix.png)

## Usage
Open the notebook in Google Colab (GPU runtime), run the cells in order. The last cell lets you upload an X-ray image and returns the predicted class with confidence.

## Tech stack
Python, TensorFlow/Keras, scikit-learn, Matplotlib, Seaborn

## Limitations
- Trained on a single public dataset; may not generalize to other devices or hospitals
- Validation split comes from the training folder, so the test set is the more reliable metric
- No clinical validatio
