# README — Pneumonia Detection Using CNN

## 📌 Project Overview

This project implements a **Convolutional Neural Network (CNN)** to classify chest X-ray images into two categories:

* **NORMAL**
* **PNEUMONIA**

The model is trained using TensorFlow/Keras and uses image preprocessing, data augmentation, convolutional layers, pooling layers, global average pooling, dropout, and a sigmoid output layer.

## 🎯 Objective

The objective of this project is to build a deep learning model capable of automatically detecting whether a chest X-ray image belongs to a **normal patient** or a patient with **pneumonia**.

This is a **binary image classification problem**.

## 📂 Dataset Structure

The dataset is organized into three directories:

```text
problemds/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The notebook loaded:

* Training images: **5,216**
* Validation images: **16**
* Testing images: **624**

The two classes are:

```text
['NORMAL', 'PNEUMONIA']
```

## 🛠️ Technologies and Libraries

The project uses:

* Python
* Google Colab
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## ⚙️ Data Loading

Images are loaded using TensorFlow's `image_dataset_from_directory()`.

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
```

The dataset uses:

* Image size: **224 × 224**
* Batch size: **32**
* Binary labels
* Training data shuffling
* Fixed random seed of `42`

## 🔄 Data Augmentation

To improve generalization, the following augmentation techniques are applied to the training images:

* Horizontal flipping
* Random rotation
* Random zoom
* Random contrast adjustment

```python
layers.RandomFlip("horizontal")
layers.RandomRotation(0.05)
layers.RandomZoom(0.1)
layers.RandomContrast(0.1)
```

## 🧹 Image Preprocessing

Pixel values are normalized from the range:

```text
[0, 255]
```

to:

```text
[0, 1]
```

using:

```python
layers.Rescaling(1./255)
```

## 🧠 CNN Architecture

The model uses a custom CNN architecture.

### Block 1

```text
Conv2D (32 filters)
Conv2D (32 filters)
MaxPooling2D
```

### Block 2

```text
Conv2D (64 filters)
Conv2D (64 filters)
MaxPooling2D
```

### Block 3

```text
Conv2D (128 filters)
Conv2D (128 filters)
MaxPooling2D
```

### Block 4

```text
Conv2D (256 filters)
MaxPooling2D
```

### Classification Layers

```text
GlobalAveragePooling2D
Dense (128, ReLU)
Dropout (0.5)
Dense (1, Sigmoid)
```

The model contains approximately:

```text
Total Parameters: 615,201
Trainable Parameters: 615,201
```

## ⚡ Model Compilation

The model uses:

```python
optimizer = Adam(learning_rate=0.0001)
loss = binary_crossentropy
```

The evaluation metrics are:

* Accuracy
* Precision
* Recall

## 🏋️ Model Training

The model was configured to train for:

```text
20 epochs
```

The training process includes the following callbacks:

### Early Stopping

Training stops when validation loss does not improve for 5 epochs.

### Model Checkpoint

The best model is saved as:

```text
best_cnn_model.keras
```

### Reduce Learning Rate

The notebook defines `ReduceLROnPlateau`, although the training call uses:

```python
EarlyStopping
ModelCheckpoint
```

## 📊 Training Results

The model stopped early after 9 epochs and restored the weights from the best epoch.

The best validation loss was obtained at:

```text
Epoch 4
```

At that epoch, the recorded values were approximately:

* Training Accuracy: **85.26%**
* Validation Accuracy: **68.75%**
* Validation Loss: **0.4932**

## 📈 Visualization

The notebook visualizes:

### Training vs Validation Accuracy

This graph compares the training accuracy and validation accuracy across epochs.

### Training vs Validation Loss

This graph compares the training loss and validation loss across epochs.

These visualizations help analyze model learning behavior and possible overfitting.

## 📦 Project Structure

```text
Pneumonia-Detection/
│
├── prob1.ipynb
├── best_cnn_model.keras
└── README.md
```

## 🚀 How to Run

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Mount Google Drive.
3. Update the dataset paths if necessary:

```python
train_dir = "/content/drive/MyDrive/problemds/train"
val_dir = "/content/drive/MyDrive/problemds/val"
test_dir = "/content/drive/MyDrive/problemds/test"
```

4. Run all notebook cells sequentially.
5. The trained model will be saved as:

```text
best_cnn_model.keras
```

## 📌 Conclusion

This project demonstrates the implementation of a custom CNN for **binary chest X-ray classification**. The model learns visual features through multiple convolutional and pooling layers and uses data augmentation, normalization, dropout, and early stopping during training.

The final workflow includes:

```text
Chest X-ray Image
        ↓
Data Augmentation
        ↓
Normalization
        ↓
Convolutional Layers
        ↓
Max Pooling
        ↓
Global Average Pooling
        ↓
Dense Layer
        ↓
Sigmoid Output
        ↓
NORMAL / PNEUMONIA
```
