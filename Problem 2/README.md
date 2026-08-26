# README — Bank Term Deposit Subscription Prediction Using Logistic Regression

## 📌 Project Overview

This project develops a **Logistic Regression machine learning model** to predict whether a bank customer will subscribe to a **term deposit**.

The prediction is based on customer information such as:

* Age
* Job
* Marital status
* Education
* Account balance
* Housing loan
* Personal loan
* Contact information
* Campaign information
* Previous marketing outcomes

The target variable is:

```text
y
```

where:

```text
yes → Customer subscribed to the term deposit
no  → Customer did not subscribe
```

## 🎯 Objective

The main objective is to build a classification model that predicts whether a customer is likely to subscribe to a term deposit based on their banking and marketing information.

This is a **binary classification problem**.

## 📊 Dataset

The dataset is loaded from Google Drive:

```text
bank-data/bank-full.csv
```

The dataset contains:

```text
45,211 rows
17 columns
```

The dataset includes:

### Numerical Features

```text
age
balance
day
duration
campaign
pdays
previous
```

### Categorical Features

```text
job
marital
education
default
housing
loan
contact
month
poutcome
```

### Target Variable

```text
y
```

The target distribution in the notebook is:

```text
No  → 39,922
Yes → 5,289
```

## 🛠️ Technologies and Libraries

The project uses:

* Python
* Google Colab
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

Important Scikit-learn components include:

* `train_test_split`
* `StandardScaler`
* `OneHotEncoder`
* `ColumnTransformer`
* `Pipeline`
* `LogisticRegression`
* `accuracy_score`
* `classification_report`
* `confusion_matrix`
* `ConfusionMatrixDisplay`

## 📥 Data Loading

The dataset is loaded using Pandas:

```python
df = pd.read_csv(file_path, sep=";")
```

The notebook then displays:

* Dataset shape
* First five rows
* Dataset information
* Missing values
* Target distribution

## 🧹 Data Checking

The notebook checks for missing values using:

```python
df.isnull().sum()
```

According to the notebook output, all columns contain:

```text
0 missing values
```

Therefore, no missing-value imputation was required.

## 🔢 Feature and Target Separation

The target column is separated from the input features.

```python
X = df.drop("y", axis=1)
```

The target is converted into binary numerical values:

```python
yes → 1
no  → 0
```

using:

```python
y = df["y"].map({
    "yes": 1,
    "no": 0
})
```

## ⚙️ Data Preprocessing

Different preprocessing techniques are applied to numerical and categorical features.

### Numerical Features

Numerical features are standardized using:

```python
StandardScaler()
```

### Categorical Features

Categorical features are converted into numerical representations using:

```python
OneHotEncoder(handle_unknown="ignore")
```

The preprocessing steps are combined using:

```python
ColumnTransformer
```

## ✂️ Train-Test Split

The dataset is divided into training and testing sets using:

```python
train_test_split()
```

Configuration:

```text
Training Data: 80%
Testing Data: 20%
Random State: 42
Stratified Split: Yes
```

Resulting dataset sizes:

```text
Training Shape: (36,168, 16)
Testing Shape:  (9,043, 16)
```

## 🧠 Logistic Regression Model

The classifier is created using:

```python
LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

The increased `max_iter` value allows the optimization algorithm more iterations to converge.

## 🔗 Machine Learning Pipeline

A Scikit-learn pipeline combines preprocessing and classification:

```text
Raw Input Data
       ↓
ColumnTransformer
       ↓
├── StandardScaler
│
└── OneHotEncoder
       ↓
Logistic Regression
       ↓
Prediction
```

The pipeline is implemented as:

```python
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", logistic_model)
    ]
)
```

## 🏋️ Model Training

The model is trained using:

```python
model.fit(X_train, y_train)
```

Because preprocessing is included inside the pipeline, the training process automatically applies:

1. Numerical feature scaling
2. Categorical feature encoding
3. Logistic Regression training

## 🔮 Prediction

Predictions are generated using:

```python
y_pred = model.predict(X_test)
```

## 📈 Model Performance

The model achieved an accuracy of:

```text
90.12%
```

### Classification Report

| Class    | Precision | Recall | F1-Score | Support |
| -------- | --------: | -----: | -------: | ------: |
| No       |      0.92 |   0.97 |     0.95 |   7,985 |
| Yes      |      0.64 |   0.35 |     0.45 |   1,058 |
| Accuracy |           |        |     0.90 |   9,043 |

The overall accuracy is high, but the model performs better for the **No** class than the **Yes** class.

## 🔲 Confusion Matrix

The confusion matrix produced by the model is:

```text
[[7782  203]
 [ 690  368]]
```

This represents:

```text
True Negatives  → 7,782
False Positives → 203
False Negatives → 690
True Positives  → 368
```

## 📊 Confusion Matrix Visualization

The notebook visualizes the confusion matrix using:

```python
ConfusionMatrixDisplay
```

The labels are:

```text
No
Yes
```

This visualization helps analyze how many customers were correctly and incorrectly classified.

## 📦 Project Structure

```text
Bank-Term-Deposit-Prediction/
│
├── prob2.ipynb
├── bank-data/
│   └── bank-full.csv
└── README.md
```

## 🚀 How to Run

1. Open the notebook in Google Colab or Jupyter Notebook.
2. Mount Google Drive.
3. Make sure the dataset is available at:

```text
/content/drive/MyDrive/bank-data/bank-full.csv
```

4. Run all cells sequentially.

The notebook will:

```text
Load Dataset
      ↓
Check Dataset
      ↓
Separate Features and Target
      ↓
Scale Numerical Features
      ↓
Encode Categorical Features
      ↓
Split Training and Testing Data
      ↓
Train Logistic Regression Model
      ↓
Make Predictions
      ↓
Evaluate Accuracy
      ↓
Generate Classification Report
      ↓
Display Confusion Matrix
```

## 📌 Conclusion

This project demonstrates an end-to-end **Logistic Regression classification workflow** for predicting customer term deposit subscriptions.

The model achieved an overall accuracy of **90.12%**. However, the classification report shows that performance for customers who actually subscribed (`Yes`) is weaker than for customers who did not subscribe (`No`).

The complete workflow integrates preprocessing and classification into a single Scikit-learn pipeline, making the model easier to train and apply to new customer data.
