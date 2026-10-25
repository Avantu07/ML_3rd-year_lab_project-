
# ❤️ Heart Disease Risk Prediction Using Machine Learning

## 📌 Project Overview

Heart Disease Risk Prediction is a machine learning-based web application developed using Python and Streamlit. The project predicts whether a patient may show signs of heart disease based on medical parameters such as age, sex, chest pain type, resting blood pressure, cholesterol level, maximum heart rate, and other clinical measurements.

The application uses the Cleveland Heart Disease dataset from the UCI Machine Learning Repository. It implements and compares four machine learning classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

The system displays the accuracy of each model and allows users to enter patient details, select a machine learning model, and generate a prediction through an interactive web interface.

**Note:** This project is intended for educational purposes only. It is not a medical diagnostic tool and should not be used to make healthcare decisions.

---

## 🎯 Objectives

- Develop a machine learning application for heart disease risk prediction.
- Load and preprocess heart disease data.
- Train and evaluate multiple classification algorithms.
- Compare the accuracy of different machine learning models.
- Build an interactive web interface using Streamlit.
- Allow users to enter patient medical information and obtain model predictions.
- Demonstrate the practical application of machine learning in healthcare data analysis.

---

## ✨ Features

- **Dataset Loading:** Loads the Cleveland Heart Disease dataset from the UCI Machine Learning Repository.
- **Data Preprocessing:** Handles missing values and converts the target variable into a binary classification format.
- **Multiple ML Models:** Trains four classification algorithms.
- **Model Comparison:** Displays model accuracy in a bar chart and a table.
- **Interactive Input Form:** Accepts patient medical details through the web interface.
- **Model Selection:** Allows users to select which trained model to use for prediction.
- **Prediction Results:** Displays whether the selected model indicates a possible heart disease signal.
- **Separate Files:** Keeps machine learning logic and user interface code in separate Python files.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Dataset loading and data manipulation |
| Scikit-learn | Machine learning model training and evaluation |
| Logistic Regression | Classification algorithm |
| Decision Tree | Classification algorithm |
| Random Forest | Ensemble classification algorithm |
| Support Vector Machine (SVM) | Classification algorithm |
| StandardScaler | Feature scaling for SVM |
| Streamlit | Interactive web application |
| UCI Machine Learning Repository | Dataset source |

---

## 📂 Project Structure

```text
Heart-Disease-Prediction/
│
├── app.py          # Data preprocessing, model training and prediction
├── ui.py           # Streamlit user interface
└── README.md       # Project documentation
```

### File Description

**1. `app.py`**

This file contains the machine learning functionality. It loads and preprocesses the dataset, splits the data into training and testing sets, trains four machine learning models, calculates their accuracy, and provides a function for predicting heart disease from patient details.

**2. `ui.py`**

This file contains the Streamlit interface. It displays dataset statistics, compares model accuracy, collects patient medical information, allows model selection, and presents the prediction result.

**3. `README.md`**

This file explains the project, its features, technologies, installation process, and execution steps.

---

## 📊 Dataset Information

**Dataset:** Cleveland Heart Disease Dataset

**Source:** UCI Machine Learning Repository

**Dataset URL:**  
https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data

The dataset is loaded directly from its URL when the application runs, so a separate dataset download is not required.

### Input Features

The application uses 13 input features:

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex encoded as a numeric value |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol level |
| `fbs` | Fasting blood sugar indicator |
| `restecg` | Resting electrocardiographic result |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina indicator |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels, represented in the dataset |
| `thal` | Thalassemia-related category in the dataset |

The dataset also contains a `target` column, which is used as the output label during training.

### Target Variable

The original target values are converted into two classes:

- **0:** No heart disease indicated by the dataset label.
- **1:** Heart disease indicated by the dataset label.

The code treats every original target value greater than zero as class `1`.

### Data Preprocessing

The application performs the following preprocessing steps:

1. Loads the dataset using Pandas.
2. Assigns column names to the dataset.
3. Treats `?` values as missing values.
4. Removes rows containing missing values.
5. Converts the original target variable into a binary classification label.
6. Separates the input features (`X`) from the target (`y`).
7. Splits the data into training and testing sets.

---

## 🤖 Machine Learning Algorithms

### 1. Logistic Regression

Logistic Regression is a classification algorithm that estimates the probability of an observation belonging to a class. It is used as one of the baseline models for heart disease prediction.

### 2. Decision Tree Classifier

A Decision Tree makes predictions by following a sequence of feature-based decision rules. It divides data into branches and produces a class prediction at the final leaf.

### 3. Random Forest Classifier

Random Forest combines predictions from multiple decision trees. In this project, the model uses 100 trees to produce a classification result.

### 4. Support Vector Machine (SVM)

SVM is a classification algorithm that learns a decision boundary between classes. In this project, `StandardScaler` is applied before SVM through a Scikit-learn pipeline.

Scaling helps place numeric features on comparable scales before model training.

---

## 🔄 Project Workflow

1. **Load Dataset:** Retrieve the Cleveland Heart Disease dataset.
2. **Preprocess Data:** Remove missing records and convert the target to two classes.
3. **Separate Data:** Divide the dataset into input features and target labels.
4. **Split Dataset:** Use 80% of the data for training and 20% for testing.
5. **Train Models:** Train Logistic Regression, Decision Tree, Random Forest, and SVM.
6. **Evaluate Models:** Generate test-set predictions and calculate accuracy scores.
7. **Display Comparison:** Show the model accuracy results in the Streamlit interface.
8. **Enter Patient Details:** Collect the 13 input values through the form.
9. **Select Model:** Choose one of the four trained models.
10. **Generate Prediction:** Pass the entered values to the selected model.
11. **Display Result:** Show the model's prediction on the interface.

---

## 📈 Model Evaluation

The application evaluates each model using **accuracy score**.

Accuracy is calculated as:

\[
\text{Accuracy} =
\frac{\text{Correct Predictions}}{\text{Total Predictions}}
\times 100
\]

The code displays accuracy as a percentage rounded to two decimal places.

The application uses a stratified train-test split with a test size of 20% and `random_state=42`. Stratification helps preserve the class distribution across the training and testing sets.

The model accuracy comparison is displayed using:

- A Streamlit bar chart.
- A table containing model names and accuracy percentages.

**Note:** Actual accuracy values depend on the dataset and the code execution environment. No fixed accuracy result is claimed in this README.

---

## 💻 Installation and Setup

### Prerequisites

Install the following before running the project:

- Python
- pip (Python package installer)
- Visual Studio Code or another Python editor
- Internet connection to retrieve the dataset

### Step 1: Clone the Repository

If the project is hosted on GitHub, clone it using:

```bash
git clone <YOUR_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd Heart-Disease-Prediction
```

Replace `<YOUR_REPOSITORY_URL>` with your actual GitHub repository URL.

Alternatively, open the folder containing `app.py` and `ui.py` directly in Visual Studio Code.

### Step 2: Install Dependencies

Open the terminal in the project directory and run:

```bash
python -m pip install pandas scikit-learn streamlit
```

These packages provide the data processing, machine learning, and web interface functionality required by the project.

### Step 3: Run the Application

Execute the following command:

```bash
python -m streamlit run ui.py
```

Streamlit will start the application and provide a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser to access the application.

---

## 🖥️ How to Use the Application

1. Start the Streamlit application.
2. Wait for the dataset to load and the models to train.
3. View the number of patient records and the number of models compared.
4. Review the model accuracy bar chart and comparison table.
5. Navigate to the **Enter Patient Details** section.
6. Enter the patient's medical parameters.
7. Select a machine learning model from the dropdown menu.
8. Click **Predict Heart Disease**.
9. Read the prediction result displayed on the page.

The application shows a warning when the selected model predicts class `1` and a success message when it predicts class `0`.

These messages represent model outputs, not confirmed medical diagnoses.

---

## 🔍 Implementation Details

### Model Training

The `load_and_train_models()` function in `app.py` handles dataset loading, preprocessing, splitting, model training, and accuracy evaluation.

It returns four objects:

- `df`: The cleaned dataset.
- `X`: The input feature dataset.
- `models`: The dictionary of trained machine learning models.
- `scores`: The accuracy scores of the models.

### Prediction

The `predict_heart_disease(patient, model)` function receives patient feature values and a trained model. It uses the model's `predict()` method and returns the predicted class.

### Streamlit Interface

The `ui.py` file imports the training and prediction functions from `app.py`.

The application uses `st.cache_resource` to cache the loaded models and avoid retraining them on every ordinary Streamlit rerun within the same running application process.

The patient form collects input values, creates a Pandas DataFrame with the expected feature columns, and passes that DataFrame to the selected model.

---

## ⚠️ Limitations

- The prediction depends on the quality and characteristics of the dataset used for training.
- Missing records are removed rather than imputed.
- Accuracy alone does not fully describe a medical classification model's performance.
- The current application does not display prediction probabilities.
- The application does not use live hospital records or real-time patient monitoring.
- The current implementation does not provide clinical validation.
- The model's output should not be treated as a confirmed diagnosis or a substitute for professional medical advice.

---

## 🚀 Future Enhancements

Possible future improvements include:

- Adding precision, recall, F1-score, and confusion matrix comparisons.
- Displaying model prediction probabilities where supported and appropriately calibrated.
- Improving the user interface with additional charts and clearer input descriptions.
- Adding model persistence to avoid retraining after every application restart.
- Evaluating model performance on additional datasets.
- Adding appropriate validation and testing before any real-world healthcare use.

These are potential enhancements and are not features currently implemented in the uploaded code.

---

## 🎓 Applications

- Educational demonstration of machine learning in healthcare.
- Comparison of classification algorithms.
- Learning data preprocessing and model evaluation.
- Demonstration of an interactive machine learning web application.
- Academic mini-projects related to data science and healthcare analytics.

---

## 👩‍💻 Project Information

**Project Title:** Heart Disease Risk Prediction Using Machine Learning

**Domain:** Machine Learning / Healthcare Data Analysis

**Programming Language:** Python

**Interface Framework:** Streamlit

**Dataset Source:** UCI Machine Learning Repository

---

## 📜 Disclaimer

This project is intended strictly for educational and academic purposes. It has not been established as a clinically validated medical device. Its predictions may be incorrect and must not be used to diagnose, treat, or rule out heart disease. Consult a qualified healthcare professional for medical evaluation and advice.

---

## 📄 License

No license is specified in the current project files. If you plan to publish this repository publicly, consider adding an appropriate license after confirming that you have the rights to distribute all included code and materials.
