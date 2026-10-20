# 🎓 Student Performance Predictor

A machine learning project that predicts a student's final mathematics grade using academic and study-related information. The project includes an interactive web application built with Streamlit.

## 📌 Project Overview

The Student Performance Predictor uses a **Random Forest Regressor** to estimate a student's final grade (`G3`) out of 20 based on selected features from the Student Performance dataset.

The goal of this project is to practice machine learning, model evaluation, Python development, and building interactive applications.

## ✨ Features

- Predicts a student's final mathematics grade.
- Uses study time, previous grades, past failures, and absences as input features.
- Provides an interactive interface built with Streamlit.
- Uses a trained machine learning model saved as a file.
- Includes input validation to help prevent invalid predictions.

## 🛠️ Technologies Used

- Python
- pandas
- scikit-learn
- Streamlit
- Jupyter Notebook

## 📊 Dataset

This project uses the **Student Performance** dataset from the UCI Machine Learning Repository.

**Source:** [UCI Student Performance Dataset](https://doi.org/10.24432/C5TG7T)

The dataset contains student demographic, social, and school-related information. This project uses a selected subset of academic features to predict the final mathematics grade.

Reference:

Cortez, P., & Silva, A. (2008). *Using Data Mining to Predict Secondary School Student Performance.*

## 🤖 Machine Learning Model

The project uses a `RandomForestRegressor` with 100 trees and a fixed random seed for reproducibility.

**Input features:**
- `studytime` — weekly study-time category
- `failures` — number of past class failures
- `absences` — number of school absences
- `G1` — first-period grade
- `G2` — second-period grade

**Prediction target:**
- `G3` — final mathematics grade, on a scale from 0 to 20

The model achieved a Mean Absolute Error (MAE) of approximately **1.05 grade points** in the evaluation previously performed for this project. This estimate depends on the evaluation split and does not guarantee the same accuracy on new students.

## 📁 Project Structure

```text
Student-Performance-Predictor/
├── data/
│   └── student-mat.csv
├── app.py
├── train_model.py
├── student_model.pkl
├── student_predictor.ipynb
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/boatengaseda41-dotcom/Student-Performance-Predictor.git
cd Student-Performance-Predictor
```

### 2. Install the dependencies

```bash
python -m pip install pandas scikit-learn streamlit jupyter
```

### 3. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL where you can open the prediction interface in your browser.

### 4. Retrain the model (optional)

If you want to train the model again, make sure the dataset is available in the expected `data/` directory, then run:

```bash
python train_model.py
```

## 📈 Evaluation

The model was evaluated using Mean Absolute Error (MAE). An MAE of approximately 1.05 means the model's predictions differed from actual grades by about 1.05 points on average on the evaluation data.

## ⚠️ Limitations

- The model is based on data from Portuguese schools and may not generalize to students in Ghana or other education systems.
- Predictions are estimates, not guaranteed final grades.
- The model's performance depends on the data used to train and evaluate it.
- Features such as previous-period grades can be strongly associated with the final grade, so predictions made without reliable previous grades may be less useful.

## 🎯 What I Learned

- Preparing data for machine learning.
- Training and evaluating a regression model.
- Saving and loading a trained model.
- Building an interactive application with Streamlit.
- Using GitHub to document and share a software project.

## 👨‍💻 Author

**Aseda Kofi Boateng**

GitHub: [@boatengaseda41-dotcom](https://github.com/boatengaseda41-dotcom)
