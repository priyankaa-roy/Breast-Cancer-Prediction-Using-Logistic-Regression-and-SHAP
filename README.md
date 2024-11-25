# 🧑‍⚕️ Breast Cancer Prediction Using Logistic Regression
This project uses logistic regression to predict whether a breast cancer tumor is benign or malignant based on various features. It also includes visualizations, model evaluation, and SHAP analysis for understanding feature importance.

## 📚 Table of Contents 
1. Overview
2. Project Setup
3. Data Exploration & Preprocessing
4. Logistic Regression Model
5. Model Evaluation
6. Feature Importance with SHAP
7. Visualizations
8. Real-World Insights
9. Requirements
10. License

## 📝 Overview

This project aims to predict the nature of breast cancer tumors (benign or malignant) using various medical features. We use a Logistic Regression model to classify the data and evaluate its performance. Additionally, SHAP (Shapley Additive Explanations) is used to explain the model's predictions and interpret feature importance.

![image](https://github.com/user-attachments/assets/60063eb5-68e4-430c-a872-b84193cb60cc)

![image](https://github.com/user-attachments/assets/ee8b4b88-88b8-4988-bb05-2ec83b058487)



## 🛠️ Project Setup
To get started with this project, follow these steps:

1. Clone the repository:
- git clone [https://github.com/yourusername/breast-cancer-prediction.git](https://github.com/priyankaa-roy/Breast-Cancer-Prediction-Using-Logistic-Regression-and-SHAP)

2. Navigate into the project directory:
- cd breast-cancer-prediction

3. Install required dependencies:
- pip install -r requirements.txt

  
## 📊 Data Exploration & Preprocessing
The dataset used is the `Breast Cancer Wisconsin (Diagnostic)` dataset, which includes 30 numerical features such as:

- `Mean Radius`: The mean of the distances from the center to points on the perimeter.
- `Mean Texture`: Standard deviation of gray-scale values in the image.

And more…

![image](https://github.com/user-attachments/assets/3a85a7a5-b04e-46f6-a3da-26684d9c3976)


#### Key Steps:
- Checked for missing values ❌.
- Verified feature types 🔢.
- Explored feature distributions using pair plots 🔍.

  
## 🤖 Logistic Regression Model
A `Logistic Regression` model is trained on the dataset to classify the tumors into two categories: `Benign` and `Malignant`.

##### Code:
    - from sklearn.linear_model import LogisticRegression
    - from sklearn.model_selection import train_test_split
    - from sklearn.datasets import load_breast_cancer

##### Load dataset
    - data = load_breast_cancer()
    - X = data.data
    - y = data.target

##### Split data into train and test sets
    - X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##### Train Logistic Regression model
    - logistic_regression = LogisticRegression(random_state=42)
    - logistic_regression.fit(X_train, y_train)

##### Make predictions
    - y_pred = logistic_regression.predict(X_test)

![image](https://github.com/user-attachments/assets/8d165147-0240-43bb-b4db-6ae57f5a749a)


## 📈 Model Evaluation

The model’s performance was evaluated using:

- `Accuracy Score ✔️`: The proportion of correct predictions.
- `Confusion Matrix 🧩`: A detailed matrix to show the true positive, true negative, false positive, and false negative counts.
- `Classification Report 📊`: Includes precision, recall, and F1-score for each class.


## 💡 Feature Importance with SHAP 

We used SHAP (Shapley Additive Explanations) to interpret the model's decision-making process and understand which features most influenced the predictions.

##### Code:
   - import shap

##### Create SHAP explainer
   - explainer = shap.LinearExplainer(logistic_regression, X_train)
   - shap_values = explainer(X_test)

##### Visualize feature importance
   - shap.summary_plot(shap_values, X_test)

![image](https://github.com/user-attachments/assets/7e45a4a8-5929-4e75-8f40-1d589c895c76)


This visualization helps identify the features that contribute most to the prediction, which can guide doctors in making better-informed decisions based on the model's output.

## 📊 Visualizations

The following visualizations were created to better understand the dataset and the model:

- Pair Plot 🎨: Shows the relationship between key features (e.g., `mean radius`, `mean texture`) and their impact on the target variable.
- Correlation Matrix Heatmap 🔥: Displays how correlated the features are to one another, helping identify redundancies.
- SHAP Summary Plot 📈: Visualizes the global feature importance for the logistic regression model.

### Example Pair Plot:

   - sns.pairplot(df, hue="target", vars=["mean radius", "mean texture", "mean perimeter", "mean area"], diag_kind="kde")
   - plt.show()

![image](https://github.com/user-attachments/assets/8632667f-70a1-44e9-a012-a0597544346e)

## 🌍 Real-World Insights
- `Early Detection`: The features used in the model (e.g., mean radius, mean area) are crucial for early-stage cancer detection.
- `Risk Stratification`: The model can be used to predict the likelihood of cancer being malignant, helping doctors prioritize high-risk patients for biopsy or treatment.
- `Improved Diagnosis`: The use of machine learning models like logistic regression in clinical settings can help reduce human error and speed up diagnoses.


## 📋 Requirements
Ensure you have the following libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`
- `shap`

You can install the required libraries with the command:
    - pip install -r requirements.txt


## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
