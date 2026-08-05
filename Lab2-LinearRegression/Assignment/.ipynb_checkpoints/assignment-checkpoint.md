# 📘 Assignment: Linear Regression

## 📌 Objective

The goal of this assignment is to help you **understand and remember the complete Machine Learning (ML) pipeline** by applying it step-by-step to a **Linear Regression** problem.

By the end of this assignment, you should be able to:

- Clearly explain each stage of the ML pipeline
- Train and evaluate linear regression models
- Interpret model coefficients and intercepts
- Compare single-feature vs multi-feature regression models

---

## 🧠 The Machine Learning Pipeline

For **both tasks**, you must explicitly follow and document **all** of the following steps:

1. **Data Retrieval and Collection**
2. **Data Cleaning**
3. **Feature Design**
4. **Algorithm Selection**
5. **Loss Function Selection**
6. **Model Learning (Training)**
7. **Model Evaluation**

⚠️ *Skipping or merging steps will result in loss of marks.*

---

## 📊 Dataset

You will use the **California Housing Dataset**, which contains information about housing districts in California.

### Target Variable (Label)
- `median_house_value`

### Input Features (Examples)
- `housing_median_age`
- `total_rooms`
- `total_bedrooms`
- `population`
- `households`
- `median_income`
- etc.

---

## 🧪 Task 1: Simple Linear Regression (Single Feature)

### 🎯 Problem Statement

Build a **simple linear regression model** using **only one input feature** to predict housing prices.

- **Input Feature:** `housing_median_age`
- **Label:** `median_house_value`

---

### 🔍 Step-by-Step Requirements

#### 1️⃣ Data Retrieval and Collection
- Load the California Housing dataset
- Display basic information (shape, column names)

#### 2️⃣ Data Cleaning
- Handle missing values (if any)
- Explain how missing values were treated
- Verify data types

#### 3️⃣ Feature Design
- Select `housing_median_age` as the only input feature
- Explain why this feature was chosen
- Separate features (`X`) and label (`y`)

#### 4️⃣ Algorithm Selection
- Choose **Linear Regression**
- Explain why linear regression is appropriate for this task

#### 5️⃣ Loss Function Selection
- Use **Mean Squared Error (MSE)**
- Briefly explain what MSE measures

#### 6️⃣ Model Learning (Training)
- Split the data into training and testing sets
- Train the linear regression model
- Clearly state the learning process

#### 7️⃣ Model Evaluation
- Evaluate the model on test data
- Report:
  - Mean Squared Error (MSE)
  - (Optional) R² Score
- Provide a short interpretation of the results

### Extra
- Visualize the regression line for Task 1
- Plot predicted vs actual values
- Discuss assumptions of linear regression
---

### 📈 Model Interpretation (Mandatory)

After training the model, report:

- **Coefficient (slope)**
- **Intercept**

Explain:
- What does the coefficient represent?
- What does the intercept mean in this context?

---

## 🧪 Task 2: Multiple Linear Regression (All Features)

### 🎯 Problem Statement

Build a **multiple linear regression model** using **all available input features** (except the label) to predict housing prices.

- **Input Features:** All remaining features
- **Label:** `median_house_value`

---

### 🔍 Step-by-Step Requirements

Repeat **all seven ML pipeline steps** as in Task 1, but with the following differences:

#### 3️⃣ Feature Design
- Use **all features except** `median_house_value`
- Apply feature scaling if necessary
- Explain why multiple features may improve prediction performance

---

### 📈 Model Interpretation (Mandatory)

After training the model, report:

- **Intercept**
- **All coefficients**

Explain:
- What does each coefficient represent?
- How does this model differ from the single-feature model?

---

## 📊 Model Comparison

Provide a short comparison between **Task 1 and Task 2**:

- Which model performs better?
- Why does using multiple features help or hurt performance?
- Which model is easier to interpret?

---

## 📦 Deliverables

Submit:

- A **Jupyter Notebook (.ipynb)** or **Python script (.py)** containing:
  - Code
  - Clear markdown explanations for each ML pipeline step
- This completed assignment document (optional, if instructed)

---

## 📌 Reminder

> **This assignment is not just about coding — it is about understanding the machine learning process.**

Make sure each step of the pipeline is **clearly explained** in your own words.

Good luck! 🚀
