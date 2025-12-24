# Exam Mock Analysis
Analyse the impact of the use of the mock resource ahead of the 'real' exam.

---

# Global Exam Performance Analysis: A/B Testing & Predictive Modeling

## 📌 Executive Summary
This project analyses a (generated) dataset of 1,000 students across 6 global locations to determine the drivers of exam success.

<img width="541" height="114" alt="image" src="https://github.com/user-attachments/assets/233b7e83-d105-44ef-9940-471157e34d68" />

**Key Business Question:** Does the optional "Mock Exam" actually improve student grades, or is it just a correlation?

Using **Linear Regression**, I isolated the impact of the mock exam and found it contributes a **+7.2 point increase** to the final score, controlling for degree status and attempt count. This suggests a strong causal link and ROI for the mock exam product.

## 🛠 Tools & Technologies
* **Python 3.9**
* **Pandas & NumPy:** For data generation and manipulation.
* **Scikit-Learn:** For Linear Regression and Logistic Classification.
* **Seaborn:** For statistical visualisation.
* **Statistical Inference:** T-Tests and p-value assessment.

## 📊 Key Insights

### 1. The A/B Test (Statistical Inference)
I compared the average scores of students who used the mock exam vs. those who didn't.
* **Result:** Students using the mock scored significantly higher (Average: 62 vs 55).
* **Statistical Significance:** A T-Test yielded a p-value < 0.05, proving the difference is real and not due to chance.
<img width="436" height="340" alt="image" src="https://github.com/user-attachments/assets/58ffe10c-6ea8-42e6-b7a6-4160885a820b" />

  
### 2. The "Mock Exam" Effect (Linear Regression)
I built a regression model to predict final scores based on student attributes.
* **R-Squared:** 18% (Variance explained by the model).
* **Mock Exam Impact:** **+7.19 points**.
* **Degree Impact:** **+3.10 points**.
* **Retake Penalty:** **-2.59 points** per additional attempt.
<img width="357" height="243" alt="image" src="https://github.com/user-attachments/assets/e23087a1-3206-4ddf-8732-7bb38e9f8f14" />

> **Business Recommendation:** The data supports aggressive marketing of the Mock Exam, as it provides a quantifiable grade improvement distinct from the student's natural aptitude or education level.

### 3. Failure Prediction (Classification)
I trained a Logistic Regression model to flag students at risk of failing (<50%).
* **Overall Accuracy:** 71.5%
* **Recall (Passes):** 98%
* **Recall (Failures):** 11%
<img width="395" height="178" alt="image" src="https://github.com/user-attachments/assets/cf94cc54-07c5-47fb-9913-eb628cb98181" />
<img width="230" height="180" alt="image" src="https://github.com/user-attachments/assets/2635f909-1a03-4049-93c9-2fcb15b8b5b0" />

>  *Note:* The current model is conservative; it underestimates the number of failures. Future iterations would utilise **SMOTE (Synthetic Minority Over-sampling Technique)** to better balance the "Fail" class and catch more at-risk students.

## 📂 Project Structure
* `data_generator.py`: Python script used to create the synthetic dataset with realistic statistical distributions (Gaussian noise injected).
* `analysis_script.py`: The end-to-end analysis script containing the A/B Test (Project A), Linear Regression (Project B), and Classification (Project C).
* `global_exam_data.csv`: The dataset used.

## 🚀 How to Run
If you want to reproduce this analysis on your own machine:

1. **Clone this repository** (or download the files).
2. **Install the required libraries**:
   ```bash
   pip install pandas scikit-learn seaborn matplotlib
   ```
3. **Generate the data using the file**:
```data_generator.py```
4. **Run the analysis script where all three Projects are combined**:
   ```analysis_script.py```
* Select all or any project, note that you will need to preprocess data ahead of running machine learning projects.
