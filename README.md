# Exam Mock Analysis
Analyse the impact of the use of mocks ahead of the real exam.

---

# Global Exam Performance Analysis: A/B Testing & Predictive Modeling

## 📌 Executive Summary
This project analyses a dataset of 1,000 students across 6 global locations to determine the drivers of exam success.

**Key Business Question:** Does the optional "Mock Exam" actually improve student grades, or is it just a correlation?

Using **Linear Regression**, I isolated the impact of the mock exam and found it contributes a **+7.2 point increase** to the final score, controlling for degree status and attempt count. This suggests a strong causal link and ROI for the mock exam product.

## 🛠 Tools & Technologies
* **Python 3.9**
* **Pandas & NumPy:** For data generation and manipulation.
* **Scikit-Learn:** For Linear Regression and Logistic Classification.
* **Seaborn:** For statistical visualization.
* **Statistical Inference:** T-Tests and p-value assessment.

## 📊 Key Insights

### 1. The "Mock Exam" Effect (Linear Regression)
I built a regression model to predict final scores based on student attributes.
* **R-Squared:** 18% (Variance explained by the model).
* **Mock Exam Impact:** **+7.19 points**.
* **Degree Impact:** **+3.10 points**.
* **Retake Penalty:** **-2.59 points** per additional attempt.

> **Business Recommendation:** The data supports aggressive marketing of the Mock Exam, as it provides a quantifiable grade improvement distinct from the student's natural aptitude or education level.

### 2. Failure Prediction (Classification)
I trained a Logistic Regression model to flag students at risk of failing (<50%).
* **Overall Accuracy:** 71.5%
* **Recall (Passes):** 98%
* **Recall (Failures):** 11%

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
