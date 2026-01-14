# ML Assignment 2 - Classification Models Evaluation

## a. Problem Statement

The objective of this assignment is to implement and evaluate multiple machine learning classification models on a real-world dataset. The goal is to predict student grade classification based on various student characteristics, academic performance, and support factors. This is a **multi-class classification problem** where we need to classify students into multiple grade categories (GradeClass: 0.0, 1.0, 2.0, 3.0, 4.0) based on their overall academic performance.

The dataset contains information about students including their demographics, academic metrics (GPA, study time, absences), parental support factors, and extracurricular activities. The target variable is GradeClass, which represents the student's grade classification.

The assignment involves:
- Implementing 6 different classification algorithms
- Evaluating each model using 6 different metrics
- Building an interactive Streamlit web application
- Deploying the application on Streamlit Community Cloud
- Comparing model performance and providing insights

## b. Dataset Description

**Dataset Name:** Student Performance Dataset (Kaggle Link - https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset)

**Source:** Student_performance_data.csv (placed in `data/` folder)

**Dataset Characteristics:**
- **Number of Instances:** 2,393 samples
- **Number of Features:** 14 features (13 input features + 1 target variable)
- **Target Variable:** GradeClass (multi-class classification with 5 classes: 0.0, 1.0, 2.0, 3.0, 4.0)
- **Missing Values:** None (clean dataset)

**Feature Description:**

1. **StudentID** - Unique identifier for each student
2. **Age** - Age of the student (continuous)
3. **Gender** - Gender of the student (encoded: 0, 1)
4. **Ethnicity** - Ethnicity of the student (encoded: 0, 1, 2, 3)
5. **ParentalEducation** - Parental education level (encoded: 0, 1, 2, 3, 4)
6. **StudyTimeWeekly** - Number of hours studied per week (continuous)
7. **Absences** - Number of absences (continuous)
8. **Tutoring** - Whether student receives tutoring (encoded: 0, 1)
9. **ParentalSupport** - Level of parental support (encoded: 0, 1, 2, 3, 4)
10. **Extracurricular** - Participation in extracurricular activities (encoded: 0, 1)
11. **Sports** - Participation in sports (encoded: 0, 1)
12. **Music** - Participation in music activities (encoded: 0, 1)
13. **Volunteering** - Participation in volunteering (encoded: 0, 1)
14. **GPA** - Grade Point Average (continuous, 0-4 scale)

**Target Variable:**
- **GradeClass** - Student grade classification (multi-class)
  - 0.0: Excellent
  - 1.0: Good
  - 2.0: Average
  - 3.0: Below Average
  - 4.0: Poor

**Data Preprocessing:**
- Dataset loaded from `data/Student_performance_data.csv`
- Target variable (GradeClass) is already in categorical format (multi-class classification)
- Categorical features are already encoded in the dataset
- Features standardized using StandardScaler for models requiring scaling (Logistic Regression, KNN, Naive Bayes)
- Train-test split: 80% training (~1,914 samples), 20% testing (~479 samples) with stratification

## c. Models Used

### Comparison Table with Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7516 | 0.8936 | 0.5965 | 0.5694 | 0.5762 | 0.6265 |
| Decision Tree | 0.9186 | 0.9170 | 0.8490 | 0.8554 | 0.8519 | 0.8794 |
| kNN | 0.6242 | 0.7750 | 0.4468 | 0.4268 | 0.4266 | 0.4363 |
| Naive Bayes | 0.7516 | 0.8987 | 0.7368 | 0.5940 | 0.5863 | 0.6380 |
| Random Forest (Ensemble) | 0.9081 | 0.9796 | 0.8790 | 0.8006 | 0.8149 | 0.8629 |
| XGBoost (Ensemble) | 0.9207 | 0.9906 | 0.8608 | 0.8546 | 0.8551 | 0.8828 |

*Note: All metrics are calculated using macro-averaging for multi-class classification. Results are rounded to 4 decimal places.*

### Observations about Model Performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Achieved moderate performance (Accuracy: 0.7516, AUC: 0.8936) with decent MCC (0.6265). The model provides a linear decision boundary and is highly interpretable. Lower precision (0.5965) and recall (0.5694) suggest it struggles with the multi-class classification task, likely due to non-linear relationships in the data. Requires feature scaling for optimal performance and is computationally efficient. |
| Decision Tree | Excellent performance (Accuracy: 0.9186, AUC: 0.9170) with strong balanced metrics (F1: 0.8519, MCC: 0.8794). The model successfully captures non-linear relationships without requiring feature scaling. High recall (0.8554) indicates good class detection. The model is highly interpretable and handles both numerical and categorical features well. Regularization (max_depth=10) helped prevent overfitting. |
| kNN | Lowest performance across all models (Accuracy: 0.6242, AUC: 0.7750). Poor precision (0.4468) and recall (0.4268) suggest the instance-based approach struggles with this multi-class problem. The model is sensitive to feature scaling and may be affected by irrelevant features or class imbalance. Despite proper scaling, performance remains suboptimal, indicating kNN may not be suitable for this dataset. |
| Naive Bayes | Moderate accuracy (0.7516) but strong AUC (0.8987) and highest precision (0.7368) among non-ensemble models. However, lower recall (0.5940) indicates the model is conservative in predictions. The feature independence assumption may not fully hold, but the model still performs reasonably well. Fast training and good for baseline comparisons. |
| Random Forest (Ensemble) | Strong performance (Accuracy: 0.9081) with exceptional AUC (0.9797), the highest among all models. Excellent precision (0.8790) but slightly lower recall (0.8006) suggests the model is slightly conservative. The ensemble approach successfully reduces overfitting and provides robust predictions. Feature importance can be extracted for interpretability. |
| XGBoost (Ensemble) | Best overall performance (Accuracy: 0.9207, AUC: 0.9906, MCC: 0.8828). The gradient boosting approach achieves the highest accuracy and AUC, demonstrating superior ability to capture complex patterns. Balanced precision (0.8608) and recall (0.8546) indicate well-calibrated predictions. The model's sequential error correction mechanism proves highly effective for this multi-class problem. |

## Project Structure

```
ML-Assignment-2/
│
├── .venv/                              # Python virtual environment
├── .gitignore                          # Git ignore file
│
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
├── generate_model_results.py           # Script to generate Excel results
├── model_results.xlsx                  # Excel file with model results
├── ML_Assignment_2_Results.xlsx        # Detailed Excel results
│
├── data/
│   ├── __pycache__/                    # Python cache (auto-generated)
│   ├── Student_performance_data.csv    # Student performance dataset
│   └── validate_dataset.py             # Dataset validation script
│
├── models/                              # Model training scripts
│   ├── __pycache__/                    # Python cache (auto-generated)
│   ├── __init__.py                     # Package initialization
│   ├── train_models.py                 # Main model training script
│   ├── logistic_regression.py          # Logistic Regression classifier
│   ├── Decision_Tree_Classifier.py    # Decision Tree classifier
│   ├── KNeighbors_Classifier.py       # K-Nearest Neighbors classifier
│   ├── Naive_Bayes_Classifier.py      # Naive Bayes classifier
│   ├── Random_Forest_Classifier.py    # Random Forest classifier
│   └── XGBoost_Classifier.py          # XGBoost classifier
│
└── saved_models/                       # Saved trained models (gitignored)
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    └── scaler.pkl
```

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps to Run

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ML-Assignment-2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset is ready**
   - The dataset `Student_performance_data.csv` is already in the `data/` folder
   - Contains 2,393 instances with 14 features (13 input + 1 target)
   - Target variable (GradeClass) is multi-class with 5 categories (0.0, 1.0, 2.0, 3.0, 4.0)

4. **Train all models**
   ```bash
   # Train all models
   python models/train_models.py
   
   # Generate detailed Excel results (optional)
   python generate_model_results.py
   ```

5. **Run Streamlit app locally**
   ```bash
   streamlit run app.py
   ```

## Streamlit Application Features

The Streamlit application includes the following features:

1. **Dataset Upload**: Users can upload CSV files containing test data for evaluation
2. **Model Selection**: Dropdown menu to select from 6 different classification models
3. **Evaluation Metrics Display**: Shows all 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
4. **Confusion Matrix**: Visual representation of model predictions
5. **Classification Report**: Detailed per-class metrics

## Deployment on Streamlit Community Cloud

1. Push your code to a GitHub repository
2. Go to https://streamlit.io/cloud
3. Sign in with your GitHub account
4. Click "New App"
5. Select your repository and branch (usually `main`)
6. Set the main file path to `app.py`
7. Click "Deploy"

## Evaluation Metrics Explanation

- **Accuracy**: Proportion of correct predictions out of all predictions
- **AUC (Area Under ROC Curve)**: Measures the model's ability to distinguish between classes
- **Precision**: Proportion of positive predictions that are actually positive
- **Recall**: Proportion of actual positives that are correctly identified
- **F1 Score**: Harmonic mean of precision and recall
- **MCC (Matthews Correlation Coefficient)**: Balanced measure that considers all confusion matrix values

## Results and Analysis

After training all models, the results are saved in:
- `model_results.xlsx` - Basic results table with all 6 metrics for each model
- `ML_Assignment_2_Results.xlsx` - Detailed results with multiple sheets including:
  - Model Comparison: Summary table with all metrics
  - Detailed Metrics: Transposed view of metrics
  - Best Models: Best performing model for each metric
  - Model Rankings: Ranking of models by each metric

## Key Findings

### Model Performance Insights
- **Best Overall Performance**: XGBoost achieves the highest accuracy (0.9207), AUC (0.9906), and MCC (0.8828), making it the best-performing model overall
- **Strong Tree-based Performance**: Decision Tree shows excellent performance (0.9186 accuracy) and is the second-best model, demonstrating that tree-based methods work well for this dataset
- **Ensemble Advantage**: Both Random Forest (0.9081 accuracy) and XGBoost outperform individual models, confirming the benefit of ensemble methods
- **Moderate Linear Models**: Logistic Regression (0.7516 accuracy) and Naive Bayes (0.7516 accuracy) show moderate performance, with Naive Bayes having better precision (0.7368) but lower recall (0.5940)
- **Lowest Performance**: kNN shows the lowest performance across all metrics (0.6242 accuracy), suggesting that instance-based learning may not be optimal for this problem
- **Multi-class Classification**: All models are evaluated using macro-averaged metrics for multi-class classification
- **Feature Scaling**: Logistic Regression, KNN, and Naive Bayes models use StandardScaler for feature normalization
- **Tree-based Models**: Decision Tree, Random Forest, and XGBoost work with original feature scales
- **Model Evaluation**: All models are evaluated using 6 comprehensive metrics: Accuracy, AUC, Precision, Recall, F1 Score, and MCC

### Technical Observations
1. **Data Preprocessing**: The dataset required minimal preprocessing as categorical features were already encoded
2. **Class Imbalance**: Stratified train-test split ensures balanced representation of all grade classes
3. **Model Selection**: Different models suit different use cases:
   - **Interpretability**: Logistic Regression and Decision Tree provide clear decision boundaries
   - **Performance**: Ensemble methods (Random Forest, XGBoost) typically achieve higher accuracy
   - **Speed**: Logistic Regression and Naive Bayes are computationally efficient
4. **Evaluation Metrics**: Multi-class classification uses macro-averaging to give equal weight to all classes, ensuring balanced evaluation across grade categories

## Author

Devata Sai Harshith
BITS Pilani - M.Tech (DSE)

## License

This project is for educational purposes as part of BITS Pilani ML Assignment 2.