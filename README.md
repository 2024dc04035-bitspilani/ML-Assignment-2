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

## Author

Devata Sai Harshith
BITS Pilani - M.Tech (DSE)

## License

This project is for educational purposes as part of BITS Pilani ML Assignment 2