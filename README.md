# 📱 Mobile Price Classification - ML Assignment 2

## 🎯 Problem Statement

The objective of this project is to predict the price range of mobile phones based on their technical specifications and features. In the mobile phone market, pricing is a critical factor influenced by various hardware and software features. This is a **multi-class classification problem** where mobile phones are categorized into four price ranges:
- **0**: Low Cost
- **1**: Medium Cost  
- **2**: High Cost
- **3**: Very High Cost

The goal is to build and compare multiple machine learning classification models to accurately predict the price range of a mobile phone given its specifications, helping both consumers make informed purchasing decisions and manufacturers establish competitive pricing strategies.

---

## 📊 Dataset Description

**Source**: [Kaggle Mobile Price Classification Dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)

### Dataset Characteristics
- **Number of Instances**: 2000
- **Number of Features**: 20
- **Target Variable**: `price_range` (4 classes: 0, 1, 2, 3)
- **Missing Values**: None
- **Data Type**: Numerical and Binary features

### Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| `battery_power` | Continuous | Total energy a battery can store in one time (mAh) |
| `blue` | Binary | Has Bluetooth or not (0/1) |
| `clock_speed` | Continuous | Speed at which microprocessor executes instructions |
| `dual_sim` | Binary | Has dual SIM support or not (0/1) |
| `fc` | Continuous | Front Camera mega pixels |
| `four_g` | Binary | Has 4G or not (0/1) |
| `int_memory` | Continuous | Internal Memory in Gigabytes |
| `m_dep` | Continuous | Mobile Depth in cm |
| `mobile_wt` | Continuous | Weight of mobile phone |
| `n_cores` | Continuous | Number of cores of processor |
| `pc` | Continuous | Primary Camera mega pixels |
| `px_height` | Continuous | Pixel Resolution Height |
| `px_width` | Continuous | Pixel Resolution Width |
| `ram` | Continuous | Random Access Memory in Megabytes |
| `sc_h` | Continuous | Screen Height of mobile in cm |
| `sc_w` | Continuous | Screen Width of mobile in cm |
| `talk_time` | Continuous | Longest time that a single battery charge will last |
| `three_g` | Binary | Has 3G or not (0/1) |
| `touch_screen` | Binary | Has touch screen or not (0/1) |
| `wifi` | Binary | Has wifi or not (0/1) |

### Target Variable
- **price_range**: Classification target with 4 classes
  - 0: Low Cost
  - 1: Medium Cost
  - 2: High Cost
  - 3: Very High Cost

---

## 🤖 Models Used

Six different machine learning classification models were implemented and evaluated:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|-------|------|
| Logistic Regression | 0.9517 | 0.9970 | 0.9517 | 0.9517 | 0.9516 | 0.9356 |
| Decision Tree | 0.8286 | 0.8888 | 0.8335 | 0.8286 | 0.8299 | 0.7719 |
| K-Nearest Neighbors | 0.4958 | 0.7359 | 0.5046 | 0.4958 | 0.4939 | 0.3276 |
| Naive Bayes | 0.7737 | 0.9398 | 0.7777 | 0.7737 | 0.7752 | 0.6983 |
| Random Forest (Ensemble) | 0.8502 | 0.9725 | 0.8493 | 0.8502 | 0.8493 | 0.8004 |
| XGBoost (Ensemble) | 0.8852 | 0.9865 | 0.8847 | 0.8852 | 0.8849 | 0.8468 |

---

## 📈 Model Performance Observations

### Logistic Regression
Logistic Regression achieved the highest accuracy of 95.17% with an outstanding AUC score of 0.997, demonstrating exceptional performance for this multi-class classification problem. The model shows perfectly balanced precision and recall (95.17%), with a strong MCC score of 0.9356 indicating excellent classification quality. Despite being a linear model, it excels on this dataset, suggesting strong linear separability in the feature space after preprocessing. Its fast training time, interpretability, and superior performance make it the best model for this task.

### Decision Tree
Decision Tree achieved 82.86% accuracy with an AUC of 0.8888, showing moderate performance. The balanced precision (83.35%) and recall (82.86%) indicate consistent predictions across price ranges, with an MCC of 0.7719 confirming reasonable classification quality. While decision trees excel at capturing non-linear patterns and provide excellent interpretability, this single tree underperforms compared to linear and ensemble methods, suggesting potential overfitting or suboptimal hyperparameters. It ranks fourth overall among the six models tested.

### K-Nearest Neighbors
KNN demonstrated the weakest performance with only 49.58% accuracy, barely better than random guessing for a 4-class problem. The low AUC (0.7359), precision (50.46%), and recall (49.58%) indicate significant classification struggles. The very poor MCC score of 0.3276 confirms weak discrimination ability. This suggests the feature space lacks clear distance-based patterns, or the default hyperparameters are inappropriate. The curse of dimensionality with 20 features severely impacts KNN's effectiveness, making it unsuitable for this dataset without extensive tuning.

### Naive Bayes
Naive Bayes achieved 77.37% accuracy with a strong AUC of 0.9398, showing the largest gap between accuracy and AUC among all models. The balanced precision (77.77%) and recall (77.37%) with MCC of 0.6983 indicate moderate classification quality. The high AUC suggests well-calibrated probability estimates despite lower accuracy. The feature independence assumption clearly limits performance, as mobile specifications have inherent correlations (e.g., RAM and processor cores). However, its simplicity and speed make it useful for baseline comparisons.

### Random Forest (Ensemble)
Random Forest delivered 85.02% accuracy with an excellent AUC of 0.9725, demonstrating strong ensemble performance. The well-balanced precision (84.93%) and recall (85.02%) with MCC of 0.8004 indicate robust classification across all price ranges. By combining multiple decision trees, it reduces overfitting while maintaining good predictive power. It significantly outperforms the single decision tree (82.86%) and Naive Bayes (77.37%), validating the ensemble approach. However, it falls short of both Logistic Regression and XGBoost in overall performance.

### XGBoost (Ensemble)
XGBoost achieved 88.52% accuracy with the second-highest AUC of 0.9865, demonstrating strong gradient boosting performance. The balanced precision (88.47%) and recall (88.52%) with the highest MCC of 0.8468 show excellent discrimination ability. While XGBoost typically excels in most scenarios and performs better than Random Forest (85.02%), it surprisingly falls short of Logistic Regression's 95.17% accuracy. This suggests the dataset exhibits strong linear patterns that gradient boosting's complexity cannot improve upon, though it remains a solid second-choice model.

### Overall Insights
1. **Logistic Regression Dominates**: Achieves highest accuracy (95.17%) and best AUC (0.997), indicating strong linear separability in the preprocessed feature space.

2. **Ensemble Methods Strong**: XGBoost (88.52%) and Random Forest (85.02%) perform well but cannot surpass the linear model, suggesting limited benefit from complex non-linear interactions.

3. **KNN Fails Completely**: With only 49.58% accuracy, KNN is unsuitable for this dataset due to curse of dimensionality and poor distance-based patterns.

4. **Linear Patterns Dominant**: The success of Logistic Regression over gradient boosting indicates that feature engineering and scaling created strong linear boundaries between price ranges.

5. **Model Ranking**: Logistic Regression (95.17%) > XGBoost (88.52%) > Random Forest (85.02%) > Decision Tree (82.86%) > Naive Bayes (77.37%) > KNN (49.58%).

---

## 🚀 Streamlit App Features

The deployed web application includes:

✅ **CSV File Upload**: Upload test data in CSV format  
✅ **Model Selection**: Choose from 6 different ML models via dropdown  
✅ **Evaluation Metrics**: Display of Accuracy, AUC, Precision, Recall, F1, and MCC  
✅ **Confusion Matrix**: Heatmap visualization of prediction results  
✅ **Classification Report**: Detailed per-class performance metrics  
✅ **Prediction Export**: Download predictions as CSV  
✅ **Data Preview**: View uploaded data with predictions and probabilities  
✅ **Test Data Download**: Download test.csv directly from GitHub  

---

## 📁 Repository Structure

```
mobile-price-classification/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── model/                          # Model directory
│   ├── artifacts/                  # Saved models pkl files
│   │   ├── logistic_regression.pkl
│   │   ├── decision_tree.pkl
│   │   ├── knn.pkl
│   │   ├── naive_bayes.pkl
│   │   ├── random_forest.pkl
│   │   ├── xgboost.pkl
│   │   └── scaler.pkl
│   │
│   ├── submission_csv/             # Training results
│   │   ├── logistic_regression_results.csv
│   │   ├── decision_tree_results.csv
│   │   ├── knn_results.csv
│   │   ├── naive_bayes_results.csv
│   │   ├── random_forest_results.csv
│   │   └── xgboost_results.csv
│   │
│   ├── 1_logistic_regression.ipynb # Training notebooks
│   ├── 2_decision_tree.ipynb
│   ├── 3_knn.ipynb
│   ├── 4_naive_bayes.ipynb
│   ├── 5_random_forest.ipynb
│   └── 6_xgboost.ipynb
│
├── data/                           # Dataset files
│   ├── train.csv
│   └── test.csv
│
└── screenshots/                    # BITS Lab execution proof
    └── bits_lab_screenshot.png
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/vinnie95/ML_Classification_Models_Comparision.git
cd ML_Classification_Models_Comparision
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**
```bash
streamlit run app.py
```

4. **Access the app**
Open your browser and navigate to `http://localhost:8501`

---

## 🌐 Deployment

### Streamlit Community Cloud

1. Push your code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New App"
5. Select your repository, branch, and `app.py`
6. Click "Deploy"

**Live App**: https://2025aa05437mlassignment2.streamlit.app/

---

## 📝 Usage Guide

### Using the Web App

1. **Access the deployed application** at the provided Streamlit link
2. **Select a model** from the dropdown menu in the sidebar
3. **Upload your test data** (CSV file with mobile specifications)
4. **View predictions** with probability scores for each price range
5. **Analyze performance** through confusion matrix and classification report
6. **Download results** as a CSV file for further analysis

### CSV Format

Your test CSV should contain the following 20 columns:
```
battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, 
mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, 
three_g, touch_screen, wifi
```

Optionally include `price_range` (0-3) for model evaluation.

---

## 📊 Model Training

The models were trained using the following approach:

1. **Data Preprocessing**
   - Handled missing values (none found)
   - Feature scaling for distance-based models (KNN, Logistic Regression)
   - Train-test split (80-20)

2. **Model Training**
   - Implemented 6 classification algorithms
   - Hyperparameter tuning using GridSearchCV
   - Cross-validation for robust evaluation

3. **Evaluation**
   - Multiple metrics: Accuracy, AUC, Precision, Recall, F1, MCC
   - Confusion matrix analysis
   - Classification reports for each model

4. **Model Persistence**
   - Saved trained models as .pkl files
   - Saved scaler for consistent preprocessing

---

## 🔍 Key Findings

- **Logistic Regression** achieved the highest accuracy (95.17%) and is recommended for production use
- **Linear separability** in the feature space makes simple models more effective than complex ensembles
- **RAM** and **battery_power** were identified as the most important features
- **KNN** performed poorly (49.58%) due to curse of dimensionality with 20 features
- The dataset exhibits strong linear patterns after proper feature scaling

---

## 🎓 Assignment Details

- **Course**: M.Tech (AIML/DSE) - Machine Learning
- **Assignment**: Assignment 2
- **Institution**: BITS Pilani WILP
- **Executed on**: BITS Virtual Lab
- **Submission Deadline**: 15-Feb-2026

---

## 📧 Contact

For questions or issues, please contact:
- **Email**: your.email@example.com
- **GitHub**: [@your-username](https://github.com/your-username)

---

## 📄 License

This project is created for educational purposes as part of BITS Pilani WILP coursework.

---

## 🙏 Acknowledgments

- Dataset: [Kaggle - Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification)
- BITS Pilani WILP for providing the virtual lab environment
- Streamlit for the amazing web app framework

---

**⭐ If you found this project helpful, please consider giving it a star!**
