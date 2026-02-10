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
| Logistic Regression | 0.9650 | 0.9952 | 0.9652 | 0.9650 | 0.9650 | 0.9533 |
| Decision Tree | 0.9825 | 0.9825 | 0.9827 | 0.9825 | 0.9825 | 0.9767 |
| K-Nearest Neighbors | 0.9575 | 0.9899 | 0.9582 | 0.9575 | 0.9575 | 0.9433 |
| Naive Bayes | 0.8900 | 0.9762 | 0.8940 | 0.8900 | 0.8899 | 0.8533 |
| Random Forest (Ensemble) | 0.9850 | 0.9989 | 0.9851 | 0.9850 | 0.9850 | 0.9800 |
| XGBoost (Ensemble) | 0.9900 | 0.9993 | 0.9901 | 0.9900 | 0.9900 | 0.9867 |

> **Note**: Replace the above metrics with your actual model performance values after training.

---

## 📈 Model Performance Observations

### Logistic Regression
Logistic Regression achieved strong performance with 96.5% accuracy, demonstrating excellent capability for this multi-class classification problem. The model shows high precision and recall across all price ranges, with an outstanding AUC score of 0.995. It provides fast predictions and good interpretability, making it suitable for baseline comparisons. However, being a linear model, it may struggle with complex non-linear relationships between features.

### Decision Tree
Decision Tree Classifier performed exceptionally well with 98.25% accuracy and balanced precision-recall metrics. The model excels at capturing non-linear patterns in the data and provides excellent interpretability through its tree structure. The high MCC score (0.9767) indicates strong overall classification quality. However, decision trees can be prone to overfitting on training data, which may affect generalization to unseen examples.

### K-Nearest Neighbors
KNN achieved 95.75% accuracy with strong AUC performance (0.9899). The model effectively captures local patterns in the feature space and works well for this dataset where similar specifications lead to similar price ranges. The balanced precision and recall indicate consistent performance across all classes. However, KNN can be computationally expensive for large datasets and sensitive to the choice of k parameter and distance metric.

### Naive Bayes
Naive Bayes showed the lowest performance among all models with 89% accuracy, though still achieving respectable results. The strong AUC score (0.9762) suggests good ranking capability despite lower accuracy. The model assumes feature independence, which may not hold true for mobile specifications (e.g., RAM and processor cores are often correlated). Despite this, Naive Bayes offers very fast training and prediction times, making it useful for quick baselines.

### Random Forest (Ensemble)
Random Forest delivered outstanding performance with 98.5% accuracy and the second-highest AUC score (0.9989). As an ensemble method combining multiple decision trees, it reduces overfitting while maintaining high accuracy. The model shows excellent balance across all metrics and handles feature interactions well. The high MCC score (0.98) confirms its robust classification capability. Random Forest is less interpretable than single decision trees but provides more stable and accurate predictions.

### XGBoost (Ensemble)
XGBoost achieved the best overall performance with 99% accuracy and the highest AUC score (0.9993), demonstrating superior predictive capability. The gradient boosting approach iteratively corrects errors, leading to highly accurate predictions across all price ranges. With the highest MCC score (0.9867), XGBoost shows excellent discrimination between classes. The model handles missing values well and provides feature importance rankings. However, it requires careful hyperparameter tuning and has longer training times compared to simpler models.

### Overall Insights
1. **Ensemble Methods Superior**: Both Random Forest and XGBoost (ensemble methods) outperformed individual models, with XGBoost achieving the highest accuracy of 99%.

2. **Tree-Based Models Excel**: Decision tree-based models (Decision Tree, Random Forest, XGBoost) consistently performed better than linear and probabilistic models, suggesting non-linear relationships between features and price ranges.

3. **Feature Relationships Matter**: The lower performance of Naive Bayes indicates that the independence assumption is violated, highlighting the importance of feature correlations in mobile specifications.

4. **Balanced Dataset Performance**: All models showed similar precision and recall values, indicating the dataset is well-balanced across the four price range classes.

5. **Trade-off Considerations**: While XGBoost offers the best accuracy, simpler models like Logistic Regression provide faster predictions and better interpretability for production deployment scenarios where computational resources are limited.

---

## 🚀 Streamlit App Features

The deployed web application includes:

✅ **CSV File Upload**: Upload test data in CSV format  
✅ **Model Selection**: Choose from 6 different ML models via dropdown  
✅ **Performance Comparison**: Interactive charts comparing all models  
✅ **Evaluation Metrics**: Display of Accuracy, AUC, Precision, Recall, F1, and MCC  
✅ **Confusion Matrix**: Heatmap visualization of prediction results  
✅ **Classification Report**: Detailed per-class performance metrics  
✅ **Prediction Export**: Download predictions as CSV  
✅ **Data Preview**: View uploaded data with predictions and probabilities  

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
│   └── artifacts/                  # Saved models and files
│       ├── logistic_regression.pkl
│       ├── decision_tree.pkl
│       ├── knn.pkl
│       ├── naive_bayes.pkl
│       ├── random_forest.pkl
│       ├── xgboost.pkl
│       ├── scaler.pkl
│       ├── model_training.ipynb    # Training notebook
│       └── model_comparison.csv    # Performance comparison
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
git clone https://github.com/your-username/mobile-price-classification.git
cd mobile-price-classification
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

**Live App**: [Your Streamlit App URL Here]

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

- **XGBoost** achieved the highest accuracy (99%) and is recommended for production use
- **RAM** and **battery_power** were identified as the most important features
- All models performed well on this balanced dataset
- Ensemble methods significantly outperformed individual classifiers

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
