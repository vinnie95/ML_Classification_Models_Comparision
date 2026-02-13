import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                             accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
from sklearn.preprocessing import label_binarize
import os

# Page configuration
st.set_page_config(
    page_title="Mobile Price Classifier",
    page_icon="📱",
    layout="wide"
)

# Title and description
st.title("📱 Mobile Price Range Classification")
st.markdown("""
This application predicts the price range of mobile phones based on their specifications.
Upload a CSV file with test data to see predictions from multiple ML models.
""")

# Sidebar
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# Model descriptions
model_info = {
    "Logistic Regression": "Linear model for multi-class classification",
    "Decision Tree": "Tree-based model using hierarchical decisions",
    "K-Nearest Neighbors": "Instance-based learning algorithm",
    "Naive Bayes": "Probabilistic classifier based on Bayes' theorem",
    "Random Forest": "Ensemble of decision trees",
    "XGBoost": "Gradient boosting ensemble model"
}

# Load models function with better error handling
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "Logistic Regression": "model/artifacts/logistic_regression.pkl",
        "Decision Tree": "model/artifacts/decision_tree.pkl",
        "K-Nearest Neighbors": "model/artifacts/knn.pkl",
        "Naive Bayes": "model/artifacts/naive_bayes.pkl",
        "Random Forest": "model/artifacts/random_forest.pkl",
        "XGBoost": "model/artifacts/xgboost.pkl"
    }
    
    missing_models = []
    
    for name, file in model_files.items():
        if os.path.exists(file):
            try:
                with open(file, 'rb') as f:
                    models[name] = pickle.load(f)
            except Exception as e:
                st.warning(f"⚠️ Error loading {name} model: {str(e)}")
                missing_models.append(name)
        else:
            missing_models.append(name)
    
    # Load scaler
    scaler = None
    if os.path.exists('model/artifacts/scaler.pkl'):
        try:
            with open('model/artifacts/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
        except Exception as e:
            st.warning(f"⚠️ Error loading scaler: {str(e)}")
    
    return models, scaler, missing_models

# Main app
def main():
    # Load models
    models, scaler, missing_models = load_models()
    
    # Check if any models are loaded
    if not models:
        st.error("❌ **No model files found!**")
        st.info("""
        **To deploy this app, you need to include the following files in your repository:**
        
        1. **Model Files (.pkl)** in `model/artifacts/` directory:
           - model/artifacts/logistic_regression.pkl
           - model/artifacts/decision_tree.pkl
           - model/artifacts/knn.pkl
           - model/artifacts/naive_bayes.pkl
           - model/artifacts/random_forest.pkl
           - model/artifacts/xgboost.pkl
           - model/artifacts/scaler.pkl
        
        2. **Required**: requirements.txt (see sidebar for contents)
        
        **Steps to fix:**
        1. Train your models and save them as .pkl files
        2. Create the directory structure: model/artifacts/
        3. Add all .pkl files to model/artifacts/ folder
        4. Add all files to your GitHub repository
        5. Create a requirements.txt file
        6. Redeploy your app
        """)
        
        # Show requirements.txt content
        with st.expander("📄 Required requirements.txt content"):
            st.code("""streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost""")
        
        return
    
    # Show warning if some models are missing
    if missing_models:
        st.warning(f"⚠️ **Some models could not be loaded**: {', '.join(missing_models)}")
    
    # Sidebar - Model Selection
    st.sidebar.subheader("📊 Select Model")
    
    if models:
        selected_model = st.sidebar.selectbox(
            "Choose a classification model:",
            list(models.keys())
        )
        
        st.sidebar.info(model_info[selected_model])
    else:
        st.sidebar.error("No models available")
        return
    
    # File upload section
    st.header("📤 Upload Test Data")
    st.markdown("Upload a CSV file with mobile phone specifications (without target column)")
    
    # Add download link for sample test data
    st.markdown("---")
    st.subheader("📥 Don't have test data?")
    
    st.markdown("""
    Download the test dataset from the GitHub repository:
    """)
    
    # Fetch the CSV from GitHub and provide as download button
    try:
        import requests
        github_csv_url = "https://raw.githubusercontent.com/vinnie95/ML_Classification_Models_Comparision/main/data/test.csv"
        response = requests.get(github_csv_url)
        
        if response.status_code == 200:
            st.download_button(
                label="📂 Download test.csv from GitHub",
                data=response.content,
                file_name='test.csv',
                mime='text/csv',
                help="Click to download the test dataset"
            )
        else:
            st.warning("Could not fetch the file from GitHub. Please check if the repository is public.")
            st.markdown(f"**Alternative:** [View on GitHub](https://github.com/vinnie95/ML_Classification_Models_Comparision/blob/main/data/test.csv)")
    except Exception as e:
        st.warning(f"Could not fetch the file: {str(e)}")
        st.markdown(f"**Alternative:** [View on GitHub](https://github.com/vinnie95/ML_Classification_Models_Comparision/blob/main/data/test.csv)")
    
    
    st.info("💡 **Tip**: The test CSV should contain 20 feature columns. Optionally include 'price_range' column for model evaluation.")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read data
            test_data = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Shape: {test_data.shape}")
            
            # Show data preview
            with st.expander("👁️ View Uploaded Data"):
                st.dataframe(test_data.head(10))
            
            # Check if target column exists
            has_target = 'price_range' in test_data.columns
            
            if has_target:
                y_true = test_data['price_range']
                X_test = test_data.drop('price_range', axis=1)
                st.info("✓ Target column detected - will show evaluation metrics")
            else:
                X_test = test_data
                y_true = None
                st.warning("⚠️ No target column found - will only show predictions")
            
            # Prepare data
            model = models[selected_model]
            
            # Scale if needed
            if selected_model in ["Logistic Regression", "K-Nearest Neighbors", "Naive Bayes"]:
                if scaler is not None:
                    X_test_processed = scaler.transform(X_test)
                else:
                    st.warning("⚠️ Scaler not found. Using unscaled data - predictions may be inaccurate.")
                    X_test_processed = X_test
            else:
                X_test_processed = X_test
            
            # Make predictions
            with st.spinner('Making predictions...'):
                try:
                    y_pred = model.predict(X_test_processed)
                    
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test_processed)
                    else:
                        y_pred_proba = None
                except Exception as e:
                    st.error(f"❌ Error making predictions: {str(e)}")
                    st.info("Please ensure your data has the correct features and format.")
                    return
            
            st.success("✅ Predictions completed!")
            
            # Display predictions
            st.header(f"🎯 Predictions using {selected_model}")
            
            # Create results dataframe
            results_df = test_data.copy()
            results_df['Predicted_Price_Range'] = y_pred
            
            price_labels = {0: 'Low Cost', 1: 'Medium Cost', 2: 'High Cost', 3: 'Very High Cost'}
            results_df['Predicted_Label'] = results_df['Predicted_Price_Range'].map(price_labels)
            
            if y_pred_proba is not None:
                for i in range(4):
                    results_df[f'Probability_Class_{i}'] = y_pred_proba[:, i]
            
            st.dataframe(results_df, use_container_width=True)
            
            # Download predictions
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Predictions as CSV",
                data=csv,
                file_name=f'predictions_{selected_model.replace(" ", "_")}.csv',
                mime='text/csv'
            )
            
            # If target exists, show evaluation metrics
            if has_target:
                st.markdown("---")
                st.header("📊 Evaluation Metrics")
                
                try:
                    # Calculate metrics
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                    mcc = matthews_corrcoef(y_true, y_pred)
                    
                    # AUC calculation
                    if y_pred_proba is not None:
                        y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
                        auc = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr', average='weighted')
                    else:
                        auc = 0.0
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.4f}")
                        st.metric("Precision", f"{precision:.4f}")
                    
                    with col2:
                        st.metric("Recall", f"{recall:.4f}")
                        st.metric("F1 Score", f"{f1:.4f}")
                    
                    with col3:
                        st.metric("AUC Score", f"{auc:.4f}")
                        st.metric("MCC Score", f"{mcc:.4f}")
                    
                    # Confusion Matrix
                    st.subheader("🔢 Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=['Low', 'Medium', 'High', 'Very High'],
                               yticklabels=['Low', 'Medium', 'High', 'Very High'])
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title(f'Confusion Matrix - {selected_model}')
                    st.pyplot(fig)
                    plt.close()
                    
                    # Classification Report
                    st.subheader("📋 Classification Report")
                    report = classification_report(y_true, y_pred, 
                                                  target_names=['Low', 'Medium', 'High', 'Very High'],
                                                  output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error calculating metrics: {str(e)}")
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV has the correct format and column names")
    
    else:
        st.info("👆 Please upload a CSV file to get started")
        
        # Show example format
        with st.expander("📋 Expected CSV Format"):
            st.markdown("""
            Your CSV should contain the following 20 features:
            - **battery_power**: Total energy a battery can store in mAh
            - **blue**: Has bluetooth or not (0/1)
            - **clock_speed**: Speed at which microprocessor executes instructions
            - **dual_sim**: Has dual sim support or not (0/1)
            - **fc**: Front Camera mega pixels
            - **four_g**: Has 4G or not (0/1)
            - **int_memory**: Internal Memory in GB
            - **m_dep**: Mobile Depth in cm
            - **mobile_wt**: Weight of mobile phone
            - **n_cores**: Number of cores of processor
            - **pc**: Primary Camera mega pixels
            - **px_height**: Pixel Resolution Height
            - **px_width**: Pixel Resolution Width
            - **ram**: Random Access Memory in MB
            - **sc_h**: Screen Height of mobile in cm
            - **sc_w**: Screen Width of mobile in cm
            - **talk_time**: Longest time that battery will last
            - **three_g**: Has 3G or not (0/1)
            - **touch_screen**: Has touch screen or not (0/1)
            - **wifi**: Has wifi or not (0/1)
            
            **Optional**: price_range (0-3) if you want evaluation metrics
            """)
            
            # Sample data
            st.markdown("**Sample Data Format:**")
            sample_data = pd.DataFrame({
                'battery_power': [842, 1021],
                'blue': [0, 1],
                'clock_speed': [2.2, 0.5],
                'dual_sim': [0, 1],
                'fc': [1, 0],
                'four_g': [0, 1],
                'int_memory': [7, 53],
                'm_dep': [0.6, 0.7],
                'mobile_wt': [188, 136],
                'n_cores': [2, 3],
                'pc': [2, 6],
                'px_height': [20, 905],
                'px_width': [756, 1988],
                'ram': [2549, 2631],
                'sc_h': [9, 17],
                'sc_w': [7, 3],
                'talk_time': [19, 7],
                'three_g': [0, 1],
                'touch_screen': [0, 1],
                'wifi': [1, 0]
            })
            st.dataframe(sample_data)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About")
st.sidebar.info("""
**Mobile Price Classification**

This app uses 6 different ML models to predict mobile phone price ranges based on specifications.

**Target Classes:**
- 0: Low Cost
- 1: Medium Cost  
- 2: High Cost
- 3: Very High Cost
""")

# Deployment help
with st.sidebar.expander("🚀 Deployment Guide"):
    st.markdown("""
    **Files needed for deployment:**
    1. app.py (this file)
    2. requirements.txt
    3. Model files (.pkl)
    
    **Common Issues:**
    - Missing model files
    - Incorrect requirements.txt
    - Large file sizes (>100MB)
    
    Use Git LFS for large files!
    """)

if __name__ == "__main__":
    main()
