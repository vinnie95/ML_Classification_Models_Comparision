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

# Load models function
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "K-Nearest Neighbors": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl"
    }
    
    for name, file in model_files.items():
        try:
            models[name] = pickle.load(open(file, 'rb'))
        except:
            st.error(f"Could not load {name} model")
    
    try:
        scaler = pickle.load(open('scaler.pkl', 'rb'))
    except:
        scaler = None
        
    return models, scaler

# Load comparison results
@st.cache_data
def load_comparison():
    try:
        return pd.read_csv('model_comparison.csv')
    except:
        return None

# Main app
def main():
    # Load models
    models, scaler = load_models()
    comparison_df = load_comparison()
    
    # Sidebar - Model Selection
    st.sidebar.subheader("📊 Select Model")
    selected_model = st.sidebar.selectbox(
        "Choose a classification model:",
        list(models.keys())
    )
    
    st.sidebar.info(model_info[selected_model])
    
    # Display overall comparison
    st.header("📈 Model Performance Comparison")
    
    if comparison_df is not None:
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
            x = np.arange(len(comparison_df))
            width = 0.2
            
            for i, metric in enumerate(metrics):
                ax.bar(x + i*width, comparison_df[metric], width, label=metric)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(comparison_df['ML Model Name'], rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(comparison_df['ML Model Name'], comparison_df['Accuracy'], color='skyblue')
            ax.set_xlabel('Accuracy Score')
            ax.set_title('Model Accuracy Comparison')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
    
    st.markdown("---")
    
    # File upload section
    st.header("📤 Upload Test Data")
    st.markdown("Upload a CSV file with mobile phone specifications (without target column)")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read data
            test_data = pd.read_csv(uploaded_file)
            
            st.success(f"✓ File uploaded successfully! Shape: {test_data.shape}")
            
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
                X_test_processed = scaler.transform(X_test)
            else:
                X_test_processed = X_test
            
            # Make predictions
            with st.spinner('Making predictions...'):
                y_pred = model.predict(X_test_processed)
                
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test_processed)
                else:
                    y_pred_proba = None
            
            st.success("✓ Predictions completed!")
            
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
                
                # Classification Report
                st.subheader("📋 Classification Report")
                report = classification_report(y_true, y_pred, 
                                              target_names=['Low', 'Medium', 'High', 'Very High'],
                                              output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure your CSV has the correct format and column names")
    
    else:
        st.info("👆 Please upload a CSV file to get started")
        
        # Show example format
        with st.expander("📝 Expected CSV Format"):
            st.markdown("""
            Your CSV should contain the following 20 features:
            - battery_power, blue, clock_speed, dual_sim, fc, four_g
            - int_memory, m_dep, mobile_wt, n_cores, pc, px_height
            - px_width, ram, sc_h, sc_w, talk_time, three_g
            - touch_screen, wifi
            
            Optional: price_range (0-3) if you want evaluation metrics
            """)

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

if __name__ == "__main__":
    main()