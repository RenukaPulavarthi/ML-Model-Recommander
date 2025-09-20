import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

st.title("ML Model Comparison App")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.write("Preview of dataset:")
    st.dataframe(df.head())
    
    target_column = st.selectbox("Select the target column", df.columns)
    if st.button("Run Comparison"):
        # Preprocess features/target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        for col in X.select_dtypes(include=['object']):
            X[col] = LabelEncoder().fit_transform(X[col])
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=200),
            "KNN": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Naive Bayes": GaussianNB(),
            "Neural Net (MLP)": MLPClassifier(max_iter=300, random_state=42),
            "SVC": SVC(random_state=42, probability=True)
        }
        if xgb_available:
            models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        
        st.subheader("Cross-Validation Accuracy (5-fold):")
        cv_results = {}
        best_score = 0
        best_model = None
        best_name = None
        
        for name, model in models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
            score_mean = np.mean(scores)
            cv_results[name] = score_mean
            st.write(f"{name}: {score_mean:.4f} (+/- {np.std(scores):.4f})")
            if score_mean > best_score:
                best_score = score_mean
                best_model = pipeline
                best_name = name
        
        # Train the best model and report details on test set
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        st.subheader(f"Results for Best Model: {best_name}")
        st.text(f"Test Set Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))
else:
    st.info("Please upload a CSV or Excel file to begin.")