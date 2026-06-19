
## STREAMLIT APP DEPLOYMENT
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from pathlib import Path

# Load the pre-trained model
model = joblib.load('churn_list.sav')

# Load and preprocess the data
@st.cache_data
def load_data():
    df = pd.read_csv('Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    df['Churn'] = df['Churn'].replace({'Yes': 1, 'No': 0})
    return df

df = load_data()

# Define categorical and numerical features
cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
            'PaperlessBilling', 'PaymentMethod']
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Function to preprocess input data
def preprocess_input(input_data):
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    scaler = StandardScaler()

    # Fit the encoder and scaler on the original data
    encoder.fit(df[cat_cols])
    scaler.fit(df[num_cols])

    # Preprocess categorical features
    cat_df = pd.DataFrame(encoder.transform(input_data[cat_cols]), columns=encoder.get_feature_names_out(cat_cols))

    # Preprocess numerical features
    num_df = pd.DataFrame(scaler.transform(input_data[num_cols]), columns=num_cols)
    
    # Concatenate the processed features
    processed_df = pd.concat([cat_df, num_df], axis=1)
    return processed_df

# Load custom CSS
def load_css():
    css_file = Path("static/css/styles.css")
    if css_file.exists():
        with open(css_file, "r") as f:
            css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

load_css()

# Create Streamlit form
st.title('Telecom Customer Churn Prediction')


st.markdown("""
<div class="card card-custom mb-4"> 
  <img src="https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1170&q=80" class="card-img-top" alt="Woman holding magnetic card">
  <div class="card-body">
    <h5 class="card-title">Welcome to the Churn Prediction App</h5>
    <p class="card-text">Use the form below to enter customer details and predict the probability of churn. Analyze data effectively with our model.</p>
  </div>
</div>
""", unsafe_allow_html=True)



with st.form(key='input_form', clear_on_submit=True):
    st.header('Enter Customer Information')

    # User input fields
    gender = st.selectbox('Gender', options=['Male', 'Female'])
    partner = st.selectbox('Partner', options=['Yes', 'No'])
    dependents = st.selectbox('Dependents', options=['Yes', 'No'])
    phone_service = st.selectbox('PhoneService', options=['Yes', 'No'])
    multiple_lines = st.selectbox('MultipleLines', options=['No phone service', 'No', 'Yes'])
    internet_service = st.selectbox('InternetService', options=['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox('OnlineSecurity', options=['No', 'Yes'])
    online_backup = st.selectbox('OnlineBackup', options=['No', 'Yes'])
    device_protection = st.selectbox('DeviceProtection', options=['No', 'Yes'])
    tech_support = st.selectbox('TechSupport', options=['No', 'Yes'])
    streaming_tv = st.selectbox('StreamingTV', options=['No', 'Yes'])
    streaming_movies = st.selectbox('StreamingMovies', options=['No', 'Yes'])
    contract = st.selectbox('Contract', options=['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox('PaperlessBilling', options=['Yes', 'No'])
    payment_method = st.selectbox('PaymentMethod', options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    
    tenure = st.number_input('Tenure', min_value=0)
    monthly_charges = st.number_input('MonthlyCharges', min_value=0.0)
    total_charges = st.number_input('TotalCharges', min_value=0.0)

    submit_button = st.form_submit_button(label='Predict Churn')

    if submit_button:
        # Prepare input data
        input_data = {
            'gender': gender,
            'Partner': partner,
            'Dependents': dependents,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess the input data
        processed_input = preprocess_input(input_df)

        # Make prediction
        churn_prob = model.predict_proba(processed_input)[:, 1][0]

        # Display results
        st.write(f'Probability of Churn: {churn_prob:.2f}')

        # Visualization
        fig = px.bar(x=['Churn Probability'], y=[churn_prob], title='Churn Probability')
        st.plotly_chart(fig)

        # Calculate confusion matrix and ROC curve
        # Preprocess the full dataset
        processed_df = preprocess_input(df)
        
        # Make predictions on the full dataset
        predictions = model.predict(processed_df)
        probabilities = model.predict_proba(processed_df)[:, 1]

        # Confusion Matrix
        conf_matrix = confusion_matrix(df['Churn'], predictions)
        fig_cm = ff.create_annotated_heatmap(z=conf_matrix, x=['Predicted No', 'Predicted Yes'], y=['Actual No', 'Actual Yes'])
        st.plotly_chart(fig_cm, use_container_width=True)

        # ROC Curve
        fpr, tpr, _ = roc_curve(df['Churn'], probabilities)
        roc_auc = roc_auc_score(df['Churn'], probabilities)
        fig_roc = px.line(x=fpr, y=tpr, title=f'ROC Curve (AUC = {roc_auc:.2f})', labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
        st.plotly_chart(fig_roc)

        # SHAP Values
        # Fit SHAP explainer
        explainer = shap.Explainer(model, processed_df)
        shap_values = explainer(processed_df)

        # SHAP Summary Plot
        st.header('SHAP Summary Plot')
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, processed_df, plot_type="bar", show=False)
        st.pyplot(fig)

        # Additional SHAP visualizations
        st.header("Additional SHAP Visualizations")
        
        # Mean SHAP Plot (bar plot)
        st.subheader("Mean SHAP Plot")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_values)
        st.pyplot(fig)

        # Waterfall SHAP Plot (explains a specific instance, e.g., first customer)
        st.subheader("Waterfall SHAP Plot for First Prediction")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0])
        st.pyplot(fig)

        # SHAP Scatter Plot (for specific feature, e.g., 'tenure')
        #st.subheader("SHAP Scatter Plot for Tenure")
        #fig, ax = plt.subplots()
        #shap.plots.scatter(shap_values[:, "tenure"])
        #st.pyplot(fig)

        # SHAP Scatter Plot (for specific feature, e.g., 'tenure')
        st.subheader("SHAP Scatter Plot for Tenure")
        if "tenure" in shap_values.feature_names:
            fig, ax = plt.subplots()
            shap.plots.scatter(shap_values[:, "tenure"], ax=ax)
            st.pyplot(fig)
        else:
            st.error("Feature 'tenure' not found in SHAP values.")

        # SHAP Force Plot (for first 100 predictions)
        st.subheader("SHAP Force Plot for First 100 Predictions")

        # Convert SHAP values to an Explanation object
        shap_values_explanation = shap.Explanation(shap_values.values[:100], base_values=explainer.expected_value, data=processed_df[:100])

        # Generate the SHAP force plot
        shap_force_plot = shap.force_plot(explainer.expected_value, shap_values.values[:100], processed_df[:100])

        # Generate the SHAP force plot as an HTML object
        shap_html = f"<head>{shap.getjs()}</head><body>{shap_force_plot.html()}</body>"

       # Render the HTML object in Streamlit using components.html
        components.html(shap_html, height=500)





       











