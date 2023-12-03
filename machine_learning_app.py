import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Page layout - Page is centered
st.set_page_config(page_title='Machine Learning App', layout='centered')

# Function to build and evaluate the machine learning model


def build_and_evaluate_model(df, target_column, missing_value_strategy):
    st.write("### Machine Learning Model Evaluation")

    # Handling missing values
    if missing_value_strategy == "Drop Rows with Missing Values":
        df = df.dropna()
    elif missing_value_strategy == "Fill with Mean":
        df = df.fillna(df.mean())
    elif missing_value_strategy == "Fill with Median":
        df = df.fillna(df.median())
    elif missing_value_strategy == "Fill with Zero":
        df = df.fillna(0)
    st.write(df.head())

    # Data preprocessing
    x = df.drop(columns=[target_column])
    y = df[target_column]

    # Data splitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(100 - split_size) / 100,
                                                        random_state=parameter_random_state)

    # Variable details
    st.write("#### Variable details")
    st.write("**Features (X):**")
    st.write(f"***{list(x.columns)}***")
    st.write("**Target variable (Y):**")
    st.write(f"***{target_column}***")
    st.divider()

    # Data splits
    st.write("#### Data splits")
    st.write("* **Training set:**", x_train.shape)
    st.write("* **Test set:**", x_test.shape)
    st.divider()

    # Model Building
    st.write("#### Model Building")

    # Check if the dtype is numeric (continuous) or discrete
    if np.issubdtype(y.dtype, np.number):  
        st.info("Target variable is ***continuous***. Applying ***RandomForestRegressor***.")
        st.divider()
        model = RandomForestRegressor(n_estimators=parameter_n_estimators, random_state=parameter_random_state)
    else:
        st.info("Target variable is ***discrete***. Applying ***RandomForestClassifier***.")
        model = RandomForestClassifier(n_estimators=parameter_n_estimators, random_state=parameter_random_state)

    # Checking the target variable distribution
    if len(y.unique()) < 2:
        st.warning("Target variable has less than 2 unique values. Check your data.")

    # Fit the train data into our model
    model.fit(x_train, y_train)

    # Prediction
    y_predict_test = model.predict(x_test)

    # Evaluation
    st.write("#### Model Performance")

   
    if np.issubdtype(y.dtype, np.number):  
        # Regression metrics        
        accuracy = model.score(x_test, y_test) * 100
        st.info(f"**Model Accuracy** = ***{round(accuracy, 3)}***")
        st.write("* ***Mean Absolute Error (MAE):***", round(metrics.mean_absolute_error(y_test, y_predict_test), 3))
        st.write("* ***Mean Squared Error (MSE):***", round(metrics.mean_squared_error(y_test, y_predict_test), 3))
        st.write("* ***R-squared:***", round(metrics.r2_score(y_test, y_predict_test), 3))
    else:  
        # Classification metrics
        st.warning("Target variable is not discrete. Classification metrics are not suitable.")

    # Cross validation
    st.write("##### Cross Validation scores")
    col1, col2 = st.columns(2)
    with col1:
        st.write("* **Accuracy score:**", np.mean(cross_val_score(model, x, y, cv=3, scoring="accuracy")))
    with col2:
        if np.issubdtype(y.dtype, np.number):  # Regression metrics for cross-validation
            st.write("* **R-squared:**", np.mean(cross_val_score(model, x, y, cv=3, scoring="r2")))

    st.success("Model evaluation completed.")
    st.divider()

# Sidebar


st.sidebar.title("Machine Learning App")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 75, 5)
parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 0, 1000, 100, 50)
parameter_random_state = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
missing_value_strategy = st.sidebar.selectbox("Missing Value Strategy", ["Drop Rows with Missing Values", "Fill with Mean", "Fill with Median", "Fill with Zero"])

# Main panel
st.divider()
st.title("Machine Learning App")
st.divider()

# Upload file 
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Information")
    st.write(df.head())
    st.write(f"* Dataset Shape : {df.shape}")
    st.write(f"* Columns : {list(df.columns)}")
    st.divider()

    # Label Encoding(convert categorical data)
    le = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = le.fit_transform(df[column])

    # Target selection from user
    st.write("#### Target Column Selection")
    st.write("If you don't select any specific column, "
             "the *model* will take the first column as target column by default.")
    target_column = st.selectbox("Select Target column", df.columns)
    st.divider()
    
    build_and_evaluate_model(df, target_column, missing_value_strategy)

else:
    st.info('Waiting for CSV file to be uploaded.')
    st.warning('Please upload a CSV file to proceed.')
