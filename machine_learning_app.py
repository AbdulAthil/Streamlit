import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

# Page layout - Page expands to full width
st.set_page_config(page_title='Machine Learning App', layout='wide')
# Model building


def build_model(df):
    st.write("#### Select Target column")
    target_column = st.text_input("Select your Target column which you want to analyse")

    if target_column:
        st.write(f"You selected {target_column} as target_column")
        x = df.drop(columns=[target_column])  # Using all column except target_column for as X
        y = df[target_column]  # Using the target_column as Y
    else:
        st.write("You did not select any specific target_column. "
                 "So the model will take the last column as target column")
        x = df.iloc[:, :-1]  # Using all column except for the last column as X
        y = df.iloc[:, -1]  # Selecting the last column as Y

    # Data splitting
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(100 - split_size) / 100,
                                                        random_state=parameter_random_state)

    st.write("#### Variable details")
    st.write("**x - variables(Independent)**")
    st.write(list(x.columns))
    st.write("**y - variable(Dependent)**")
    st.write(y.name)
    st.divider()

    st.write("#### Data splits")
    st.write("* **Training set**", x_train.shape)
    st.write("* **Test set**", x_test.shape)
    st.divider()

    # Model Building
    rf = RandomForestClassifier(n_estimators=parameter_n_estimators,
                                random_state=parameter_random_state)
    rf.fit(x_train, y_train)

    st.write(y.dtypes)
    st.write(y.unique())
    try:
        rf.fit(x_train, y_train)
    except ValueError as e:
        st.write(e)
        st.write("Unique labels in y_train:", np.unique(y_train))

    
    
    # Prediction
    y_predict_test = rf.predict(x_test)

    # Evaluation
    st.write("#### Model Performance")
    st.write("* ###### Classification Report")
    report = metrics.classification_report(y_test, y_predict_test)
    st.text(report)
    st.write("")
    st.write("* ###### Accuracy")
    accuracy = rf.score(x_test, y_test)*100
    st.info(f"Model Accuracy = ***{round(accuracy,3)}***")
    st.write(" ")
    st.write("* ###### Confusion Matrix")
    matrix = metrics.confusion_matrix(y_test, y_predict_test)
    st.table(matrix)

    st.write("* ###### Cross Validation scores")
    col1, col2 = st.columns(2)
    with col1:
        cv_acc = cross_val_score(rf, x, y, cv=3, scoring="accuracy")
        st.info(f"* Accuracy score = ***{round(np.mean(cv_acc),4)*100}***")
        cv_precision = cross_val_score(rf, x, y, cv=3, scoring="precision_macro")
        st.info(f"* Precision score = ***{round(np.mean(cv_precision),4)*100}***")
    with col2:
        cv_recall = cross_val_score(rf, x, y, cv=3, scoring="recall_macro")
        st.info(f"* Recall score = ***{round(np.mean(cv_recall),4)*100}***")
        cv_f1 = cross_val_score(rf, x, y, cv=3, scoring="f1_macro")
        st.info(f"* F1 score = ***{round(np.mean(cv_f1),4)*100}***")
    st.divider()


# --------------------------------- #


st.write("""# The Machine Learning App
* By using this app, we can make *Machine Learning prediction* for any dataset.

* In this app, ***RandomForestClassifier()*** function is used to build the model.

* Upload your dataset in the sidebar or Click on the button shown below to run with the default dataset.
""")
st.divider()

# --------------------------------- #

# Sidebar
with st.sidebar.header("Upload Your Data"):
    # st.write("#### Upload your input CSV file")
    uploaded_file = st.sidebar.file_uploader("*File must be in CSV format.*", type=["csv"])

with st.sidebar.subheader("Set Parameters"):
    split_size = st.sidebar.slider('**Data split ratio** (% for Training Set)', 10, 90, 75, 5)
    parameter_n_estimators = st.sidebar.slider('Number of estimators (**n_estimators**)', 0, 1000, 100, 50)
    parameter_random_state = st.sidebar.slider('Seed number (**random_state**)', 0, 1000, 42, 1)

# --------------------------------- #
# Main panel

# Displays the dataset

if uploaded_file is not None:
    df_user = pd.read_csv(uploaded_file)
    df = df_user.dropna()
    # st.write(df.isnull().sum())
    st.write("#### Dataset Information")
    st.write(df.head())
    st.write("* Dataset Shape :", df.shape)
    cols = []
    for columns in df.columns:
        cols.append(columns)
    st.write(f"* Columns : {cols}")
    # st.write(df.describe().head(3))
    st.divider()
    le = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == "object":
            df.loc[:, column] = le.fit_transform(df[column])
    # st.write(df.head())
    build_model(df)

else:
    st.info('Waiting for CSV file to be uploaded.')
    button = st.checkbox('Click here to use Example Dataset')
    if button:
        df = pd.read_csv("diabetes.csv")

        st.subheader("Dataset Information")
        st.markdown('The Diabetes dataset is used as an example.')
        st.write(df.head())
        st.write("* Dataset Shape :", df.shape)
        cols = []
        for columns in df.columns:
            cols.append(columns)
        st.write(f"* Columns : {cols}")
        # st.write(df.describe().head(3))

        st.divider()

        build_model(df)
