import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Iris Flower Prediction App

* By using this app, we can predict the type of 
iris flower based on its parameters.
""")
st.divider()

# Sidebar
st.sidebar.header("Specify Input Parameters")


def user_input_features():
    sepal_length = st.sidebar.slider("Sepal length", 4.3, 7.9, 5.1)
    sepal_width = st.sidebar.slider("Sepal width", 2.0, 4.4, 2.9)
    petal_length = st.sidebar.slider("Petal length", 1.0, 6.9, 1.2)
    petal_width = st.sidebar.slider("Petal width", 0.1, 2.5, 0.3)
    data = {"sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.write("##### Specified Input Parameters")
st.write(df)

# Load dataset and split it into X and Y
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Model building
model = RandomForestClassifier()
model.fit(X, Y)

# Prediction
prediction = model.predict(df)
if prediction == 0:
    predict = "setosa"
elif prediction == 1:
    predict = "veriscolor"
else:
    predict = "virginica"

# Prediction Probability
predict_prob = model.predict_proba(df)

col1, col2 = st.columns(2)
with col1:
    st.write("##### Prediction")
    st.write(iris.target_names[prediction])
    st.success(f"The type of iris flower is ***{predict}***")
with col2:
    st.write("##### Prediction Probability")
    st.write(predict_prob)
    # Class labels
    st.info("""
     * 0 - **setosa**, 1 - **veriscolor**, 2 - **virginica**
    """)
