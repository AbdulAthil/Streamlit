import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Boston House Price Prediction App

* By using this app, we can predict the ***Boston House Price*** 
based on different paramters.

""")

# Loads the Boston House Price Dataset
bos = pd.read_csv("HousingData.csv")
# st.dataframe(bos.tail())
# Check if the dataset has any missing values
null = bos.isnull().sum()
# st.write(null)

# Fill the nan values with 0
boston = bos.fillna(0)
# null1 = boston.isnull().sum()
# st.write(null)
# Now the dataset has no missing(NaN) Values

# Dataset and its information
check = st.checkbox("View Dataset")
if check:
    st.dataframe(boston)
    st.info("""### Dataset Information
    * CRIM - per capital crime rate by town
    * ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
    * INDUS - proportion of non-retail business acres per town.
    * CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
    * NOX - nitric oxides concentration (parts per 10 million)
    * RM - average number of rooms per dwelling
    * AGE - proportion of owner-occupied units built prior to 1940
    * DIS - weighted distances to five Boston employment centres
    * RAD - index of accessibility to radial highways
    * TAX - full-value property-tax rate per $10,000
    * PTRATIO - pupil-teacher ratio by town
    * B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    * LSTAT - % lower status of the population
    * MEDV - Median value of owner-occupied homes in $1000's""")

    # Download dataset as csv
    data = pd.read_csv("HousingData.csv")


    @st.cache_data
    def data_read(data):
        return data.to_csv().encode("utf-8")
    csv = data_read(data)
    
    st.download_button(
        label="Download dataset as csv",
        data=csv,
        file_name="Boston_HousingData.csv",
        mime="text.csv")
st.divider()

X = pd.DataFrame(boston.drop(columns="MEDV"))
Y = pd.DataFrame(boston["MEDV"])
# st.dataframe(X)
# st.dataframe(Y)


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')


def user_input_features():
    crim = st.sidebar.slider('CRIM', X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
    zn = st.sidebar.slider('ZN', X.ZN.min(), X.ZN.max(), X.ZN.mean())
    indus = st.sidebar.slider('INDUS', X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())
    chas = st.sidebar.slider('CHAS', X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())
    nox = st.sidebar.slider('NOX', X.NOX.min(), X.NOX.max(), X.NOX.mean())
    rm = st.sidebar.slider('RM', X.RM.min(), X.RM.max(), X.RM.mean())
    age = st.sidebar.slider('AGE', X.AGE.min(), X.AGE.max(), X.AGE.mean())
    dis = st.sidebar.slider('DIS', X.DIS.min(), X.DIS.max(), X.DIS.mean())
    rad = st.sidebar.slider('RAD', X.RAD.min(), X.RAD.max(), X.RAD.min())
    tax = st.sidebar.slider('TAX', X.TAX.min(), X.TAX.max(), X.TAX.min())
    ptratio = st.sidebar.slider('PTRATIO', X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean())
    b = st.sidebar.slider('B', X.B.min(), X.B.max(), X.B.mean())
    lstat = st.sidebar.slider('LSTAT', X.LSTAT.min(), X.LSTAT.max(), X.LSTAT.mean())
    data = {'CRIM': crim,
            'ZN': zn,
            'INDUS': indus,
            'CHAS': chas,
            'NOX': nox,
            'RM': rm,
            'AGE': age,
            'DIS': dis,
            'RAD': rad,
            'TAX': tax,
            'PTRATIO': ptratio,
            'B': b,
            'LSTAT': lstat}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

# Print specified input parameters
st.write('##### Specified Input parameters')
st.write(df)
st.divider()

# Build Regression Model
model = RandomForestRegressor(random_state=1)
model.fit(X, Y)

# Apply Model to Make Prediction
prediction = model.predict(df)
st.write('##### Prediction of MEDV')
st.write(prediction)
st.success(f"The price of the house will be $***{round(float(prediction),3)*1000}*** ")
st.divider()

