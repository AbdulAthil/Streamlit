pip install streamlit
pip install matplotlib
pip install sklearn
pip install pandas
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.title("""Used Car Sales Price Prediction App

* By using this app, we can predict the price of the car
with the customer's selected specifications.

* We can change the input variables as per our requirements.""")

# Read the dataset
car_data = pd.read_csv("car-sales-extended-missing-data.csv")
# st.dataframe(car_data)
null = car_data.isnull().sum()
# st.write(null)

# Fill the NaN values
car_data["Make"] = car_data["Make"].fillna(car_data["Make"].mode()[0])
car_data["Colour"] = car_data["Colour"].fillna(car_data["Colour"].mode()[0])
car_data["Odometer (KM)"] = car_data["Odometer (KM)"].fillna(car_data["Odometer (KM)"].mode()[0])
car_data["Doors"] = car_data["Doors"].fillna(car_data["Doors"].mode()[0])
car_data["Price"] = car_data["Price"].fillna(car_data["Price"].median())
null1 = car_data.isnull().sum()
# st.write(null1)
# Dataset has no null values

# Dataset and its information
view_d = st.checkbox("View Dataset")
if view_d:
    st.subheader("Dataset")
    st.dataframe(car_data)
    st.markdown(""" ### Dataset Information
    
         * Make - Car make (Brand) - BMW - 0, Honda - 1, Nissan - 2, Toyota - 3 
         
         * Colour - Color of the car - Black - 0, Blue - 1, Green - 2, Red - 3, White - 4 

         * Odometer (KM) - How many kilometers the car has been driven 

         * Doors - How many doors the car have 
        """)
st.divider()

# Data Preprocessing
le = LabelEncoder()
cols = ["Make", "Colour"]
for col in cols:
    car_data[col] = le.fit_transform(car_data[col])
# st.dataframe(car_data)

# Split the dataset into X and Y
X = car_data.drop(columns="Price")
# st.write(X)
Y = car_data["Price"]
# st.write(Y)

# Sidebar
st.sidebar.header("Specify input parameters ")


def user_inputs_features():
    # Make selection
    make = None
    makes = ["BMW", "Honda", "Nissan", "Toyota"]
    select_make = st.sidebar.selectbox("Select Make", makes)
    if select_make == "BMW":
        make = 0
    elif select_make == "Honda":
        make = 1
    elif select_make == "Nissan":
        make = 2
    elif select_make == "Toyota":
        make = 3
    # Colour selection
    colours = ["Black", "Blue", "Green", "Red", "White"]
    select_colour = st.sidebar.selectbox("Select colour", colours)
    if select_colour == "Black":
        colour = 0
    elif select_colour == "Blue":
        colour = 1
    elif select_colour == "Green":
        colour = 2
    elif select_colour == "Red":
        colour = 3
    else:
        colour = 4
    # Odometer selection
    odometer = st.sidebar.slider("Odometer (KM)", X["Odometer (KM)"].min(),
                                 X["Odometer (KM)"].max(), X["Odometer (KM)"].median())
    doors = st.sidebar.slider("Doors", 3, 5, 4)
    # Store user inputs in a dictionary
    data = {"Make": make,
            "Colour": colour,
            "Odometer (KM)": odometer,
            "Doors": doors}
    # Create a dataframe using the dictionary
    features = pd.DataFrame(data, index=[0])
    return features  # Return the dataframe


df = user_inputs_features()

# Print specified input parameters
st.write("##### Specified Input Parameters")
st.write(df)
st.divider()

# Model Building
model = RandomForestRegressor(random_state=0)
model.fit(X, Y)

# Prediction
st.write("##### Price Prediction")
prediction = model.predict(df)
st.write(prediction)
st.success(f"The price of the selected car will be {int(prediction)} rupees")
st.divider()
