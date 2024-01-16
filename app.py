import streamlit as st
import pandas as pd
import pickle

# Load the model
model = pickle.load(open('finalized_model.sav', 'rb'))

# Define a function to preprocess input data
def preprocess_data(data):
    df = pd.DataFrame([data])

    # Get dummy variables, ensuring correct column order and handling missing categories
    df = pd.get_dummies(df, columns=['ocean_proximity'], prefix='ocean_proximity')
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)  # Reindex to match model's expected columns

    return df



# Create an engaging title and subtitle
st.title("✨  Predict House Prices with Confidence!  ✨")
st.write("Welcome to this interactive tool for predicting house prices in California!")

# Input features
longitude = st.number_input("Longitude", value=-118.2437)
latitude = st.number_input("Latitude", value=34.0522)
housing_median_age = st.number_input("Housing Median Age", value=30)
total_rooms = st.number_input("Total Rooms", value=5610)
total_bedrooms = st.number_input("Total Bedrooms", value=1283)
population = st.number_input("Population", value=1425)
households = st.number_input("Households", value=496)
median_income = st.number_input("Median Income", value=3.5214)
ocean_proximity = st.selectbox("Ocean Proximity", ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY'])

# Engage the user with a visually appealing prediction button
if st.button("Predict House Price "):
    # Preprocess input data
    data = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }
    df = preprocess_data(data)

    # Make prediction
    prediction = model.predict(df)[0]

    # Display prediction in a visually appealing way
    st.success(f" **Predicted Median House Value:** ${prediction:,.2f}")

else:
    st.info("Input house features and click the button to predict the price.")
