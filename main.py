import streamlit as st
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Set the page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon=":computer:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set the theme
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
        color: #1f1f1f;
        font-family: 'Helvetica', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.subheader('About')
st.sidebar.write('This web app is designed to predict the price of a laptop based on its specifications. The model was trained on a dataset of over 1,000 laptops and achieved an R-squared value of 0.88.')
st.sidebar.write('The dataset used to train the model can be found [here](https://www.kaggle.com/ionaskel/laptop-prices).')

# Load the trained model
pipe = joblib.load('model.joblib')

# Define a function to make predictions
def predict_price(Company, Type_name, Ram, Weight, Touchscreen, Ips, ppi, Cpu_brand, HDD, SSD, Gpu_brand, os):
    # Create a dictionary with the input data
    input_dict = {
        'Company': [Company],
        'TypeName': [Type_name],
        'Ram': [Ram],
        'Weight': [Weight],
        'Touchscreen': [Touchscreen],
        'Ips': [Ips],
        'ppi': [ppi],
        'Cpu brand': [Cpu_brand],
        'HDD': [HDD],
        'SSD': [SSD],
        'Gpu brand': [Gpu_brand],
        'os': [os]
    }

    # Convert the dictionary to a DataFrame
    input_df = pd.DataFrame(input_dict)

    # Make the prediction
    predicted_price = pipe.predict(input_df)
    predicted_price = np.exp(predicted_price)

    return predicted_price[0]

# Create a web app
st.title('Laptop Price Predictor')

Company = st.selectbox("Select the company name:", ['Apple', 'Asus', 'Acer', 'Dell', 'HP', 'Lenovo', 'MSI', 'Razer', 'Toshiba'])
Type_name = st.selectbox("Select the type name:", ['Ultrabook', 'Gaming', 'Notebook', 'Netbook', '2 in 1 Convertible'])
Ram = st.selectbox("Select the RAM size in GB:", [2, 4, 6, 8, 12, 16, 32, 64])
Weight = st.number_input("Select the weight in kg:")
Touchscreen = st.selectbox("Does it have touchscreen?", ["No", "Yes"])
if Touchscreen == "Yes":
    Touchscreen=1
else :
    Touchscreen=0


Ips = st.selectbox("Does it have IPS screen?", ["No", "Yes"])
if Ips == "Yes":
    Ips=1
else :
   Ips=0
ppi = st.number_input("Enter the PPI value:", min_value=0, step=1, max_value=100000)

Cpu_brand = st.selectbox("Select the CPU brand:", ['Intel Core i7','Intel Core i5','Other Intel Processor',
'Intel Core i3',
'AMD Processor'])
HDD = st.selectbox("Select the HDD size in GB (0 if not applicable):", [0, 128, 256, 512, 1024, 2048, 4096])
SSD = st.selectbox("Select the SSD size in GB (0 if not applicable):", [0, 128, 256, 512, 1024, 2048, 4096])
Gpu_brand = st.selectbox("Select the GPU brand:", ['Intel', 'Nvidia', 'AMD'])
os = st.selectbox("Select the operating system:", ['Mac', 'Windows', 'Others'])

if st.button('Predict price'):
    # Make the prediction and display the result
    price = predict_price(Company, Type_name, Ram, Weight, Touchscreen, Ips, ppi, Cpu_brand, HDD, SSD, Gpu_brand, os)
    st.success('The predicted price is: RUPEES %.2f' % price)
