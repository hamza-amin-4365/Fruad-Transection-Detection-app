import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

# Function to load model and scaler
def load_model():
    df = pd.read_csv('creditcard.csv')
    legit = df[df.Class == 0]
    fraud = df[df.Class == 1]
    legit_sample = legit.sample(n=492)
    new_df = pd.concat([legit_sample, fraud], axis=0)
    X = new_df.drop(columns='Class', axis=1)
    Y = new_df['Class']

    model = LogisticRegression()
    model.fit(X, Y)

    return model, X

# Function to make a prediction
def predict_fraud(model, user_input):

    # Reshape the input to a 2D array
    user_input_2d = np.array(user_input).reshape(1, -1)

    # Make prediction
    prediction = model.predict(user_input_2d)

    return prediction[0]

# Function to create the Streamlit app
def main():
    # Set page title and icon
    st.set_page_config(page_title="Fraud Detection App", page_icon=":money_with_wings:")

    # Load model and features
    model, X = load_model()

    # Add a header with custom styling
    st.title("Fraud Detection App")
        # Upload image and display it with improved styling
    img = Image.open("So.png")
    #img = img.resize((400, 300))
    st.image(img, caption="Transection Image", use_column_width=True)
    
    st.markdown("### Adjust Feature Values:")

    # Create number_input for each feature
    user_input = []
    for feature in X.columns:
        value = st.number_input(f"{feature}:",  step=0.1)
        user_input.append(value)

    # Make prediction when the user clicks the "Predict" button
    if st.button("Predict", key="predict_button"):
        prediction = predict_fraud(model, user_input)

        # Display prediction with custom styling
        st.sidebar.title("Inputs:")
        for feature, value in zip(X.columns, user_input):
            st.sidebar.text(f"{feature}: {value}")

        st.sidebar.title("Prediction:")
        if prediction == 1:
            st.sidebar.error("Fraudulent Transaction")
        else:
            st.sidebar.success("Legitimate Transaction")

if __name__ == "__main__":
    main()
