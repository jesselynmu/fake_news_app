import pickle
import streamlit as st

# Load the trained model
tf_idf = pickle.load(open('tfidf_vectorizer.sav', 'rb'))
model = pickle.load(open('random_forest_model.sav', 'rb'))

# Title
st.title("Fake News Classification")

# Form Input
title = st.text_input("Input title here:")
text = st.text_area("Enter the news text here:")

content = title.strip() + " " + text.strip()
prediction = ''

# Prediction
if st.button("Predict"):
    if content.strip():
        try:
            # Transform the input using TF-IDF Vectorizer
            text_tfidf = tf_idf.transform([content])
            
            # Make a prediction
            prediction = model.predict(text_tfidf)
            
            # Display the prediction
            if prediction == 0:
                st.success("The news is likely to be **REAL**.")
            else:
                st.warning("The news is likely to be **FAKE**.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter both title and news text.")