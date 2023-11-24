import json
import requests
import streamlit as st

st.title("Answer Questions Project")
text = st.text_area('Texto', "")
question = st.text_input('Question', "")

if st.button('Predict', type="primary"):
    if not text:
        st.error("Empty Text")
    elif not question:
        st.error("Empty Question")
    else:
        st.text("Predicting...")
        data_json = {
            "text": text,
            "question": question
        }
        response = requests.post("http://127.0.0.1:8000/model/predict", data=json.dumps(data_json))
        response_json = response.json()
        
        st.text(f"Answer: {response_json['answer']}")
        st.success('Done')

# streamlit run ml_ui.py
