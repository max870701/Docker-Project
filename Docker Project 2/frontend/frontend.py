import streamlit as st
import requests

# Streamlit 設定
st.title('Model Prediction Frontend')
st.write('Provide input features and get the prediction from the model.')

# 用戶輸入
features_input = st.text_input("Enter features (comma-separated)", "")

# 提交按鈕
if st.button('Get Prediction'):
    # 整理輸入數據
    features = [float(i) for i in features_input.split(",")]

    # 發送請求到後端 API
    response = requests.post('http://backend:8000/predict/', json={'features': features})

    if response.status_code == 200:
        st.success(f'Prediction: {response.json()["prediction"]}')
    else:
        st.error(f'Error: {response.text}')