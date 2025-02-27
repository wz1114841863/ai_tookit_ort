import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import requests

st.title("图像识别演示")

FASTAPI_URL = "http://localhost:8000/upload_images/"

uploaded_file = st.file_uploader("请选择要上传的图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="上传的图片", use_container_width=True)

    files = {"file": uploaded_file}

    with st.spinner("正在识别中"):
        try:
            response = requests.post(FASTAPI_URL, files=files)
            if response.status_code == 200:
                result = response.json().get("result", "无返回结果")
                st.subheader("识别结果")
                st.write(result)
            else:
                st.error(f"请求失败, 状态码: {response.status_conde}")
        except requests.exceptions.RequestException as e:
            st.error(f"请求出错:{str(e)}")
        except Exception as e:
            st.error(f"发生未知错误:{str(e)}")
