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

    try:
        # 显式重置文件指针
        uploaded_file.seek(0)

        # 构建符合要求的文件参数
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }

        with st.spinner("正在识别中"):
            response = requests.post(FASTAPI_URL, files=files)

            if response.status_code == 200:
                st.subheader("识别结果")
                st.write(response.json())
            else:
                st.error(f"请求失败,状态码:{response.status_code}")

    except Exception as e:
        st.error(f"发生错误:{str(e)}")
