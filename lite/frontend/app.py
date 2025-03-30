import io
import requests
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


st.title("图像处理")

FASTAPI_URL = "http://localhost:8000/upload_images/"

NETWORK_OPTIONS = [
    "AgeGoogleNet",
    "AgeVGG16",
    "Colorize",
    "EmotionFerPlus",
    "UltraFace",
    "FaceYolov8",
    "FastStyleTransfer",
    "FSANet",
    "GenderGoogleNet",
    "GenderVGG16",
    "PFLD106",
    "SSRNet",
    "SubPixelCNN",
    "Yolov5",
    "YoloX",
    "MobileNetV2",
    "ShuffleNetV2",
    "EfficientDetAnchor",
    "SSD",
]

uploaded_file = st.file_uploader("请选择要上传的图片", type=["jpg", "jpeg", "png"])

selected_model = st.selectbox(
    "选择网络模型",
    options=NETWORK_OPTIONS,
)

if uploaded_file is not None and selected_model:
    image = Image.open(uploaded_file)
    st.image(image, caption="原始图片", use_container_width=True)
    try:
        # 准备表单数据
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        data = {
            "network_name": selected_model,
        }

        with st.spinner("正在处理图像"):
            response = requests.post(FASTAPI_URL, files=files, data=data)

            if response.status_code == 200:
                # 检查返回的内容类型
                content_type = response.headers.get("content-type", "")
                if "image" in content_type:
                    # 如果是图像响应，显示图像
                    processed_image = Image.open(io.BytesIO(response.content))
                    st.image(
                        processed_image,
                        caption="处理后的图像",
                        use_container_width=True,
                    )
                else:
                    # 如果是JSON响应，显示结果
                    st.subheader("处理结果")
                    st.json(response.json())
            else:
                try:
                    error_detail = response.json()
                    st.error(f"处理失败: {error_detail.get('message', '未知错误')}")
                except:
                    st.error(
                        f"请求失败, 状态码: {response.status_code}, 错误信息: {response.text}"
                    )

    except Exception as e:
        st.error(f"发生错误:{str(e)}")
else:
    st.warning("请上传图片并选择模型")
