import requests
import cv2 as cv
import numpy as np

url = "http://localhost:8000/upload_images/"
files = {
    "file": open(
        "/home/wz/AI/ai_tookit_ort/resources/test_lite_age_googlenet.jpg", "rb"
    )
}
data = {"network_name": "FastStyleTransfer"}

response = requests.post(url, files=files, data=data)
if response.status_code == 200:
    # 检查响应内容类型
    content_type = response.headers.get("Content-Type", "")
    if "image" in content_type:
        # 将字节流转换为 NumPy 数组
        img_raw_code = np.frombuffer(response.content, np.uint8)
        # 解码图像
        mat = cv.imdecode(img_raw_code, cv.IMREAD_COLOR)
        if mat is not None:
            cv.imwrite("./test.jpg", mat)
            print("图像已成功保存为 test.jpg")
        else:
            print("错误：无法解码图像数据")
    else:
        # 如果返回的不是图像，打印响应内容
        print("接口返回的内容不是图像：")
        print(response.content)
else:
    print("错误：", response.status_code, response.json())
