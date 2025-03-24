import os
import numpy as np
import cv2 as cv
from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from fastapi.responses import JSONResponse, Response
from lite.cv import *
from lite.utils import *


class ModelInfo:
    """定义与网络相关的文件和方法"""

    def __init__(self, model_name, onnx_path, draw_method):
        self.model_name = model_name
        self.onnx_path = onnx_path
        self.draw_method = draw_method
        self.net = self.model_name(onnx_path)


name2network = {
    "AgeGoogleNet": ModelInfo(
        AgeGoogleNet,
        "lite/hub/ort/age_googlenet.onnx",
        draw_age,
    ),
    "AgeVGG16": ModelInfo(
        AgeVGG16,
        "lite/hub/ort/vgg16_age.onnx",
        draw_age,
    ),
    "Colorize": ModelInfo(
        Colorize,
        "lite/hub/ort/siggraph17-colorizer.onnx",
        None,
    ),
    "EmotionFerPlus": ModelInfo(
        EmotionFerPlus,
        "lite/hub/ort/emotion-ferplus-7.onnx",
        draw_emotion,
    ),
    "UltraFace": ModelInfo(
        UltraFace,
        "lite/hub/ort/ultraface-slim-320.onnx",
        draw_boxes,
    ),
    "FaceYolov8": ModelInfo(
        FaceYolov8,
        "lite/hub/ort/yoloface_8n.onnx",
        draw_boxes,
    ),
    "FastStyleTransfer": ModelInfo(
        FastStyleTransfer,
        "lite/hub/ort/style-udnie-8.onnx",
        None,
    ),
    "FSANet": ModelInfo(  # TODO, 需要特殊处理
        FSANet,
        "lite/hub/ort/fsanet-var.onnx",
        None,
    ),
    "GenderGoogleNet": ModelInfo(
        GenderGoogleNet,
        "lite/hub/ort/gender_googlenet.onnx",
        draw_gender,
    ),
    "GenderVGG16": ModelInfo(
        GenderVGG16,
        "lite/hub/ort/vgg16_gender.onnx",
        draw_gender,
    ),
    "PFLD106": ModelInfo(
        PFLD106,
        "lite/hub/ort/pfld-106-lite.onnx",
        draw_landmarks,
    ),
    "SSRNet": ModelInfo(
        SSRNet,
        "lite/hub/ort/ssrnet.onnx",
        draw_age,
    ),
    "SubPixelCNN": ModelInfo(  # TODO: 接口得处理一下， 针对None的方法， 需要统一封装返回mat
        SubPixelCNN,
        "lite/hub/ort/subpixel-cnn.onnx",
        None,
    ),
    "Yolov5": ModelInfo(
        Yolov5,
        "lite/hub/ort/yolov5s.onnx",
        draw_boxes,
    ),
    "YoloX": ModelInfo(
        YoloX,
        "lite/hub/ort/yolox_nano.onnx",
        draw_boxes,
    ),
}
print(name2network.keys())
app = FastAPI()


@app.get("/upload_images/")
async def test_get():
    return {"file_size": 10}


@app.post("/upload_images/")
async def test_interface(
    file: UploadFile = File(..., description="待处理的图像文件"),
    network_name: str = Form(None, description="网络名称"),
    params: str = Form(None, description="其他参数"),
):
    """对图像输入进行测试"""
    try:
        contents = await file.read()
        # 将字节流转换为OpenCV格式
        nparr = np.frombuffer(contents, np.uint8)
        mat = cv.imdecode(nparr, cv.IMREAD_COLOR)
        # 检查图像是否有效
        if mat is None:
            raise HTTPException(status_code=400, detail="无法解析图像内容")
        print(mat.shape)  # 确保此处有输出

        if network_name not in name2network:
            raise HTTPException(
                status_code=400, detail=f"不支持的网络名称{network_name}"
            )
        # 假设所有模型都接受一个onnx路径作为参数，这里使用默认路径
        model_info = name2network[network_name]
        result = model_info.net.detect(mat)

        if model_info.draw_method:
            model_info.draw_method(mat, result)
            return_mat = mat
        else:
            return_mat = result.mat

        # 将处理后的图像编码为字节流
        _, encoded_image = cv.imencode(".jpg", return_mat)
        image_bytes = encoded_image.tobytes()
        # 返回图像字节流
        return Response(content=image_bytes, media_type="image/jpeg")

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": f"处理时发生错误: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
