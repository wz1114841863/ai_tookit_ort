import os
import tempfile
import numpy as np
import cv2 as cv
from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import JSONResponse
from lite.cv.age_googlenet import AgeGoogleNet

app = FastAPI()


@app.get("/upload_images/")
async def test_get():
    return {"file_size": 10}


@app.post("/upload_images/")
async def test_interface(
    file: UploadFile = File(..., description="待处理的图像文件"),
    params: str = None,
):
    """ 对图像输入进行测试 """
    # allowed_types = ["image/jpeg", "image/png", "image/jpg",]
    # print(file.content_type)
    # print(file)
    # if file.content_type not in allowed_types:
    #     raise HTTPException(status_code=400, detail="仅支持图片")

    try:
        # 直接读取字节数据(无需临时文件)
        contents = await file.read()

        # 将字节流转换为OpenCV格式
        nparr = np.frombuffer(contents, np.uint8)
        mat = cv.imdecode(nparr, cv.IMREAD_COLOR)

        # 检查图像是否有效
        if mat is None:
            raise HTTPException(status_code=400, detail="无法解析图像内容")

        print(mat.shape)  # 确保此处有输出

        onnx_path = "./lite/hub/ort/age_googlenet.onnx"
        net = AgeGoogleNet(onnx_path, 2)
        result = net.detect(mat)
        return {"result": str(result.age)}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"处理时发生错误: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
