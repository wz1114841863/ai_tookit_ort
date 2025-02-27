import os
import tempfile
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
    allowed_types = ["image/jpeg", "image/png", "image/jpg",]
    print(file.content_type)
    print(file)
    # if file.content_type not in allowed_types:
    #     raise HTTPException(status_code=400, detail="仅支持图片")

    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_path = temp_file.name

        onnx_path = "./lite/hub/ort/age_googlenet.onnx"
        net = AgeGoogleNet(onnx_path, 2)
        mat = cv.imread(temp_path)
        print(mat.shape)
        result = net.detect(mat)
        return {"result": {result.age}}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"处理时发生错误: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
