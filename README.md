## Finish
1. age_googlenet: 年龄检测
2. yolox: 目标检测
3. yolov5: 目标检测
4. face_yolov8: 人脸检测
5. face_ultra: 人脸检测
6. emotion_ferplus: 情绪识别
7. fsanet: 人脸的欧拉角识别
8. pfld106：人脸关键点识别
9. ssrnet: 年龄识别
10. gender_googlenet: 性别识别
11. subpixel_cnn: 超分辨率网络
12. age_vgg16: 年龄检测
13. gender_vgg16: 性别检测
14. fast_style_transfer: 艺术风格转换
15. glint_arcface: 人脸相似度计算
16. colorize：图像上色
17. MobilenetV2: 图像分类
18. ShuffleNetV2: 图像分类
19. EfficientDetAnchor: 目标检测网络
20. SSD: 目标检测网络
21. Resnet: 目标检测网络

## 添加网络流程
1. lite/cv/目录下添加对应的模型文件，包括前处理、模型推理、后处理和测试步骤
2. 下载对应的onnx文件添加到hub/ort
3. 在lite/backend中添加对应的模型处理流程
4. 在lite/frontend中添加对应的网络模型名称用于调用


## 启动流程
后端采用FastAPI
python -m lite.backend.interface

前端采用Streamlit
streamlit run ./lite/fronted/app.py

## ONNX 文件链接
[谷歌drive](https://drive.google.com/drive/folders/1p6uBcxGeyS1exc-T61vL8YRhwjYL4iD2)
