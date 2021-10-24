# Ball Detection

## 安装
终端中运行以下shell，用于安装程序所需的运行依赖：
```
pip install -r requirements.txt
```

## 目录结构

`model.onnx`：训练好的球体检测模型文件。

`onnx_inference.py`：检测运行程序。该文件包含加载预训练模型、图片视频摄像头数据读取、球体检测和输出等功能，详情见注释。

`utils.py`：工具类。该文件包含图片预处理、NMS计算和检测结果输出等功能，详情见注释。

`requirements.txt`：运行依赖文件。

`test.jpg`：测试图片。

`test.mp4`：测试视频。

## 运行
1. 读取图片进行检测
```
python onnx_inference.py --mode=image --input_path=test.jpg
```
2. 读取视频进行检测
```
python onnx_inference.py --mode=video --input_path=test.mp4
```
3. 读取摄像头进行检测
```
python onnx_inference.py --mode=webcam --camid=0
```

## 检测过程介绍
模型的检测过程，对应`onnx_inference.py`中的`inference`方法（`Line: 72-100`）

### 输入
**图片**。当输入数据为视频或摄像头时，将按帧读取成图片输入进行检测（见`onnx_inference.py Line: 132`）。

### 输出
**图片**。当输入数据为视频或摄像头时，将按每帧的检测结果写入（见`onnx_inference.py Line: 135`）。

### 检测过程
1. 图片预处理

图片在进入检测模型前，会对其进行预处理操作（见`onnx_inference.py Line: 76`）。预处理操作主要是对原始图片放缩至预设尺寸（默认640*640）。（见`utils.py Line: 95-111 image_preprocess方法 `）

2. 送入模型进行检测

初始化onnx推理模型，并将预处理后的图片送入onnx模型中，得到检测结果predictions（见`onnx_inference.py Line: 78-82`）。

predictions的形状为(8400, 15)。8400行表示一张图片进入模型后会输出8400个检测框。15列分别表示每个检测框的
|  index   |0|1|2|3|4|5-14|
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 含义  | 中心点横坐标|中心点纵坐标|检测框宽度|检测框高度|置信度|属于每个类别的概率|

得到predictions后，计算左上角坐标与右下角坐标，并映射回原始图片上。（见`onnx_inference.py Line: 87-92`）

3. 挑选有效检测框

挑选依据：NMS得分>0.45 且 类别得分>0.1（见`onnx_inference.py Line: 93`，详细计算见`utils.py Line: 113-119 multiclass_nms方法`）。

4. 输出检测结果

将有效检测框绘制在原图中（见`onnx_inference.py Line: 94-97`）。

