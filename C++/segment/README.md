# YOLO26 / YOLO11 实例分割模型转 TensorRT

![_10014](./output/_10014.jpeg)

默认使用 `onnx_model/yolo26m-seg.onnx`，推理接口仍兼容相同输出布局的 YOLO11 分割模型。项目使用 TensorRT 8.x API；当前模型使用 Opset 16，建议使用 TensorRT 8.6。

## 模型要求

当前提供的 YOLO26 ONNX 由 Ultralytics 8.4.211 导出，`end2end=False`、`nms=None`，输入输出均为 FP32：

| 张量 | 形状 | 内容 |
| --- | --- | --- |
| `images` | `[1, 3, 640, 640]` | RGB、CHW、归一化到 0–1 |
| `output0` | `[1, 116, 8400]` | `xywh` + 80 个类别分数 + 32 个掩膜系数 |
| `output1` | `[1, 32, 160, 160]` | 掩膜原型 |

程序按张量名称查询 TensorRT 绑定索引，并校验形状和数据类型；后处理保留转置、置信度过滤、NMS 和掩膜重建。Letterbox 填充值为 114。

不支持 `[1, 300, 38]` 等端到端或内置 NMS 输出。需要重新导出时，以下示例适用于 Ultralytics 8.4.211：

```python
from ultralytics import YOLO

model = YOLO("./weights/yolo26m-seg.pt")
model.export(
    format="onnx", imgsz=640, batch=1, dynamic=False,
    opset=16, simplify=False, nms=None,
)
```

将导出文件放入 `onnx_model`。不同 Ultralytics 版本的导出选项可能不同，应核实实际输出符合上述布局。

## 编译和运行

依赖 CUDA、TensorRT 8.x 和 OpenCV。从当前项目目录执行：

```bash
mkdir -p build
cd build
cmake .. -DTENSORRT_ROOT=/home/leslie/TensorRT-8.6.1.6
cmake --build . -j
./main ../images
```

`TENSORRT_ROOT` 替换为实际安装目录；系统已安装 TensorRT 时可省略。需要选择其他 CUDA 时，可额外传入 `-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.7`。

首次运行构建 `yolo26m-seg.plan`，之后直接加载该文件。引擎和推理结果图像保存在运行目录。

更换模型或切换回 YOLO11 时，在 `include/config.h` 中同时修改 `onnxFile` 和 `trtFile`，并检查类别数、类别名称、输入尺寸和掩膜通道数。修改模型、构建精度或 TensorRT/CUDA 环境后，应移走旧 `.plan` 重新构建；启用 INT8 时也应使用对应模型和预处理的校准缓存。
