# TensorRT deployment for YOLO26 / YOLO11 instance segmentation

![_10014](output/_10014.jpeg)

The default model is `onnx_model/yolo26m-seg.onnx`. YOLO11 segmentation exports with the same output layout are also supported. This project uses TensorRT 8.x APIs. TensorRT 8.6 is recommended for the supplied Opset 16 model.

## Model contract

The supplied YOLO26 model was exported with Ultralytics 8.4.211, `end2end=False` and `nms=None`. All I/O tensors are FP32:

| Tensor | Shape | Contents |
| --- | --- | --- |
| `images` | `[1, 3, 640, 640]` | RGB, CHW, normalized to 0–1 |
| `output0` | `[1, 116, 8400]` | `xywh`, 80 class scores, 32 mask coefficients |
| `output1` | `[1, 32, 160, 160]` | Mask prototypes |

Bindings are resolved by tensor name and checked for compatible shapes and data types. Postprocessing applies transpose, confidence filtering, NMS and mask reconstruction. Letterbox padding uses 114.

End-to-end or embedded-NMS outputs such as `[1, 300, 38]` are unsupported. To export a compatible model using Ultralytics 8.4.211:

```python
from ultralytics import YOLO

model = YOLO("./weights/yolo26m-seg.pt")
model.export(
    format="onnx", imgsz=640, batch=1, dynamic=False,
    opset=16, simplify=False, nms=None,
)
```

Place the ONNX file in `onnx_model`. Export options can differ between Ultralytics versions; verify the actual tensor layout.

## Build and run

Install CUDA, TensorRT 8.x and OpenCV, then run from this project directory:

```bash
mkdir -p build
cd build
cmake .. -DTENSORRT_ROOT=/home/leslie/TensorRT-8.6.1.6
cmake --build . -j
./main ../images
```

Replace `TENSORRT_ROOT` with your installation directory, or omit it for a system installation. To select another CUDA toolkit, also pass `-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.7`.

The first run builds `yolo26m-seg.engine`; subsequent runs load it directly. Engines and annotated images are saved in the working directory.

When changing models or switching back to YOLO11, update both `onnxFile` and `trtFile` in `include/config.h`, and check class count, class names, input size and mask channels. Move aside the cached engine file and rebuild after changing the model, build precision or TensorRT/CUDA environment. INT8 calibration caches must also match the model and preprocessing.
