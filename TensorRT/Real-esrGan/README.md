# Real-ESRGAN
The Pytorch implementation is [real-esrgan](https://github.com/xinntao/Real-ESRGAN).

<p align="center">
<img src="https://user-images.githubusercontent.com/40158321/170728105-0a1429e8-d117-4844-9c4b-a2d9db4a4ada.png">
</p>

## Config
- Input shape(**INPUT_H**, **INPUT_W**, **INPUT_C**) defined in real-esrgan.cpp
- GPU id(**DEVICE**) can be selected by the macro in real-esrgan.cpp
- **BATCH_SIZE** can be selected by the macro in real-esrgan.cpp
- FP16/FP32 can be selected by **PRECISION_MODE** in real-esrgan.cpp
- The example result can be visualized by **VISUALIZATION**. 

## How to Run, real-esrgan as example

build tensorrtx/real-esrgan and run

```
cd {TensorRT}/real-esrgan/
mkdir build
cd build
wegt https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/lvmd/real-esrgan.wts
cmake ..
make
./real-esrgan -s [.wts] [.engine]   // serialize model to plan file
./real-esrgan -d [.engine] [image folder]  // deserialize and run inference, the images in [image folder] will be processed.
```


