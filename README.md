[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />

<div align="center">
  <a href="https://github.com/niehen6174/LVMD">
    <img src="https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/img/202302130903057.png" alt="Logo" width="200" height="200">
  </a>



<h3 align="center">LVMD</h3>

  <p align="center">
    Low-Level Vision Model Deployment.
    <br />
    <a href="https://github.com/niehen6174/LVMD/tree/master/Docs"><strong>Explore the Docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/niehen6174/LVMD/tree/master/TensorRT/Spynet">View Demo</a>
    ·
    <a href="https://github.com/niehen6174/LVMD/issues">Report Bug</a>
    ·
    <a href="https://github.com/niehen6174/LVMD/issues">Request Feature</a>
  </p>

</div>

<!-- TABLE OF CONTENTS -->

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>





<!-- ABOUT THE PROJECT -->

## About The Project

> I used to study `Deblurring` and `Video Super-Resolution`, but I am very interested in `model deployment`. I found that there are very few deployment cases for `low-level vision` tasks, which may be due to the low demand for such tasks and the loss of accuracy.<br/>
> After learning `TensorRT` and `NCNN`, which are excellent inference frameworks, I decided to make deployment cases of `low-level vision` and open source it.

This repo will be dedicated to providing deployment cases in the `low-level-vision` field. Including using inference frameworks such as `TensorRT` and `NCNN` to deploy tasks such as `Deblurring`, `Image Super-resolution`, `Video super-resolution`,  `Image Denoising`.

The repo will also provide a series of tutorials such as `TensorRT` custom operators, using API to build a network, multiple input and multiple output, as well as performance testing and bottleneck analysis of the engine generated by `TensorRT`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

For the installation of `TensorRT`, the docker file of mmdeploy is used. `NCNN` can be installed directly in the container mentioned above.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.

* Docker

* TensorRT

* NCNN

* ---

### Installation

1. Git clone MMdeploy.

   ```shell
   git clone -b master https://github.com/open-mmlab/mmdeploy.git MMDeploy
   ```

2. Build docker image(GPU).

   ```sh
   cd mmdeploy
   docker build docker/GPU/ -t mmdeploy:master-gpu
   ```

3. Run docker container

   ```shell
   docker run --gpus all -it mmdeploy:master-gpu
   ```

4. Install NCNN

   ```sh
   apt install build-essential git cmake libprotobuf-dev protobuf-compiler libvulkan-dev vulkan-utils libopencv-dev
   git clone https://github.com/Tencent/ncnn.git
   cd ncnn
   git submodule update --init
   wget https://sdk.lunarg.com/sdk/download/1.2.189.0/linux/vulkansdk-linux-x86_64-1.2.189.0.tar.gz?Human=true -O vulkansdk-linux-x86_64-1.2.189.0.tar.gz
   tar -xf vulkansdk-linux-x86_64-1.2.189.0.tar.gz
   export VULKAN_SDK=$(pwd)/1.2.189.0/x86_64
   ```

5. Compile NCNN

   ```shell
   mkdir -p build
   cd build
   cmake -DNCNN_VULKAN=ON ..
   make -j4
   make install
   ```

   

6. Demo of TensorRT-Spynet

   ```shell
   git clone https://github.com/niehen6174/LVMD.git
   cd TensorRT/Spynet
   mkdir build
   cd build
   cmake ..
   make
   ./spynet -s
   ./spynet -d
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

Use this space to show useful examples of how a project can be used.

Demo of TensorRT-Spynet

1. Git clone repo

   ```shell
   git clone https://github.com/niehen6174/LVMD.git
   ```

2. Ready to compile

   ```shell
   cd TensorRT/Spynet
   mkdir build
   cd build
   ```

3. Download wts file

   ```
   wget https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/lvmd/Spynet.wts
   ```

4. Compile

   ```
   cmake ..
   make
   ```

5. Generating serialization model

   ```
   ./spynet -s
   ```

6. Inference 

   ```
   ./spynet -d
   ```

7. Testing the serialization model takes time

   ```
   trtexec --loadEngine=./addplugin.engine --plugins=./libFlowWarp.so --shapes=ref:3x32x32,supp:3x32x32 --verbose > result.log
   -- result.lgo
   [02/13/2023-02:07:47] [I] Host Latency
   [02/13/2023-02:07:47] [I] min: 1.02942 ms (end to end 1.271 ms)
   [02/13/2023-02:07:47] [I] max: 5.34741 ms (end to end 5.45886 ms)
   [02/13/2023-02:07:47] [I] mean: 1.21322 ms (end to end 1.40523 ms)
   [02/13/2023-02:07:47] [I] median: 1.18549 ms (end to end 1.29443 ms)
   [02/13/2023-02:07:47] [I] percentile: 1.32043 ms at 99% (end to end 2.52673 ms at 99%)
   [02/13/2023-02:07:47] [I] throughput: 0 qps
   [02/13/2023-02:07:47] [I] walltime: 2.44427 s
   [02/13/2023-02:07:47] [I] Enqueue Time
   [02/13/2023-02:07:47] [I] min: 1.11456 ms
   [02/13/2023-02:07:47] [I] max: 5.32129 ms
   [02/13/2023-02:07:47] [I] median: 1.13934 ms
   [02/13/2023-02:07:47] [I] GPU Compute
   [02/13/2023-02:07:47] [I] min: 1.01123 ms
   [02/13/2023-02:07:47] [I] max: 5.33582 ms
   [02/13/2023-02:07:47] [I] mean: 1.19868 ms
   [02/13/2023-02:07:47] [I] median: 1.17108 ms
   [02/13/2023-02:07:47] [I] percentile: 1.29785 ms at 99%
   [02/13/2023-02:07:47] [I] total compute time: 2.44051 s
   ```

8. Viewing each layer of the model takes time

   ```
   nsys profile --force-overwrite=true --stats=true -o model-OnlyRun ./spynet -d
   -- output
   NVTX Push-Pop Range Statistics:
    Time(%)  Total Time (ns)  Instances    Average      Minimum     Maximum                                         Range                                      
    -------  ---------------  ---------  ------------  ----------  ----------  --------------------------------------------------------------------------------
       50.0       1284566211          1  1284566211.0  1284566211  1284566211  TensorRT:ExecutionContext::enqueue                                              
       49.9       1280651003          1  1280651003.0  1280651003  1280651003  TensorRT:(Unnamed Layer* 19) [Convolution] + (Unnamed Layer* 20) [Activation]   
        0.0           228912          1      228912.0      228912      228912  TensorRT:(Unnamed Layer* 21) [Convolution] + (Unnamed Layer* 22) [Activation]   
        0.0           221764          1      221764.0      221764      221764  TensorRT:(Unnamed Layer* 79) [Convolution] + (Unnamed Layer* 80) [Activation]   
        0.0           206198          1      206198.0      206198      206198  TensorRT:ExecutionContext::recompute                                            
        0.0           181466          1      181466.0      181466      181466  TensorRT:(Unnamed Layer* 97) [Convolution] + (Unnamed Layer* 98) [ElementWise]  
        0.0           153100          1      153100.0      153100      153100  TensorRT:(Unnamed Layer* 35) [Convolution] + (Unnamed Layer* 36) [Activation]   
        0.0           129186          1      129186.0      129186      129186  TensorRT:(Unnamed Layer* 81) [Convolution] + (Unnamed Layer* 82) [Activation]   
        0.0           118147          1      118147.0      118147      118147  TensorRT:(Unnamed Layer* 23) [Convolution] + ---eta
   ```

​		Analysis of NVIDIA Nsight Systems.![NVIDIA Nsight Systems analyis](https://xsj-niehen.oss-cn-hangzhou.aliyuncs.com/img/202302131022398.png)



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->

## Roadmap

- [x] TensorRT-DeblurGAN
- [x] TensorRT-Real-EsrGAN
- [x] TensorRT-Spynet
- [ ] TensorRT-Basicvsr
  - [x] TensorRT-flow_warp Plgin
  - [ ] TensoRT-Basicvsr backbone
  - [ ] TensoRT-Basicvsr Triton
- [ ] NCNN-DeblurGAN
- [ ] NCNN-Real-EsrGAn

See the [open issues](https://github.com/niehen6174/LVMD/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Niehen6174 - email@niehen6174@qq.com

Project Link: [LVMD](https://github.com/niehen6174/LVMD)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

* [TensorRT](https://developer.nvidia.com/tensorrt)
* [tensorrtx](https://github.com/wang-xinyu/tensorrtx)
* [onnx-tensort](https://github.com/onnx/onnx-tensorrt)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/niehen6174/LVMD.svg?style=for-the-badge
[contributors-url]: https://github.com/niehen6174/LVMD/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/niehen6174/LVMD.svg?style=for-the-badge
[forks-url]: https://github.com/niehen6174/LVMD/network/members
[stars-shield]: https://img.shields.io/github/stars/niehen6174/LVMD.svg?style=for-the-badge
[stars-url]: https://github.com/niehen6174/LVMD/stargazers
[issues-shield]: https://img.shields.io/github/issues/niehen6174/LVMD.svg?style=for-the-badge
[issues-url]: https://github.com/niehen6174/LVMD/issues
[license-shield]: https://img.shields.io/github/license/niehen6174/LVMD.svg?style=for-the-badge
[license-url]: https://github.com/niehen6174/LVMD/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
