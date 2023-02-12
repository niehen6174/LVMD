#include "gridSampler.h"
#include "gridSampler.cuh"
#include <stdio.h>
#include <assert.h>

using half = __half;

// alignd with torch Macros.h
// CUDA_MAX_THREADS_PER_BLOCK is same for all architectures currently
constexpr unsigned int CUDA_MAX_THREADS_PER_BLOCK = 1024;
// CUDA_THREADS_PER_BLOCK_FALLBACK is the "canonical fallback" choice of block size.
// 256 is a good number for this fallback and should give good occupancy and
// versatility across all architectures.
constexpr unsigned int CUDA_THREADS_PER_BLOCK_FALLBACK = 256;

#define C10_MAX_THREADS_PER_BLOCK(val) (((val) <= CUDA_MAX_THREADS_PER_BLOCK) ? (val) : CUDA_THREADS_PER_BLOCK_FALLBACK)
#define C10_LAUNCH_BOUNDS_1(max_threads_per_block) __launch_bounds__((C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))))

// aligned with KernelUtils.h
#define CUDA_KERNEL_LOOP(i, n) \
  int64_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;                                \
  for (int i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=blockDim.x * gridDim.x, i=_i_n_d_e_x)


constexpr int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  assert(N > 0);
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename scalar_t>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void grid_sampler_2d_kernel(
    int nthreads,
    const scalar_t* inputPtr,
    const scalar_t* gridPtr,      
    scalar_t* const outputPtr,      
    int C,
    int inp_H,
    int inp_W,
    int out_H, // same as grid_H
    int out_W, // same as grid_W
    int inp_sN,
    int inp_sC,
    int inp_sH,
    int inp_sW,
    int grid_sN,//gridHeight*gridWidth*2
    int grid_sH, // gridWidth*gridHeight
    int grid_sW,// gridWidth
    int grid_sCoor,//1
    int out_sN, // gridWidth*gridHeight*InputChannel
    int out_sC, // gridWidth*gridHeight
    int out_sH, // gridWidth
    int out_sW, //1
    torch::detail::GridSamplerInterpolation interpolation_mode,
    torch::detail::GridSamplerPadding padding_mode,
    bool align_corners) {
// 输入参数 第一个顾名思义 后三个为 输入（input和grid） 输出指针
//紧接着四个为通道数输入 输出的 H W size
//inp_sN 及下面三个为  input的 NCHW 排序为了获取到 第几个N 几个C ....  其中inp_sN=C*inp_H*inp_W; inp_sC = inp_H*inp_W; inp_sH=inp_W; inp_sW=1;
//grid_sN 及下面三个为 grid的排序 不input不同 顺序为 NHW2 计算方法通input，
//同理out_sN yu inp_sN 类似
// 差值模式和padd模式  在头文件中用了枚举进行了定义
//nchw 与线性排列之间关系。linear_index = (n * C * H * W) + (c * H * W) + (h * W) + w
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % out_W;
    const int h = (index / out_W) % out_H;
    const int n = index / (out_H * out_W); // n是batchSize \
    
    const int grid_offset = n * grid_sN + h * grid_sW + w ;

    // get the corresponding input x, y co-ordinates from grid
    scalar_t ix = gridPtr[grid_offset];
    scalar_t iy = gridPtr[grid_offset + grid_sH];

    ix = ix + w;
    iy = iy + h;
    //ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
    //iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);

    if (interpolation_mode == torch::detail::GridSamplerInterpolation::Bilinear) {
      // get NE, NW, SE, SW pixel values from (x, y) 
      int ix_nw = static_cast<int>(floorf(ix)); //floorf 返回不大于 ix 的最大整数值
      int iy_nw = static_cast<int>(floorf(iy));  //北为N 南为S 西为 W 东为E
      int ix_ne = ix_nw + 1;                    // nw 为西北左上角 sw 为西南左下角 ne为东北右上角 se为东南 右下角
      int iy_ne = iy_nw;
      int ix_sw = ix_nw;
      int iy_sw = iy_nw + 1;
      int ix_se = ix_nw + 1;
      int iy_se = iy_nw + 1;

      // get surfaces to each neighbor:
      scalar_t nw = (static_cast<scalar_t>(ix_se) - ix)    * (static_cast<scalar_t>(iy_se) - iy);
      scalar_t ne = (ix    - ix_sw) * (static_cast<scalar_t>(iy_sw) - iy);
      scalar_t sw = (static_cast<scalar_t>(ix_ne) - ix)    * (iy    - iy_ne);
      scalar_t se = (ix    - ix_nw) * (iy    - iy_nw);
      //  计算该四个值为 双线性插值 化简后公式 可参考 https://www.cnblogs.com/wancy/p/15212604.html 最后化简后的公式

      // calculate bilinear weighted pixel value and set output pixel
      auto inp_ptr_NC = inputPtr + n * inp_sN;
      auto out_ptr_NCHW = outputPtr + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        *out_ptr_NCHW = static_cast<scalar_t>(0);
        if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_nw * inp_sH + ix_nw * inp_sW] * nw;
        }
        if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_ne * inp_sH + ix_ne * inp_sW] * ne;
        }
        if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_sw * inp_sH + ix_sw * inp_sW] * sw;
        }
        if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
          *out_ptr_NCHW += inp_ptr_NC[iy_se * inp_sH + ix_se * inp_sW] * se;
        }
      }
    } else if (interpolation_mode == torch::detail::GridSamplerInterpolation::Nearest) {
      int ix_nearest = static_cast<int>(roundf(ix));
      int iy_nearest = static_cast<int>(roundf(iy));

      // assign nearest neighor pixel value to output pixel
      auto inp_ptr_NC = inputPtr + n * inp_sN;
      auto out_ptr_NCHW = outputPtr + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCHW += out_sC) {
        if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
          *out_ptr_NCHW = inp_ptr_NC[iy_nearest * inp_sH + ix_nearest * inp_sW];
        } else {
          *out_ptr_NCHW = static_cast<scalar_t>(0);
        }
      }

    }
  }
}

// No shape checking needed here. See # NOTE [ grid_sampler Native Functions ].
int grid_sampler_2d_cuda(int batchSize, const void* inputPtr, const void* gridPtr,
  void* const outputPtr,
  int C,
  int inp_H,
  int inp_W,
  int out_H, // same as grid_H
  int out_W, // same as grid_W
  int inp_sN,
  int inp_sC,
  int inp_sH,
  int inp_sW,
  int grid_sN,
  int grid_sH,
  int grid_sW,
  int grid_sCoor,
  int out_sN,
  int out_sC,
  int out_sH,
  int out_sW,
  torch::detail::GridSamplerInterpolation interpolation_mode,
  torch::detail::GridSamplerPadding padding_mode,
  bool align_corners, torch::detail::GridSamplerDataType dataType, cudaStream_t stream)
{

  int count = out_H * out_W * batchSize; // will have inner loop over C
  if (count > 0) {
    if (dataType == torch::detail::GridSamplerDataType::GHALF)
    {
      grid_sampler_2d_kernel<half>
        <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
          count,
          static_cast<const half*>(inputPtr), static_cast<const half*>(gridPtr),
          static_cast<half*>(outputPtr),
          C, inp_H, inp_W, out_H, out_W, inp_sN, inp_sC, inp_sH, inp_sW,
          grid_sN, grid_sH, grid_sW, grid_sCoor, out_sN, out_sC, out_sH, out_sW,
          interpolation_mode,
          padding_mode,
          align_corners);
    }
    else
    {
      grid_sampler_2d_kernel<float>
        <<<GET_BLOCKS(count), CUDA_NUM_THREADS, 0, stream>>>(
          count,
          static_cast<const float*>(inputPtr), static_cast<const float*>(gridPtr),
          static_cast<float*>(outputPtr),
          C, inp_H, inp_W, out_H, out_W, inp_sN, inp_sC, inp_sH, inp_sW,
          grid_sN, grid_sH, grid_sW, grid_sCoor, out_sN, out_sC, out_sH, out_sW,
          interpolation_mode,
          padding_mode,
          align_corners);
    }
  }

  return cudaGetLastError() != cudaSuccess;
}

