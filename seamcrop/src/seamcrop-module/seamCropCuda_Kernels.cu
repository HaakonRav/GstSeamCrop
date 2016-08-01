#include "seamCropCuda_Kernels.h"

#define BLOCK_SIZE_1D 256 // Cuda threads per block
#define BLOCKS_1D(x) ((x + BLOCK_SIZE_1D-1) / BLOCK_SIZE_1D)

#define BLOCK_SIZE_2D_SQR 16 // Cuda threads per block per dimension (for square blocks)
#define BLOCKS_2D_SQR(x) ((x + BLOCK_SIZE_2D_SQR-1) / BLOCK_SIZE_2D_SQR)

#define BLOCK_SIZE_2D_HOR 32 // Cuda threads per block (horizontal)
#define BLOCK_SIZE_2D_VER  8 // Cuda threads per block (vertical)
#define BLOCKS_2D_HOR(x) ((x + BLOCK_SIZE_2D_HOR-1) / BLOCK_SIZE_2D_HOR)
#define BLOCKS_2D_VER(x) ((x + BLOCK_SIZE_2D_VER-1) / BLOCK_SIZE_2D_VER)

dim3 const threads2Dsqr(BLOCK_SIZE_2D_SQR, BLOCK_SIZE_2D_SQR);
dim3 const threads2D(BLOCK_SIZE_2D_HOR, BLOCK_SIZE_2D_VER);


#define MAX_GAUSS_KERNEL_SIZE 127

__constant__ float gaussKernel[MAX_GAUSS_KERNEL_SIZE];
__constant__ unsigned int gaussKernelSize;


inline __device__ unsigned char* getPixel8U(CudaImage8UDataDescriptor const& img, unsigned int const x, unsigned int const y, unsigned int const chan = 0)
{
  return (unsigned char*) ((char*)img.d_pointer + y*img.d_pitch) + (x*img.d_stride) + chan;
}


inline __device__ float* getPixel32F(CudaImage32FDataDescriptor const& img, unsigned int const x, unsigned int const y, unsigned int const chan = 0)
{
  return (float*) ((char*)img.d_pointer + y*img.d_pitch) + (x*img.d_stride) + chan;
}


inline __device__ void syncBlocks(unsigned int* syncVar, unsigned int syncMul)
{
  __syncthreads();

  if(threadIdx.x == 0)
  {
    atomicAdd(syncVar, 1);
    unsigned int syncVal = syncMul * gridDim.x;
    while(atomicCAS(syncVar, syncVal, syncVal) != syncVal); // ugly busy wait.
    // Note: the number of concurrent blocks is limited, i.e. the GPU will hang if the kernel launch specifies too many blocks!
  }

  __syncthreads();
}


__global__ void cuda_computeGradient_step2(CudaImage32FDataDescriptor energy, unsigned int* d_min, unsigned int* d_max)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int w = energy.d_width / sizeof(float); // energy.d_stride;
  unsigned int h = energy.d_height;

  if(x < w)
  {
    unsigned int minVal = (unsigned int)*getPixel32F(energy,x,0);
    unsigned int maxVal = minVal;

    for(unsigned int y = 1; y < h; ++y)
    {
      unsigned int val = (unsigned int)*getPixel32F(energy,x,y);
      if(val < minVal)
      {
        minVal = val;
      }
      if(val > maxVal)
      {
        maxVal = val;
      }
    } 

    atomicMin(d_min, minVal);
    atomicMax(d_max, maxVal);
  }
}


__global__ void cuda_computeGradient_step3(unsigned int* d_min, unsigned int* d_max, float* d_factor)
{
  *d_factor = 255.0f / (*d_max - *d_min);
}


__global__ void cuda_computeGradient_step4(CudaImage32FDataDescriptor energy, unsigned int* d_min, float* d_factor)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int w = energy.d_width / energy.d_stride;
  unsigned int h = energy.d_height;

  if(x < w && y < h)
  {
    float* pixel = getPixel32F(energy,x,y);
    *pixel = *d_factor * (*pixel - *d_min);
  }
}


__global__ void cuda_findMotionSaliency_step1(CudaImage8UDataDescriptor frame1, CudaImage8UDataDescriptor frame2, CudaImage8UDataDescriptor saliency)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int w = frame1.d_width / frame1.d_stride;
  unsigned int h = frame1.d_height;
  unsigned int chans = frame1.channels;

  if(x < w && y < h)
  {
    unsigned int diff = 0;
    unsigned char* frame1Data = getPixel8U(frame1,x,y);
    unsigned char* frame2Data = getPixel8U(frame2,x,y);

    for(unsigned int c = 0; c < chans; ++c)
      diff += abs(frame1Data[c] - frame2Data[c]);

    *getPixel8U(saliency,x,y) = diff / chans; //TODO: calculate max difference of all channels instead?
  }
}


__global__ void cuda_findMotionSaliency_step2(CudaImage8UDataDescriptor saliency, unsigned int* d_max)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int w = saliency.d_width / saliency.d_stride;
  unsigned int h = saliency.d_height;

  if(x < w)
  {
    unsigned int maxVal = *getPixel8U(saliency,x,0);

    for(unsigned int y = 1; y < h; ++y)
    {
      unsigned int val = *getPixel8U(saliency,x,y);
      
      if(val > maxVal)
        maxVal = val;
    }

    atomicAdd(d_max, maxVal);
  }
}


__global__ void cuda_findMotionSaliency_step3(unsigned int* d_max, float* averageMax, unsigned int w)
{
  *averageMax = ((float)*d_max/w)*0.25f; //TODO: adjust because of the fix to the difference calculation?
}


__global__ void cuda_findMotionSaliency_step4(CudaImage8UDataDescriptor saliency, float *averageMax)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int w = saliency.d_width / saliency.d_stride;
  unsigned int h = saliency.d_height;

  if(x < w && y < h)
  {
    unsigned char* pixel = getPixel8U(saliency,x,y);
    *pixel = *pixel > *averageMax ? 255 : 0;
  }
}


__global__ void cuda_smooth(CudaImage8UDataDescriptor src, CudaImage8UDataDescriptor dst, unsigned int pitch)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int w = src.d_width / src.d_stride;
  int h = src.d_height;

  __shared__ unsigned char buffer[BLOCK_SIZE_2D_SQR][BLOCK_SIZE_2D_SQR];
  extern __shared__ unsigned char image[];

  float val = 0.0f;

  int offset = (gaussKernelSize-1) / 2;
  unsigned char* line = &image[threadIdx.y*pitch];

  line[threadIdx.x+offset] = *getPixel8U(src,min(w-1,x),y);
  if(threadIdx.x < offset) //Note: assumes offset < BLOCK_SIZE_2D_SQR
  {
    line[threadIdx.x] = *getPixel8U(src,max(0,min(w-1,x-offset)),y);
    line[threadIdx.x+BLOCK_SIZE_2D_SQR+offset] = *getPixel8U(src,min(w-1,x+BLOCK_SIZE_2D_SQR),y);
  }

  __syncthreads();

  for(int z = 0; z < gaussKernelSize; ++z)
    val += gaussKernel[z]*line[threadIdx.x+z];

  buffer[threadIdx.y][threadIdx.x] = val;

  __syncthreads();

  x = blockIdx.y * blockDim.x + threadIdx.x;
  y = blockIdx.x * blockDim.y + threadIdx.y;

  if(x < h && y < w)
    *getPixel8U(dst,x,y) = buffer[threadIdx.x][threadIdx.y];
}


__global__ void cuda_mergeSaliency(CudaImage8UDataDescriptor motionSaliency, CudaImage32FDataDescriptor gradient, CudaImage32FDataDescriptor result)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int w = motionSaliency.d_width / motionSaliency.d_stride;
  unsigned int h = motionSaliency.d_height;

  if(x < w && y < h)
  {
    *getPixel32F(result,x,y) = 2*(*getPixel8U(motionSaliency,x,y)) + 1*(*getPixel32F(gradient,x,y));
  }
}


__global__ void cuda_addColumnCost(CudaImage32FDataDescriptor frame, CudaImage32FDataDescriptor columnCost, unsigned int t)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int w = frame.d_width / sizeof(float); // frame.d_stride;
  unsigned int h = frame.d_height;

  if(x < w)
  {
    float val = *getPixel32F(frame,x,0);
    for(unsigned int y = 1; y < h; ++y)
      val += *getPixel32F(frame,x,y);

    *getPixel32F(columnCost,x,t) = val;
  }
}


__global__ void cuda_calculateCostCroppingWindowTime(CudaImage32FDataDescriptor columnCost, CudaImage32FDataDescriptor croppingWindow, unsigned int tw)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int ow = croppingWindow.d_width / sizeof(float); // croppingWindow.d_stride;
  unsigned int h = columnCost.d_height;

  if(x < ow && y < h)
  {
    float val = *getPixel32F(columnCost,x,y);
    for(unsigned int i = 1; i < tw; ++i)
      val += *getPixel32F(columnCost,x+i,y);

    *getPixel32F(croppingWindow,x,y) = val;
  }
}


__global__ void cuda_calculateMaxEnergyPath_step1(CudaImage32FDataDescriptor croppingWindow, CudaImage32FDataDescriptor predecessors, unsigned int* blockCnt)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int w = croppingWindow.d_width / sizeof(float); // croppingWindow.d_stride;
  unsigned int h = croppingWindow.d_height;

  if(x < w)
  {
    unsigned int xl = x - 1;
    if(x == 0)
      xl = 0;

    unsigned int xr = x + 1;    
    if(x == w-1)
      xr = w-1;

    float oldVal = *getPixel32F(croppingWindow,x,0);

    for(unsigned int y = 1; y < h; ++y)
    {
      float* enPos = getPixel32F(croppingWindow,x,y);
      float enVal = *enPos;

      float v1 = enVal + *getPixel32F(croppingWindow,xl,y-1);
      float v2 = enVal + oldVal;
      float v3 = enVal + *getPixel32F(croppingWindow,xr,y-1);

      float* predPos = getPixel32F(predecessors,x,y);
      if(v2 >= v1 && v2 >= v3)
      {
        oldVal = v2;
        *predPos = x;
      }
      else if(v1 >= v3)
      {
        oldVal = v1;
        *predPos = xl;
      }
      else
      {
        oldVal = v3;
        *predPos = xr;
      }
      *enPos = oldVal;

      syncBlocks(blockCnt, y);
    }
  }
}


__global__ void cuda_calculateMaxEnergyPath_step2(CudaImage32FDataDescriptor croppingWindow, CudaImage32FDataDescriptor predecessors, CudaVectorHandleDataDescriptor<unsigned int> cropLeft)
{
  unsigned int w = croppingWindow.d_width / sizeof(float); // optimalCost.d_stride;
  unsigned int h = croppingWindow.d_height;

  float maxVal = *getPixel32F(croppingWindow,0,h-1);
  unsigned int maxPos = 0;

  for(unsigned int x = 1; x < w; ++x)
  {
    float val = *getPixel32F(croppingWindow,x,h-1);
    if(val > maxVal)
    {
      maxVal = val;
      maxPos = x;
    }
  }

  for(unsigned int y = h; y > 0; --y)
  {
    cropLeft.d_pointer[y-1] = maxPos;
    maxPos = (unsigned int)*getPixel32F(predecessors,maxPos,y-1);
  }
}


__global__ void cuda_cropImage8U(CudaImage8UDataDescriptor src, CudaImage8UDataDescriptor dst, unsigned int cropLeft)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int w = dst.d_width / dst.d_stride;
  unsigned int h = dst.d_height;
  unsigned int chans = dst.channels;

  if(x < w && y < h)
  {
    unsigned char* dstPx = getPixel8U(dst,x,y);
    unsigned char* srcPx = getPixel8U(src,x+cropLeft,y);

    for(unsigned int c = 0; c < chans; ++c)
      dstPx[c] = srcPx[c];
  }
}

  
__global__ void cuda_cropImage32F(CudaImage32FDataDescriptor src, CudaImage32FDataDescriptor dst, unsigned int cropLeft)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int w = dst.d_width / sizeof(float); //src.d_stride;
  unsigned int h = dst.d_height;

  if(x < w && y < h)
  {
    *getPixel32F(dst,x,y) = *getPixel32F(src,x+cropLeft,y);
  }
}


__global__ void cuda_addTemporalCoherenceCost(CudaImage32FDataDescriptor energy, CudaImage32FDataDescriptor tmpEnergy, CudaImage32FDataDescriptor previousSeams, unsigned int seamID, int cropOffset, unsigned int* count)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int w = energy.d_width / sizeof(float);
  int h = energy.d_height;

  if(x < w && y < h)
  {
    int prevSeamPos = *getPixel32F(previousSeams,seamID,y) + cropOffset;
    int temporalCoherenceCost = min(30*abs(prevSeamPos-x), 300);

    if(prevSeamPos < 0 || prevSeamPos >= w)
    {
      temporalCoherenceCost = 300;
      if(x == 0)
        atomicAdd(count, 1);
    }

    *getPixel32F(tmpEnergy,x,y) = *getPixel32F(energy,x,y) + temporalCoherenceCost;    
  }
}


__global__ void cuda_computeCostWidth(CudaImage32FDataDescriptor energy, CudaImage32FDataDescriptor fwdEnergy, CudaImage32FDataDescriptor costWidth, CudaImage32FDataDescriptor predecessors, unsigned int* blockCnt)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int wx = threadIdx.x & 31; // ID of this thread in the current warp
  unsigned int mask = 1 << wx;

  unsigned int w = energy.d_width / sizeof(float); // energy.d_stride;
  unsigned int h = energy.d_height;

  if(x < w)
  {
    unsigned int xl = x - 1;
    if(x == 0)
      xl = 0;

    unsigned int xr = x + 1;
    if(x == w-1)
      xr = w-1;

    float oldVal = *getPixel32F(energy,x,0);
    *getPixel32F(costWidth,x,0) = oldVal;

    syncBlocks(blockCnt, 1);

    for(unsigned int y = 1; y < h; ++y)
    {
      float enVal = *getPixel32F(energy,x,y);
      float* fwdEn = getPixel32F(fwdEnergy,3*x,y);

      float v1 = *getPixel32F(costWidth,xl,y-1) + enVal + fwdEn[0];
      float v2 =  oldVal                        + enVal + fwdEn[1];
      float v3 = *getPixel32F(costWidth,xr,y-1) + enVal + fwdEn[2];

      oldVal = min(v1,min(v2,v3));
      *getPixel32F(costWidth,x,y) = oldVal;

      int pos = x;
      pos -= (__ballot(v1 < v2 && v1 <= v3) & mask) >> wx;
      pos += (__ballot(v3 < v2 && v3 <  v1) & mask) >> wx;
      *getPixel32F(predecessors,x,y) = pos;

      syncBlocks(blockCnt, y+1);
    }
  }
}


__global__ void cuda_markSeamWidth(CudaImage32FDataDescriptor costWidth, CudaImage32FDataDescriptor energy, CudaImage32FDataDescriptor predecessors, CudaImage32FDataDescriptor seams, unsigned int seamID, unsigned int* blockCnt)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int bx = blockIdx.x;
  unsigned int tx = threadIdx.x;

  unsigned int w = energy.d_width / sizeof(float); // energy.d_stride;
  unsigned int h = energy.d_height;

  __shared__ float cost[BLOCK_SIZE_1D];
  __shared__ unsigned int pos[BLOCK_SIZE_1D];

  cost[tx] = MOCA_FLOAT_MAX;
  pos[tx] = x;

  if(x < w)
    cost[tx] = *getPixel32F(costWidth,x,h-1);

  __syncthreads();

  unsigned int offset = 1;
  while(offset < BLOCK_SIZE_1D)
  {
    unsigned int nextOffset = offset * 2;
    if(tx % nextOffset == 0)
    {
      unsigned int tx2 = tx + offset;
      if(cost[tx2] < cost[tx])
      {
        cost[tx] = cost[tx2];
        pos[tx] = pos[tx2];
      }
    }

    offset = nextOffset;
    __syncthreads();
  }

  if(tx == 0)
  {
    float* dst = getPixel32F(costWidth,bx*2,h-1);
    dst[0] = cost[0];
    dst[1] = pos[0];
  }

  syncBlocks(blockCnt, 1);

  if(x == 0)
  {
    unsigned int minPos = pos[0];
    for(unsigned int i = 1; i < gridDim.x; ++i)
    {
      float val = *getPixel32F(costWidth,i*2,h-1);
      if(val < cost[0])
      {
        cost[0] = val;
        minPos = *getPixel32F(costWidth,i*2+1,h-1);
      }
    }

    for(unsigned int y = h; y > 0; --y)
    {
      *getPixel32F(energy,minPos,y-1) = 99999999.0f;
      *getPixel32F(seams,seamID,y-1) = minPos;
      minPos = (unsigned int)*getPixel32F(predecessors,minPos,y-1);
    }
  }
}


__global__ void cuda_removeSeams(CudaImage8UDataDescriptor src, CudaImage8UDataDescriptor dst, CudaImage32FDataDescriptor seams)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int ew = src.d_width / src.d_stride;
  unsigned int tw = dst.d_width / dst.d_stride;
  unsigned int sw = seams.d_width / sizeof(float); // seams.d_stride;
  unsigned int h = src.d_height;
  unsigned int chans = src.channels;

  if(x < ew && y < h)
  {
    float* seamsLine = getPixel32F(seams,0,y);

    int offset = 0;
    for(unsigned int n = 0; n < sw; ++n)
    {
      unsigned int seamPos = (unsigned int)seamsLine[n];
      if(seamPos == x)
        offset = MOCA_INT32_MAX / 2;
      if(seamPos < x)
        offset -= 1;
    }

    unsigned int dstPos = x + offset;
    if(dstPos < tw)
    {
      unsigned char* dstPx = getPixel8U(dst,dstPos,y);
      unsigned char* srcPx = getPixel8U(src,x,y);
      for(unsigned int c = 0; c < chans; ++c)
        dstPx[c] = srcPx[c];
    }
  }
}


#define THREAD_ID 0
#include "seamCropCuda_KernelsTex.cu"
#undef THREAD_ID

#define THREAD_ID 1
#include "seamCropCuda_KernelsTex.cu"
#undef THREAD_ID

#define THREAD_ID 2
#include "seamCropCuda_KernelsTex.cu"
#undef THREAD_ID

#define THREAD_ID 3
#include "seamCropCuda_KernelsTex.cu"
#undef THREAD_ID

#define THREAD_ID 4
#include "seamCropCuda_KernelsTex.cu"
#undef THREAD_ID

#define THREAD_ID 5
#include "seamCropCuda_KernelsTex.cu"
#undef THREAD_ID

#define THREAD_ID 6
#include "seamCropCuda_KernelsTex.cu"
#undef THREAD_ID

#define THREAD_ID 7
#include "seamCropCuda_KernelsTex.cu"
#undef THREAD_ID


//##################################################################################|
//wrapper functions								    |
//##################################################################################|

void SeamCropCuda::computeGradient(CudaImage8UDataDescriptor const& image, CudaImage32FDataDescriptor const& energy)
{
  computeGradientFunc(image, energy, d_min, d_max, averageMax, stream);
}


void SeamCropCuda::findMotionSaliency(CudaImage8UDataDescriptor const& frame1, CudaImage8UDataDescriptor const& frame2, CudaImage8UDataDescriptor const& saliency)
{
  unsigned int w = frame1.d_width / frame1.d_stride;
  unsigned int h = frame1.d_height;

  dim3 blocks(BLOCKS_2D_HOR(w), BLOCKS_2D_VER(h));

  cudaMemsetAsync(d_max, 0, sizeof(unsigned int), *stream);

  cuda_findMotionSaliency_step1<<<blocks, threads2D, 0, *stream>>>(frame1, frame2, saliency);
  cuda_findMotionSaliency_step2<<<BLOCKS_1D(w), BLOCK_SIZE_1D, 0, *stream>>>(saliency, d_max);
  cuda_findMotionSaliency_step3<<<1, 1, 0, *stream>>>(d_max, averageMax, w);
  cuda_findMotionSaliency_step4<<<blocks, threads2D, 0, *stream>>>(saliency, averageMax);
}


void SeamCropCuda::smooth(CudaImage8UDataDescriptor const& image)
{
  unsigned int w = image.d_width / image.d_stride;
  unsigned int h = image.d_height;

  dim3 blocks1(BLOCKS_2D_SQR(w), BLOCKS_2D_SQR(h));
  dim3 blocks2(BLOCKS_2D_SQR(h), BLOCKS_2D_SQR(w));

  unsigned int pitch = BLOCK_SIZE_2D_SQR + gaussKernelSize_CPU - 1;
  unsigned int dynSharedMem = pitch * BLOCK_SIZE_2D_SQR;

  cuda_smooth<<<blocks1, threads2Dsqr, dynSharedMem, *stream>>>(image, *smoothTmpData, pitch);
  cuda_smooth<<<blocks2, threads2Dsqr, dynSharedMem, *stream>>>(*smoothTmpData, image, pitch);
}


void SeamCropCuda::mergeSaliency(CudaImage8UDataDescriptor const& motionSaliency, CudaImage32FDataDescriptor const& gradient, CudaImage32FDataDescriptor const& result)
{
  unsigned int w = motionSaliency.d_width / motionSaliency.d_stride;
  unsigned int h = motionSaliency.d_height;

  dim3 blocks(BLOCKS_2D_HOR(w), BLOCKS_2D_VER(h));

  cuda_mergeSaliency<<<blocks, threads2D, 0, *stream>>>(motionSaliency, gradient, result);
}


void SeamCropCuda::addColumnCost(CudaImage32FDataDescriptor const& frame, CudaImage32FDataDescriptor const& columnCost, unsigned int t)
{
  unsigned int w = frame.d_width / sizeof(float); // frame.d_stride;

  cuda_addColumnCost<<<BLOCKS_1D(w), BLOCK_SIZE_1D, 0, *stream>>>(frame, columnCost, t);
}


void SeamCropCuda::calculateCostCroppingWindowTime(CudaImage32FDataDescriptor const& columnCost, CudaImage32FDataDescriptor const& croppingWindow, unsigned int tw)
{
  unsigned int ow = croppingWindow.d_width / sizeof(float); // croppingWindow.d_stride;
  unsigned int h = columnCost.d_height;

  dim3 blocks(BLOCKS_2D_HOR(ow), BLOCKS_2D_VER(h));

  cuda_calculateCostCroppingWindowTime<<<blocks, threads2D>>>(columnCost, croppingWindow, tw);
}


void SeamCropCuda::calculateMaxEnergyPath(CudaImage32FDataDescriptor const& croppingWindow, CudaImage32FDataDescriptor const& predecessors, CudaVectorHandleDataDescriptor<unsigned int> const& cropLeft)
{
  unsigned int w = croppingWindow.d_width / sizeof(float); // croppingWindow.d_stride;

  cudaMemset(staticBlockCnt, 0, sizeof(unsigned int));

  cuda_calculateMaxEnergyPath_step1<<<BLOCKS_1D(w), BLOCK_SIZE_1D>>>(croppingWindow, predecessors, staticBlockCnt);
  cuda_calculateMaxEnergyPath_step2<<<1, 1>>>(croppingWindow, predecessors, cropLeft);
}


void SeamCropCuda::cropImage8U(CudaImage8UDataDescriptor const& src, CudaImage8UDataDescriptor const& dst, unsigned int cropLeft)
{
  unsigned int w = dst.d_width / dst.d_stride;
  unsigned int h = dst.d_height;

  dim3 blocks(BLOCKS_2D_HOR(w), BLOCKS_2D_VER(h));

  cuda_cropImage8U<<<blocks, threads2D, 0, *stream>>>(src, dst, cropLeft);
}


void SeamCropCuda::cropImage32F(CudaImage32FDataDescriptor const& src, CudaImage32FDataDescriptor const& dst, unsigned int cropLeft)
{
  unsigned int w = dst.d_width / sizeof(float); // src.d_stride;
  unsigned int h = dst.d_height;

  dim3 blocks(BLOCKS_2D_HOR(w), BLOCKS_2D_VER(h));

  cuda_cropImage32F<<<blocks, threads2D, 0, *stream>>>(src, dst, cropLeft);
}


void SeamCropCuda::computeForwardEnergy(CudaImage8UDataDescriptor const& image, CudaImage32FDataDescriptor const& fwdEnergy)
{
  computeFwdEnergyFunc(image, fwdEnergy, stream);
}


bool SeamCropCuda::addTemporalCoherenceCost(CudaImage32FDataDescriptor const& energy, CudaImage32FDataDescriptor const& tmpEnergy, CudaImage32FDataDescriptor const& previousSeams, unsigned int seamID, int cropOffset)
{
  unsigned int w = energy.d_width / sizeof(float); //TODO: energy.d_stride set incorrectly in CudaImage32FHandle?
  unsigned int h = energy.d_height;

  dim3 blocks(BLOCKS_2D_HOR(w), BLOCKS_2D_VER(h));

  *count = 0;
  cuda_addTemporalCoherenceCost<<<blocks, threads2D, 0, *stream>>>(energy, tmpEnergy, previousSeams, seamID, cropOffset, count);
  cudaStreamSynchronize(*stream);

  return (*count > h*0.2);
}


void SeamCropCuda::computeCostWidth(CudaImage32FDataDescriptor const& energy, CudaImage32FDataDescriptor const& fwdEnergy, CudaImage32FDataDescriptor const& costWidth, CudaImage32FDataDescriptor const& predecessors)
{
  unsigned int w = energy.d_width / sizeof(float); // energy.d_stride;

  cudaMemsetAsync(blockCnt, 0, sizeof(unsigned int), *stream);

  cuda_computeCostWidth<<<BLOCKS_1D(w), BLOCK_SIZE_1D, 0, *stream>>>(energy, fwdEnergy, costWidth, predecessors, blockCnt);
}


void SeamCropCuda::markSeamWidth(CudaImage32FDataDescriptor const& costWidth, CudaImage32FDataDescriptor const& energy, CudaImage32FDataDescriptor const& predecessors, CudaImage32FDataDescriptor const& seams, unsigned int seamID)
{
  unsigned int w = energy.d_width / sizeof(float); // energy.d_stride;

  cudaMemsetAsync(blockCnt, 0, sizeof(unsigned int), *stream);

  cuda_markSeamWidth<<<BLOCKS_1D(w), BLOCK_SIZE_1D, 0, *stream>>>(costWidth, energy, predecessors, seams, seamID, blockCnt);
}


void SeamCropCuda::removeSeams(CudaImage8UDataDescriptor const& src, CudaImage8UDataDescriptor const& dst, CudaImage32FDataDescriptor const& seams)
{
  unsigned int w = src.d_width / src.d_stride;
  unsigned int h = src.d_height;

  dim3 blocks(BLOCKS_2D_HOR(w), BLOCKS_2D_VER(h));

  cuda_removeSeams<<<blocks, threads2D, 0, *stream>>>(src, dst, seams);
}


void SeamCropCuda::setupFunctionPointerMaps()
{
  computeGradientFuncPtrMap[0] = cudaWrapper_computeGradient_thread0;
  computeGradientFuncPtrMap[1] = cudaWrapper_computeGradient_thread1;
  computeGradientFuncPtrMap[2] = cudaWrapper_computeGradient_thread2;
  computeGradientFuncPtrMap[3] = cudaWrapper_computeGradient_thread3;
  computeGradientFuncPtrMap[4] = cudaWrapper_computeGradient_thread4;
  computeGradientFuncPtrMap[5] = cudaWrapper_computeGradient_thread5;
  computeGradientFuncPtrMap[6] = cudaWrapper_computeGradient_thread6;
  computeGradientFuncPtrMap[7] = cudaWrapper_computeGradient_thread7;

  computeFwdEnergyFuncPtrMap[0] = cudaWrapper_computeForwardEnergy_thread0;
  computeFwdEnergyFuncPtrMap[1] = cudaWrapper_computeForwardEnergy_thread1;
  computeFwdEnergyFuncPtrMap[2] = cudaWrapper_computeForwardEnergy_thread2;
  computeFwdEnergyFuncPtrMap[3] = cudaWrapper_computeForwardEnergy_thread3;
  computeFwdEnergyFuncPtrMap[4] = cudaWrapper_computeForwardEnergy_thread4;
  computeFwdEnergyFuncPtrMap[5] = cudaWrapper_computeForwardEnergy_thread5;
  computeFwdEnergyFuncPtrMap[6] = cudaWrapper_computeForwardEnergy_thread6;
  computeFwdEnergyFuncPtrMap[7] = cudaWrapper_computeForwardEnergy_thread7;
}


bool SeamCropCuda::uploadGaussKernel(float* kernel, unsigned int size)
{
  if(size > MAX_GAUSS_KERNEL_SIZE)
    return false;

  cudaMemcpyToSymbol(gaussKernel, kernel, size*sizeof(float));
  cudaMemcpyToSymbol(gaussKernelSize, &size, sizeof(unsigned int));
  gaussKernelSize_CPU = size;

  return true;
}

