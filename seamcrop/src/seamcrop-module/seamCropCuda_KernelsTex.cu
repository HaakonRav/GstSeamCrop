// using the TMU seems to help only with small videos (e.g. 320x180). Using the texture
// fetch function tex2D(...) doesn't seem to offer any benefit at all.

#define MAKE_TEXTURE_NAME1(x,y) x ## _thread ## y
#define MAKE_TEXTURE_NAME2(x,y) MAKE_TEXTURE_NAME1(x,y) // w/o this step the pre-processor produces incorrect names
#define TEX_NAME(x) MAKE_TEXTURE_NAME2(x,THREAD_ID)

#define MAKE_KERNEL_NAME1(x,y) cuda_ ## x ## _thread ## y
#define MAKE_KERNEL_NAME2(x,y) MAKE_KERNEL_NAME1(x,y)
#define KRNL_NAME(x) MAKE_KERNEL_NAME2(x,THREAD_ID)

#define MAKE_FUNC_NAME1(x,y) cudaWrapper_ ## x ## _thread ## y
#define MAKE_FUNC_NAME2(x,y) MAKE_FUNC_NAME1(x,y)
#define FUNC_NAME(x) MAKE_FUNC_NAME2(x,THREAD_ID)


texture<unsigned char, 2, cudaReadModeElementType> TEX_NAME(imageTex);


__global__ void KRNL_NAME(computeGradient_step1)(CudaImage8UDataDescriptor image, CudaImage32FDataDescriptor energy)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int w = image.d_width / image.d_stride;
  int h = image.d_height;
  int chans = image.channels;

  if(x < w && y < h)
  {
    unsigned int xl = x-1;
    if(x == 0)
      xl = 0;

    unsigned int xr = x+1;
    if(x == w-1)
      xr = w-1;

    unsigned int yt = y-1;
    if(y == 0)
      yt = 0;

    unsigned int yb = y+1;
    if(y == h-1)
      yb = h-1;

    unsigned int grad = 0;
    unsigned char* leftData = getPixel8U(image, xl, y);
    unsigned char* rightData = getPixel8U(image, xr, y);
    unsigned char* topData = getPixel8U(image, x, yt);
    unsigned char* bottomData = getPixel8U(image, x, yb);
    for(unsigned int c = 0; c < chans; ++c)
    {
      // accessing the data through the Texture Mapping Unit is not working correctly. Wrong coordinates?
      /*
      unsigned char leftData = tex2D(TEX_NAME(imageTex), xl+c, y);
      unsigned char rightData = tex2D(TEX_NAME(imageTex), xr+c, y);
      unsigned char topData = tex2D(TEX_NAME(imageTex), x+c, yt);
      unsigned char bottomData = tex2D(TEX_NAME(imageTex), x+c, yb);

      grad += abs(rightData - leftData) + abs(bottomData - topData);
      */
      grad += abs(rightData[c] - leftData[c]) + abs(bottomData[c] - topData[c]);
    }

    *getPixel32F(energy, x, y) = grad;
  }
}


__global__ void KRNL_NAME(computeForwardEnergy)(CudaImage8UDataDescriptor image, CudaImage32FDataDescriptor fwdEnergy)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

  unsigned int w = image.d_width / image.d_stride;
  unsigned int h = image.d_height;
  unsigned int chans = image.channels;
  float fChans = 1.0f / chans;

  if(x < w && y < h)
  {
    unsigned int xl = x - 1;
    if(x == 0)
      xl = 0;

    unsigned int xr = x + 1;
    if(x == w-1)
      xr = w-1;

    float Cl = 0.0f;
    float Cu = 0.0f;
    float Cr = 0.0f;

    unsigned char* rightData = getPixel8U(image,xr,y);
    unsigned char* leftData = getPixel8U(image,xl,y);
    unsigned char* topData = getPixel8U(image,x,y-1);

    for(unsigned int c = 0; c < chans; ++c)
    {
      float right = rightData[c];
      float left = leftData[c];
      float top = topData[c];

      float center = abs(right - left);
        
      Cu += center;
      Cl += center + abs(top - left);
      Cr += center + abs(top - right);
    }

    float* dst = getPixel32F(fwdEnergy,3*x,y);
    dst[0] = Cl * fChans;
    dst[1] = Cu * fChans;
    dst[2] = Cr * fChans;
  }
}


//##################################################################################|
//wrapper functions								    |
//##################################################################################|

void FUNC_NAME(computeGradient)(CudaImage8UDataDescriptor const& image, CudaImage32FDataDescriptor const& energy, unsigned int* d_min, unsigned int* d_max, float* averageMax, cudaStream_t* stream)
{
  unsigned int w = image.d_width / image.d_stride;
  unsigned int h = image.d_height;

  dim3 blocks(BLOCKS_2D_HOR(w), BLOCKS_2D_VER(h));

  cudaMemsetAsync(d_min, 255, sizeof(unsigned int), *stream);
  cudaMemsetAsync(d_max, 0, sizeof(unsigned int), *stream);

  // bind texture
  TEX_NAME(imageTex).addressMode[0] = cudaAddressModeClamp;
  TEX_NAME(imageTex).addressMode[1] = cudaAddressModeClamp;
  TEX_NAME(imageTex).filterMode = cudaFilterModePoint;
  TEX_NAME(imageTex).normalized = 0;

  cudaChannelFormatDesc chanDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
  cudaBindTexture2D(0, &TEX_NAME(imageTex), image.d_pointer, &chanDesc, image.d_width, h, image.d_pitch);

  KRNL_NAME(computeGradient_step1)<<<blocks, threads2D, 0, *stream>>>(image, energy);
  cuda_computeGradient_step2<<<BLOCKS_1D(w), BLOCK_SIZE_1D, 0, *stream>>>(energy, d_min, d_max);
  cuda_computeGradient_step3<<<1, 1, 0, *stream>>>(d_min, d_max, averageMax);
  cuda_computeGradient_step4<<<blocks, threads2D, 0, *stream>>>(energy, d_min, averageMax);
}


void FUNC_NAME(computeForwardEnergy)(CudaImage8UDataDescriptor const& image, CudaImage32FDataDescriptor const& fwdEnergy, cudaStream_t* stream)
{
  unsigned int w = image.d_width / image.d_stride;
  unsigned int h = image.d_height;

  dim3 blocks(BLOCKS_2D_HOR(w), BLOCKS_2D_VER(h-1));

  TEX_NAME(imageTex).addressMode[0] = cudaAddressModeClamp;
  TEX_NAME(imageTex).addressMode[1] = cudaAddressModeClamp;
  TEX_NAME(imageTex).filterMode = cudaFilterModePoint;
  TEX_NAME(imageTex).normalized = 0;

  cudaChannelFormatDesc chanDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
  cudaBindTexture2D(0, &TEX_NAME(imageTex), image.d_pointer, &chanDesc, image.d_width, h, image.d_pitch);

  KRNL_NAME(computeForwardEnergy)<<<blocks, threads2D, 0, *stream>>>(image, fwdEnergy);
}

