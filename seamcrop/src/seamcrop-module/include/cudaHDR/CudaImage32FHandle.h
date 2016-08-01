#ifndef CUDAIMAGE32FHANDLE_H
#define CUDAIMAGE32FHANDLE_H

#include "types/Image32F.h"
#include "types/MocaException.h"
#include "driver_types.h"
#include "cuda_runtime.h"
#include "cudaHDR/DataStructures.h"
#include "cudaHDR/CudaImageHandle.h"

class CudaImage32FHandle : public CudaImageHandle<float>
{
 public:
  //Standard constructor
  CudaImage32FHandle() {}

 CudaImage32FHandle(const CudaImage32FHandle& rhs)
   : CudaImageHandle<float>(rhs) { }

  CudaImage32FHandle& operator=(const CudaImage32FHandle& rhs) 
    {
      //Upcall operator
      CudaImageHandle<float>::operator=(rhs);
      return *this;
    }

  void put(Image32F const& img, cudaStream_t stream = 0)
  {
    CudaImageHandle<float>::put((float const*)img.ptr(), img.widthStep(), img.width(), img.height(), img.channels(), img.channels(), stream);
  }

  void getImage32FData(Image32F& img, cudaStream_t stream = 0)
  {
    //Does downloaded data fit into Image32F
    if (d_pointer != NULL && 
        (unsigned int) img.width() == d_width/(d_stride * sizeof(float)) && 
        (unsigned int) img.height() == d_height &&
        (unsigned int) img.channels() == d_stride)
      {
        CudaImageHandle<float>::getData((float*) img.ptr(), img.widthStep(), stream);
      }
    else
      BOOST_THROW_EXCEPTION(MocaException("Image size does not fit data size on device!"));
  }

  //Setter/Getter
  const CudaImage32FDataDescriptor* getDataDescPtr(void) const
  {
    return (CudaImage32FDataDescriptor*) CudaImageHandle<float>::getDataDescPtr(); 
  }
};


#endif
