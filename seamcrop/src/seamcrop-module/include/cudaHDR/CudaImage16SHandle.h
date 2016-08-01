#ifndef CUDAIMAGE16SHANDLE_H
#define CUDAIMAGE16SHANDLE_H

#include "types/Image16S.h"
#include "types/MocaException.h"
#include "driver_types.h"
#include "cuda_runtime.h"
#include "cudaHDR/DataStructures.h"
#include "cudaHDR/CudaImageHandle.h"

class CudaImage16SHandle : public CudaImageHandle<int16>
{
 public:
  //Standard constructor
  CudaImage16SHandle() {}

 CudaImage16SHandle(const CudaImage16SHandle& rhs)
   : CudaImageHandle<int16>(rhs) {}

  CudaImage16SHandle& operator=(const CudaImage16SHandle& rhs) 
    {
      //Upcall operator
      CudaImageHandle<int16>::operator=(rhs);
      return *this;
    }

  void put(Image16S const& img, cudaStream_t stream = 0)
  {
    CudaImageHandle<int16>::put((int16 const*)img.ptr(), img.widthStep(), img.width(), img.height(), img.channels(), img.channels(), stream);
  }

  void getImage16SData(Image16S& img, cudaStream_t stream = 0)
  {
    //Does downloaded data fit into Image16S?
    if (d_pointer != NULL && 
        (unsigned int) img.width() == d_width/d_stride && 
        (unsigned int) img.height() == d_height &&
        (unsigned int) img.channels() == d_stride)
      {
        CudaImageHandle<int16>::getData((int16*)img.ptr(), img.widthStep(), stream);
      }
    else
      BOOST_THROW_EXCEPTION(MocaException("Image size does not fit data size on device!"));
  }

  //Setter/Getter
  const CudaImage16SDataDescriptor* getDataDescPtr(void) const
  {
    return (CudaImage16SDataDescriptor*) CudaImageHandle<int16>::getDataDescPtr(); 
  }
};


#endif
