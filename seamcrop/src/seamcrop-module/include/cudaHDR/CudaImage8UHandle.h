#ifndef CUDAIMAGE8UHANDLE_H
#define CUDAIMAGE8UHANDLE_H

#include "types/Image8U.h"
#include "types/MocaException.h"
#include "driver_types.h"
#include "cuda_runtime.h"
#include "cudaHDR/DataStructures.h"
#include "cudaHDR/CudaImageHandle.h"

class CudaImage8UHandle : public CudaImageHandle<unsigned char>
{
 public:
  //Standard constructor
  CudaImage8UHandle() {}

 CudaImage8UHandle(const CudaImage8UHandle& rhs) :
  CudaImageHandle<unsigned char>(rhs) { }

  CudaImage8UHandle& operator=(const CudaImage8UHandle& rhs) 
    {
      //Upcall operator
      CudaImageHandle<unsigned char>::operator=(rhs);
      return *this;
    }

  void put(Image8U const& img, cudaStream_t stream = 0)
  {
    CudaImageHandle<unsigned char>::put(img.ptr(), img.widthStep(), img.width(), img.height(), img.channels(), img.channels(), stream);
  }

  void getImage8UData(Image8U& img, cudaStream_t stream = 0)
  {
    //Does downloaded data fit into Image8U?
    if (d_pointer != NULL && 
        (unsigned int) img.width() == d_width/d_stride && 
        (unsigned int) img.height() == d_height &&
        (unsigned int) img.channels() == d_stride)
      {
        CudaImageHandle<unsigned char>::getData(img.ptr(), img.widthStep(), stream);
      }
    else
      BOOST_THROW_EXCEPTION(MocaException("Image size does not fit data size on device!"));
  }

  //Setter/Getter
  const CudaImage8UDataDescriptor* getDataDescPtr(void) const
  {
    return (CudaImage8UDataDescriptor*) CudaImageHandle<unsigned char>::getDataDescPtr(); 
  }
};


#endif
