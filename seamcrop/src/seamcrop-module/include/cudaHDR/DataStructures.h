#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H

#include "types/MocaTypes.h"

template <class T>
struct Cuda2DPitchedMemoryDataDescriptor
{
	//Data pointer
	T*					d_pointer;

	//Values are needed to describe pitched memory
	size_t				d_pitch;		//Represents the pitch ("line length") used in device
	size_t				d_height;		//Represents the height in rows in the device memory
	size_t				d_width;		//Represents the width in bytes in the device memory
	size_t				element_size;	//Bytes per stored element (normally sizeof(T))
};

template <class T>
struct CudaImageHandleDataDescriptor : public Cuda2DPitchedMemoryDataDescriptor<T>
{
	//Data about the device memory
	unsigned int		d_stride;		//Represents the padding of every in channels
	unsigned int		channels;		//Number of channels
};

typedef CudaImageHandleDataDescriptor<unsigned char> CudaImage8UDataDescriptor;
typedef CudaImageHandleDataDescriptor<int16> CudaImage16SDataDescriptor;
typedef CudaImageHandleDataDescriptor<float> CudaImage32FDataDescriptor;

struct CudaExposureHandleDataDescriptor : public CudaImageHandleDataDescriptor<unsigned char>
{
	//Data about the device memory
	int 		x_offset;		//Represents the padding of every pixel
	int			y_offset;		//Represents the width in bytes in the device memory
	float		shutter;		//Number of channels
	int			parent;			//Parent frame
};

template <class T = unsigned int>
struct CudaVectorHandleDataDescriptor
{
	//Data pointer
	T*					d_pointer;		//Points to the beginning of the histogram

	//Data about the device memory
	unsigned int		numelements;	//The number of elements contained in this Histogram
};

typedef CudaVectorHandleDataDescriptor<CudaExposureHandleDataDescriptor> CudaExposureVectorHandleDataDescriptor;

//NCC type
enum cudaErrorCode
{
	success               = 0, //Everything went fine
	kernelLaunchFailure   = 1, //Kernel could not be launched
	dimMismatch           = 2, //Height or width of input data are wrong
	chMismatch            = 3,  //Number of channels of input or output data are wrong
	vecMismatch           = 4   //An input vector does not match the kernel requirements
};

#endif
