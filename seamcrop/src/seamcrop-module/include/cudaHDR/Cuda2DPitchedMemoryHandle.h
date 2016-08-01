#ifndef CUDA2DPITCHED_H
#define CUDA2DPITCHED_H

#include "types/MocaException.h"
#include "driver_types.h"
#include "cuda_runtime.h"
#include "cudaHDR/DataStructures.h"
#include <limits>

template <typename T>
class Cuda2DPitchedMemoryHandle
{
protected:
	//Describes image data
	Cuda2DPitchedMemoryDataDescriptor<T> _cpmdd;

	//Data pointer
	T*					d_pointer;

	//Values are needed to describe pitched memory
	size_t				d_pitch;		//Represents the pitch ("line length") used in device
	size_t				d_height;		//Represents the height in rows in the device memory
	size_t				d_width;		//Represents the width in bytes in the device memory
	size_t				element_size;	//Bytes per stored element (normally sizeof(T))

	//Update data descriptor
	void updateDataDescriptor()
	{
		_cpmdd.d_pointer    = d_pointer;
		_cpmdd.d_pitch      = d_pitch;
		_cpmdd.d_height     = d_height;
		_cpmdd.d_width      = d_width;
		_cpmdd.element_size = element_size;
	}

	//reallocates memory if needed
	//Width new is new width in BYTES and height new is new height in LINES!
	size_t				initial_height;			//Holds initially reserved memory height
	void reallocmem(size_t width_new, size_t height_new)
	{
		//If no reallocation is needed just adjust height
		if (d_pointer != NULL && width_new == d_width && height_new <= initial_height)
		{
			d_height = height_new;
		//Else reallocate memory
		} else {
			//Set new properties
			d_width			= width_new;
			d_height		= height_new;
			element_size	= sizeof(T);

			initial_height = height_new;

			//Free old pointer
			cudaFree(d_pointer);
			//Allocate new memory
			cudaError_t ret = cudaMallocPitch((void**) &d_pointer, &d_pitch, d_width, d_height);

			if (ret != cudaSuccess)
			{
				d_pointer = NULL;
				BOOST_THROW_EXCEPTION(MocaException(ret,"Could not allocate memory!"));
			}
		}
	}

	//Free the allocated memory
	inline void freemem()
	{
		cudaFree(d_pointer);
		d_pointer = NULL;
	}

public:
	//Standard constructors
	Cuda2DPitchedMemoryHandle() 
	{
		//Set pointer to NULL
		d_pointer = NULL;
		updateDataDescriptor();
	}

	//Copy constructor
	Cuda2DPitchedMemoryHandle(const Cuda2DPitchedMemoryHandle<T>& rhs)
	{
		//Initialize members
		d_pointer = NULL;

		//If right hand side has not allocated memory skip copy ...
		if (rhs.d_pointer != NULL)
		{
			//(re)allocate memory
			reallocmem(rhs.d_width, rhs.d_height);

			//copy data
			if (cudaMemcpy2D(d_pointer, d_pitch, rhs.d_pointer, rhs.d_pitch, d_width, d_height, cudaMemcpyDeviceToDevice) != cudaSuccess)
				BOOST_THROW_EXCEPTION(MocaException("Could not copy data to device!"));
		} else {
			d_pointer = NULL;
		}
		updateDataDescriptor();
	}

	//Operators
	virtual Cuda2DPitchedMemoryHandle& operator=(const Cuda2DPitchedMemoryHandle<T>& rhs)
	{
		//Avoid self assignment
		if (this != &rhs)
		{
			if (rhs.d_pointer != NULL)
			{
				//(re)allocate memory
				reallocmem(rhs.d_width, rhs.d_height);

				//copy data
				if (cudaMemcpy2D(d_pointer, d_pitch, rhs.d_pointer, rhs.d_pitch, d_width, d_height, cudaMemcpyDeviceToDevice) != cudaSuccess)
					BOOST_THROW_EXCEPTION(MocaException("Could not copy data on device!"));
			} else freemem();
		}
		updateDataDescriptor();
		return *this;
	}

	//Methods
	virtual void allocate(unsigned int width, unsigned int height)
	{
		//Allocate memory
		reallocmem(width * sizeof(T), height);
		updateDataDescriptor();
	}

	virtual void free()
	{
		//Free ressources
		freemem();
		updateDataDescriptor();
	}

	virtual void put(T const* data, int32 srcPitch, unsigned int width, unsigned int height, cudaStream_t stream = 0)
	{
		//(re)allocate memory
		reallocmem(width * sizeof(T), height);

		//Copy data to device
		if (cudaMemcpy2DAsync(d_pointer, d_pitch, data, srcPitch, d_width, d_height, cudaMemcpyHostToDevice, stream) != cudaSuccess)
			BOOST_THROW_EXCEPTION(MocaException("Could not copy data to device!"));
		updateDataDescriptor();
	}

	virtual void getData(T* ptr, int32 dstPitch, cudaStream_t stream = 0)
	{
		if (d_pointer != NULL)
		{
                        cudaError_t ret = cudaMemcpy2DAsync(ptr, dstPitch, d_pointer, d_pitch, d_width, d_height, cudaMemcpyDeviceToHost, stream);
			if (ret != cudaSuccess) {
                          if (ret == cudaErrorInvalidValue)
                            BOOST_THROW_EXCEPTION(MocaException("Could not download data from device: parameter passed to function is not within an acceptable range."));
                          if (ret == cudaErrorInvalidDevicePointer)
                            BOOST_THROW_EXCEPTION(MocaException("Could not download data from device: not a valid device pointer."));
                          if (ret == cudaErrorInvalidPitchValue)
                            BOOST_THROW_EXCEPTION(MocaException("Could not download data from device: invalid pitch value."));
                          if (ret == cudaErrorInvalidMemcpyDirection)
                            BOOST_THROW_EXCEPTION(MocaException("Could not download data from device: invalid memcpy direction."));
                          BOOST_THROW_EXCEPTION(MocaException(ret,"Could not download data from device!"));
                        }
		} else BOOST_THROW_EXCEPTION(MocaException("No data present!"));
	}

	//Destructor
	virtual ~Cuda2DPitchedMemoryHandle()
	{
		//Free ressources
		cudaFree(d_pointer);
	}

	//Setter/Getter
	virtual const Cuda2DPitchedMemoryDataDescriptor<T>* getDataDescPtr(void) const
	{
		if (d_pointer != NULL)
		{
			return &_cpmdd; 
		} else BOOST_THROW_EXCEPTION(MocaException("No data uploaded!"));
	}
};

#endif
