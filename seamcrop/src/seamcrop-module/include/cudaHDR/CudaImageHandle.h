#ifndef CUDAIMAGEHANDLE_H
#define CUDAIMAGEHANDLE_H

#include "types/MocaException.h"
#include "driver_types.h"
#include "cuda_runtime.h"
#include "cudaHDR/DataStructures.h"
#include "cudaHDR/Cuda2DPitchedMemoryHandle.h"

template <typename T>
class CudaImageHandle : public Cuda2DPitchedMemoryHandle<T>
{
protected:
	//Describes image data
	CudaImageHandleDataDescriptor<T> _cihdd;

	//Additional data
	unsigned int		d_stride;		//Represents the padding of every pixel in channels
	unsigned int		channels;		//Number of channels

	//Update data descriptor
	void updateDataDescriptor()
	{
		_cihdd.d_pointer    = this->d_pointer;
		_cihdd.d_pitch      = this->d_pitch;
		_cihdd.d_height     = this->d_height;
		_cihdd.d_width      = this->d_width;
		_cihdd.element_size = this->element_size;
		_cihdd.d_stride     = this->d_stride;
		_cihdd.channels     = this->channels;
	}

public:
	//Standard constructor
	CudaImageHandle()
	{
		updateDataDescriptor();
	}

	//Copy Constructor
	CudaImageHandle(const CudaImageHandle<T>& rhs)
	{
		if (rhs.d_pointer != NULL)
		{
			//(re)allocate memory
			this->reallocmem(rhs.d_width, rhs.d_height);

			//copy additional members
			this->d_stride    = rhs.d_stride;
			this->channels     = rhs.channels;

			//copy data
			if (cudaMemcpy2D(this->d_pointer, this->d_pitch, rhs.d_pointer, rhs.d_pitch, this->d_width, this->d_height, cudaMemcpyDeviceToDevice) != cudaSuccess)
				BOOST_THROW_EXCEPTION(MocaException("Could not copy data from device to device!"));
		} else {
			this->d_pointer = NULL;
		}
		updateDataDescriptor();
	}

	//Operators
	CudaImageHandle& operator=(const CudaImageHandle<T>& rhs)
	{
		//Avoid self assignment
		if (this != &rhs)
		{
			if (rhs.d_pointer != NULL)
			{
				//(re)allocate memory
				this->reallocmem(rhs.d_width, rhs.d_height);

				//copy additional members
				this->d_stride    = rhs.d_stride;
				this->channels     = rhs.channels;
        
        cudaError_t ret = cudaMemcpy2D(this->d_pointer, this->d_pitch, rhs.d_pointer, rhs.d_pitch, this->d_width, this->d_height, cudaMemcpyDeviceToDevice); 

				//copy data
				if (ret != cudaSuccess)
					BOOST_THROW_EXCEPTION(MocaException(ret,"Could not copy data on device!"));
			} else this->freemem();
		}
		updateDataDescriptor();
		return *this;
	}

	//Methods
	void allocate(unsigned int width_px, unsigned int height_px, unsigned int channels, unsigned int stride)
	{
		//padding must be >= channels
		if (stride < channels) BOOST_THROW_EXCEPTION(MocaException("Padding is smaller than number of channels!"));

		//Allocate memory
		this->reallocmem(width_px * stride * sizeof(T), height_px);
		
		//Set additional data fields
		this->d_stride     = stride;
		this->channels     = channels;

		//Update data descriptor
		updateDataDescriptor();
	}

	void free()
	{
		//Free ressources
		this->freemem();
		updateDataDescriptor();
	}

	void put(T const* data, int32 srcPitch, unsigned int width_px, unsigned int height_px, unsigned int channels, unsigned int stride, cudaStream_t stream = 0)
	{
		//padding must be >= channels
		if (stride < channels) BOOST_THROW_EXCEPTION(MocaException("Padding is smaller than number of channels!"));

		//(re)allocate memory
		this->reallocmem(width_px * stride * sizeof(T), height_px);

		//Set additional data fields
		this->d_stride     = stride;
		this->channels     = channels;

		//Copy data to device
		if (cudaMemcpy2DAsync(this->d_pointer, this->d_pitch, data, srcPitch, this->d_width, this->d_height, cudaMemcpyHostToDevice, stream) != cudaSuccess)
			BOOST_THROW_EXCEPTION(MocaException("Could not copy data to device!"));
		updateDataDescriptor();
	}

	//Destructor
	~CudaImageHandle() {}

	//Setter/Getter
	const CudaImageHandleDataDescriptor<T>* getDataDescPtr(void) const
	{ 
		if (this->d_pointer != NULL)
		{
			return &_cihdd; 
		} else BOOST_THROW_EXCEPTION(MocaException("No data present!"));
	}
	int getWidth() const { return this->d_width/(this->d_stride * this->element_size); }
	int getHeight() const { return this->d_height; }
	int getChannels() const { return this->channels; }
	int getStride() const { return this->d_stride; }
};

#endif
