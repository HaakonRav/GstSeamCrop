#ifndef CUDAEXPOSUREHANDLE_H
#define CUDAEXPOSUREHANDLE_H

#include "types/Exposure.h"
#include "types/MocaException.h"
#include "driver_types.h"
#include "cuda_runtime.h"
#include "cudaHDR/DataStructures.h"
#include "cudaHDR/CudaImage8UHandle.h"

class CudaExposureHandle: public CudaImage8UHandle
{
protected:
	//Describes image data
	CudaExposureHandleDataDescriptor _cehdd;

	//Additional data
	int					x_offset;		//Offset in x direction compared to parent frame
	int					y_offset;		//Offset in y direction compared to parent frame
	float					shutter;		//Shutter value
	int					parent;

	//Update data descriptor
	void updateDataDescriptor()
	{
		_cehdd.d_pointer    = d_pointer;
		_cehdd.d_pitch      = d_pitch;
		_cehdd.d_height     = d_height;
		_cehdd.d_width      = d_width;
		_cehdd.element_size = element_size;
		_cehdd.d_stride     = d_stride;
		_cehdd.channels     = channels;
		_cehdd.x_offset     = x_offset;
		_cehdd.y_offset     = y_offset;
		_cehdd.shutter      = shutter;
		_cehdd.parent	    = parent;
	}

public:
	//Standard constructor
	CudaExposureHandle() : CudaImage8UHandle() {}

	void allocate(unsigned int width_px, unsigned int height_px, unsigned int channels, unsigned int padding, 
		int x_offset, int y_offset, float shutter, int parent)
	{
		//(Re)allocate memory
		CudaImage8UHandle::allocate(width_px, height_px, channels, padding);
		
		//Set additional data fields
		this->x_offset = x_offset;
		this->y_offset = y_offset;
		this->shutter  = shutter;
		this->parent   = parent;

		//Update data descriptor
		updateDataDescriptor();
	}

	void put(unsigned char* data, int32 srcPitch, unsigned int width_px, unsigned int height_px, unsigned int channels, unsigned int padding, 
		int x_offset, int y_offset, float shutter, int parent, cudaStream_t stream = 0)
	{
		//Upcall descriptor
		CudaImageHandle<unsigned char>::put(data, srcPitch, width_px, height_px, channels, padding, stream);
		
		//Set additional data fields
		this->x_offset = x_offset;
		this->y_offset = y_offset;
		this->shutter  = shutter;
		this->parent   = parent;

		//Update data descriptor
		updateDataDescriptor();
	}

	void put(Image8U& img, int x_offset, int y_offset, float shutter, int parent, cudaStream_t stream = 0)
	{
		//Upcall
		CudaImage8UHandle::put(img, stream);

		//Set additional data fields
		this->x_offset = x_offset;
		this->y_offset = y_offset;
		this->shutter  = shutter;
		this->parent   = parent;

		//Update data descriptor
		updateDataDescriptor();
	}

	void put(Exposure& exp, cudaStream_t stream = 0)
	{
		//Upcall
		CudaImage8UHandle::put(*(exp.image), stream);
		
		//Set additional data
		x_offset = exp.topLeft[0];
		y_offset = exp.topLeft[1];
		shutter  = exp.shutter;
		parent   = exp.parent;

		//Update data descriptor
		updateDataDescriptor();
	}

	void getExposureData(Exposure& exp, cudaStream_t stream = 0)
	{
		//Download data
		CudaImage8UHandle::getData(exp.image->ptr(), exp.image->widthStep(), stream);
	}

	CudaExposureHandle& operator=(const CudaExposureHandle& rhs) 
	{
		//Avoid self assignment
		if (this != &rhs)
		{
			//Upcall operator
			CudaImage8UHandle::operator=(rhs);

			if (rhs.d_pointer != NULL)
			{
				//Copy member variables
				this->x_offset		= rhs.x_offset;
				this->y_offset		= rhs.y_offset;
				this->shutter		= rhs.shutter;
				this->parent            = rhs.parent;
			}
		} else freemem();
		updateDataDescriptor();
		return *this;
	}

	CudaExposureHandle(const CudaExposureHandle& rhs) :
		CudaImage8UHandle(rhs)
	{
		if (rhs.d_pointer != NULL)
		{
			//Copy member variables
			x_offset		= rhs.x_offset;
			y_offset		= rhs.y_offset;
			shutter			= rhs.shutter;
			parent			= rhs.parent;
		}
		updateDataDescriptor();
	}

	//Setter/Getter
	const CudaExposureHandleDataDescriptor* getDataDescPtr(void) const
	{
		if (d_pointer != NULL)
		{
			return &_cehdd;
		} else BOOST_THROW_EXCEPTION(MocaException("No data present!"));
	}
	int getOffsetX() const     { return this->x_offset; }
	void addOffsetX(int value) 
	{ 
		this->x_offset += value; 
		updateDataDescriptor();
	}
	void addOffsetY(int value) 
	{
		this->y_offset += value; 
		updateDataDescriptor();
	}
	void setOffsetX(int value) 
	{
		this->x_offset = value;
		updateDataDescriptor();
	}
	void setOffsetY(int value) 
	{ 
		this->y_offset = value; 
		updateDataDescriptor();
	}
	int getOffsetY() const     { return this->y_offset; }
	int getChannels() const    { return this->channels; }
	float getShutter() const   { return this->shutter;  }
	int getParent() const      { return this->parent; }
	void setParent(int value)  { this->parent = value; }
};

#endif
