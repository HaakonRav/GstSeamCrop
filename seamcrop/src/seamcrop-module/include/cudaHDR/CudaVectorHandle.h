#ifndef CUDAVECTORHANDLE_H
#define CUDAVECTORHANDLE_H

#include "feature/Histogram.h"
#include "types/Image8U.h"
#include "types/MocaException.h"
#include "driver_types.h"
#include "cuda_runtime.h"
#include "cudaHDR/DataStructures.h"

#include <boost/numeric/ublas/vector.hpp>

template <class T>
class CudaVectorHandle
{
protected:
	//Describes vector data (data descriptor).
	CudaVectorHandleDataDescriptor<T> _desc;

	//Members describe on-device data
	T*				d_pointer;
	unsigned int	numelements;

	//Update the data descriptor
	void updateDataDescriptor()
	{
		_desc.d_pointer   = d_pointer;
		_desc.numelements = numelements;
	}

	//Reallocates memory if neccessary
	unsigned int			initial_numelements;
	void reallocmem(unsigned int numelements_new)
	{
		//If no reallocation is needed
		if (d_pointer != NULL && numelements_new <= initial_numelements)
		{
			numelements = numelements_new;
		//Else reallocate memory
		} else {
			//Set new data properties
			numelements = numelements_new;
			initial_numelements = numelements_new;

			//Free old pointer
			cudaFree(d_pointer);

			//Allocte anew
			if (cudaMalloc((void**) &d_pointer, numelements * sizeof(T)) != cudaSuccess)
			{
				d_pointer = NULL;
				BOOST_THROW_EXCEPTION(MocaException("Could not allocate device memory!"));
			}
		}
	}

	//Frees allocated memory
	inline void freemem()
	{
		cudaFree(d_pointer);
		d_pointer = NULL;
	}

public:
	//Standard constructor, does not allocate on-device memory until data is uploaded or the vector is being resized
	CudaVectorHandle()
	{
		d_pointer = NULL;
		numelements = 0;
		updateDataDescriptor();
	}

	//Constructor initializes vector with given numer of elements.
	CudaVectorHandle(unsigned int numelements)
	{
		//Allocate memory
		d_pointer = NULL;
		reallocmem(numelements);
		updateDataDescriptor();
	}

	//Copy constructor.
	CudaVectorHandle(const CudaVectorHandle<T>& rhs)
	{
		//Initialize members
		d_pointer = NULL;

		//If right hand side has not allocated memory skip copy ...
		if (rhs.d_pointer != NULL)
		{
			//Set member variables
			reallocmem(rhs.numelements);

			//Copy data
			if (cudaMemcpy(d_pointer, rhs.d_pointer, rhs.numelements * sizeof(T), cudaMemcpyDeviceToDevice) != cudaSuccess)
				BOOST_THROW_EXCEPTION(MocaException("Could not copy data on device!"));
		} else {
			d_pointer = NULL;
		}
		updateDataDescriptor();
	}

	//Overloading the equals operator.
	virtual CudaVectorHandle& operator=(const CudaVectorHandle<T>& rhs) 
	{
		//Avoid self assigment
		if (this != &rhs)
		{
			if (rhs.d_pointer != NULL)
			{
				//(re)allocate memory
				reallocmem(rhs.numelements);

				//Copy Data
				if (cudaMemcpy(d_pointer, rhs.d_pointer, rhs.numelements * sizeof(T), cudaMemcpyDeviceToDevice) != cudaSuccess)
					BOOST_THROW_EXCEPTION(MocaException("Could not copy data on device!"));
			} else freemem();
		}
		updateDataDescriptor();
		return *this;
	}

	//Resizes on-device memory.
	virtual void resize(unsigned int numelements)
	{
		//Allocate memory
		reallocmem(numelements);
		updateDataDescriptor();
	}

	//Clears device memory (sets on-device data bytewise to zero, should be working for all basic types).
	virtual void clear()
	{
		//Clear memory
		if (cudaMemset(d_pointer, 0, numelements * sizeof(T)) != cudaSuccess)
		{
			d_pointer = NULL;
			BOOST_THROW_EXCEPTION(MocaException("Could not clear device memory!"));
		}
	}

	//Frees on-device memory
	virtual void free()
	{
		//Free memory
		freemem();
		updateDataDescriptor();
	}

	//Uploads data from a pointer to the device
	virtual void put(T* h_pointer, unsigned int numelements, cudaStream_t stream = 0)
	{
		//Reallocate memory
		reallocmem(numelements);

		//Copy data to device
		if (cudaMemcpyAsync(d_pointer, h_pointer, numelements * sizeof(T), cudaMemcpyHostToDevice, stream) != cudaSuccess)
			BOOST_THROW_EXCEPTION(MocaException("Could not upload data to device!"));
		updateDataDescriptor();
	}

	//Uploads data from a standard vector object to the device
	virtual void put(std::vector<T>& vec, cudaStream_t stream = 0)
	{
		//Get number of elements
		unsigned int tmpsize = vec.size();

		//Temporary array to copy vector content to
		T* tmp = new T[tmpsize];
		for (unsigned int i = 0; i < tmpsize; i++) tmp[i] = vec[i];

		//Upload tmp to device
		put(tmp, tmpsize, stream);

		//Free temporary array
		delete[] tmp;
	}

	//Uploads data from a boost vector object to the device
	virtual void put(boost::numeric::ublas::vector<T>& vec, cudaStream_t stream = 0)
	{
		//Get number of elements
		unsigned int tmpsize = vec.size();

		//Temporary array to copy vector content to
		T* tmp = new T[tmpsize];
		for (unsigned int i = 0; i < tmpsize; i++) tmp[i] = vec[i];

		//Upload tmp to device
		put(tmp, tmpsize, stream);

		//Free temporary array
		delete[] tmp;
	}

	//Downloads on-device data to array
	virtual void getData(T* ptr, cudaStream_t stream = 0)
	{
		if (d_pointer != NULL)
		{
			if (cudaMemcpyAsync(ptr, d_pointer, numelements * sizeof(T), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
				BOOST_THROW_EXCEPTION(MocaException("Could not download data from device!"));
		} else BOOST_THROW_EXCEPTION(MocaException("No data uploaded!"));
	}

	//Downloads on-device data to standard vector
	virtual void getData(std::vector<T>& vec, cudaStream_t stream = 0)
	{
		//Resize vector if necessary
		if (vec.size() != numelements) vec.resize(numelements);

		//Temporary array to copy content to
		T* tmp = new T[numelements];

		//Download to tmp
		getData(tmp, stream);

		//Copy from tmp to vector
		for (unsigned int i = 0; i < numelements; i++) vec[i] = tmp[i];

		//Free temporary array
		delete[] tmp;
	}

	//Downloads on-device data to boost vector
	virtual void getData(boost::numeric::ublas::vector<T>& vec, cudaStream_t stream = 0)
	{
		//Resize vector if necessary
		if (vec.size() != numelements) vec.resize(numelements);

		//Temporary array to copy content to
		T* tmp = new T[numelements];

		//Download to tmp
		getData(tmp, stream);

		//Copy from tmp to vector
		for (unsigned int i = 0; i < numelements; i++) vec[i] = tmp[i];

		//Free temporary array
		delete[] tmp;
	}

	//Destructor, frees on-device memory
	virtual ~CudaVectorHandle()
	{
		//Free ressources
		cudaFree(d_pointer);
	}
	
	//Returns a pointer to the data descriptor describing the on-device data
	const CudaVectorHandleDataDescriptor<T>* getDataDescPtr(void) const
	{
		if (d_pointer != NULL)
		{
			return &_desc; 
		} else BOOST_THROW_EXCEPTION(MocaException("No data uploaded!"));
	}

	//Returns the number of elements stored on-device
	unsigned int size(void) const
	{
		return numelements;
	}
};

#endif
