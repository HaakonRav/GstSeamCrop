#ifndef DEVICEMANAGER_H
#define DEVICEMANAGER_H

#include <vector>
#include "cuda_runtime.h"
#include "driver_types.h"
#include "boost/shared_ptr.hpp"
#include "types/Image8U.h"
#include "cudaHDR/DataStructures.h"
#include "cudaHDR/CudaImageHandle.h"

//This class manages the selection and querying 
//for devices and the data management on the device.
class DeviceManager
{
public:
	//Device Management functions
	static int getNumberOfDevices();
	static int getMaxGFLOPSDevice();
	static void deviceQuery(std::vector<cudaDeviceProp>&);
	static void setDevice(int);

	//Execution Management functions
	static void threadSynchronize(void);

private:
	//Constructors (this class is intended to be never instantiated
	DeviceManager(void){}
	~DeviceManager(void){}
};

#endif