#ifndef KERNELFUNCTIONS_H
#define KERNELFUNCTIONS_H

//Include common data types
#include "cudaHDR/DataStructures.h"

namespace KernelFunctions
{
	//Change padding
	cudaErrorCode changeStride(const CudaImage8UDataDescriptor* desc, 
		const CudaImage8UDataDescriptor* target);

	//Bayerpattern Interpolation
	cudaErrorCode BayerBilinear(const CudaImage8UDataDescriptor* source, 
		const CudaImage8UDataDescriptor* target);

	//Normalized Cross Correlation
	cudaErrorCode advancedNcc(const CudaVectorHandleDataDescriptor<unsigned int>* bt_vector, 
		const CudaVectorHandleDataDescriptor<unsigned int>* wt_vector, 
		const CudaVectorHandleDataDescriptor<unsigned int>* bp_vector, 
		const CudaVectorHandleDataDescriptor<unsigned int>* wp_vector, 
		const CudaVectorHandleDataDescriptor<float>* result,
		int p_offset = 0);

	//Color space conversion
	cudaErrorCode BGRYxy(const CudaImage8UDataDescriptor* source, 
		const CudaImage8UDataDescriptor* target);
	cudaErrorCode BGRYxy(const CudaImage8UDataDescriptor* source);
	cudaErrorCode YxyBGR(const CudaImage8UDataDescriptor* source, 
		const CudaImage8UDataDescriptor* target);
	cudaErrorCode YxyBGR(const CudaImage8UDataDescriptor* source);

	//Stitching of HDR Exposure
	cudaErrorCode stitchHDR(const CudaVectorHandleDataDescriptor<CudaExposureHandleDataDescriptor>* source, 
		const CudaImage32FDataDescriptor* target);

	//64 bin histogram
	cudaErrorCode histo64(const CudaImage8UDataDescriptor* desc, 
		const CudaVectorHandleDataDescriptor<int>* h_histogram, 
		unsigned int channel = 0, 
		float scale = 1.0f,
		unsigned int* numpixels = NULL);

	//Line histogram
	cudaErrorCode lineHisto(const CudaImage8UDataDescriptor* source, 
		const CudaVectorHandleDataDescriptor<unsigned int>* blackhisto, 
		const CudaVectorHandleDataDescriptor<unsigned int>* whitehisto, 
		unsigned int mean,
		unsigned int noiseThreshold, 
		bool rows,
		unsigned int channel = 0, 
		unsigned int roix = 0, 
		unsigned int roiy = 0, 
		unsigned int roiwidth = 0, 
		unsigned int roiheight = 0);

	//Tonemapping
	cudaErrorCode minMaxAvg(const CudaImage32FDataDescriptor* desc, 
		float& min, 
		float& max, 
		float& avg, 
		float minVal);
	cudaErrorCode logHisto(const CudaImage32FDataDescriptor* desc, 
		const CudaVectorHandleDataDescriptor<unsigned int>* histo, 
		float min, 
		float max);
	cudaErrorCode mapImage(const CudaImage32FDataDescriptor* source,
		const CudaImage8UDataDescriptor* result,
		const CudaVectorHandleDataDescriptor<float>* histo,
		float min,
		float max);
}

#endif