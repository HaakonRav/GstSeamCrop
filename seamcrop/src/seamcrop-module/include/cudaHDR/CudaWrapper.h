#ifndef CUDAWRAPPER_H
#define CUDAWRAPPER_H

#include "cudaHDR/CudaImage8UHandle.h"
#include "cudaHDR/CudaImage32FHandle.h"
#include "cudaHDR/CudaExposureHandle.h"
#include "cudaHDR/CudaVectorHandle.h"

class CudaWrapper
{
private:
	static void checkErrorCode(cudaErrorCode err);

public:
	static void changeStride(CudaImage8UHandle const& source,
		CudaImage8UHandle& target);

	static void BayerBilinear(CudaImage8UHandle const& source, 
		CudaImage8UHandle& target);

	//Normalized Cross Correlation	
	static int corrOffset(CudaVectorHandle<unsigned int> const& bt_vector,
		CudaVectorHandle<unsigned int> const& wt_vector,
		CudaVectorHandle<unsigned int> const& bp_vector,
		CudaVectorHandle<unsigned int> const& wp_vector,
		int guess,
		int maxOffset);

	//Color space conversion
	static void BGRYxy(CudaImage8UHandle const& source,
		CudaImage8UHandle& target);
	static void BGRYxy(CudaImage8UHandle& source);
	static void YxyBGR(CudaImage8UHandle const& source,
		CudaImage8UHandle& target);
	static void YxyBGR(CudaImage8UHandle& source);

	//64 bin histogram
	static unsigned int histo64(CudaImage8UHandle const& desc, 
		CudaVectorHandle<int>& histo, 
		unsigned int channel = 0, 
		float scale = 1.0f);

	//Line histogram
	static void lineHisto(CudaImage8UHandle const& source, 
		CudaVectorHandle<unsigned int>& blackhisto, 
		CudaVectorHandle<unsigned int>& whitehisto, 
		unsigned int mean,
		unsigned int noiseThreshold,
		bool rows,
		unsigned int channel = 0);
	static void lineHisto(CudaImage8UHandle const& source, 
		CudaVectorHandle<unsigned int>& blackhisto, 
		CudaVectorHandle<unsigned int>& whitehisto, 
		Rect& roi,
		unsigned int mean,
		unsigned int noiseThreshold,
		bool rows,
		unsigned int channel = 0);

	//Stitching of HDR Exposure
	static void stitchHDR(std::vector<CudaExposureHandle> const& source, 
		CudaImage32FHandle& target);
};

#endif