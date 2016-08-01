#ifndef SEAMCROPCUDA_KERNELS_H
#define SEAMCROPCUDA_KERNELS_H

#include "cudaHDR/DataStructures.h"
#include <map>
#include <boost/shared_ptr.hpp>


class CudaImage8UHandle;
class CudaImage32FHandle;
template <class T> class CudaVectorHandle;


class SeamCropCuda
{
public:
  SeamCropCuda(uint32 threadID, uint32 w, uint32 ew, uint32 h, double sigma, cudaStream_t* stream);
  ~SeamCropCuda();

  // pass 1
  // compute a gradient-based energy map
  void computeGradient(CudaImage8UDataDescriptor const& image, CudaImage32FDataDescriptor const& energy);
  // substract the second from the first frame to find the positions of changing pixel values
  void findMotionSaliency(CudaImage8UDataDescriptor const& frame1, CudaImage8UDataDescriptor const& frame2, CudaImage8UDataDescriptor const& saliency);
  // gaussian smoothing
  void smooth(CudaImage8UDataDescriptor const& image);
  // combine gradient energy with motion energy into one energy map
  void mergeSaliency(CudaImage8UDataDescriptor const& motionSaliency, CudaImage32FDataDescriptor const& gradient, CudaImage32FDataDescriptor const& result);
  // add column sums of the energy in frame to columnCost in row t  
  void addColumnCost(CudaImage32FDataDescriptor const& frame, CudaImage32FDataDescriptor const& columnCost, unsigned int t);

  // intermediate
  // Note: static calls only work properly after at least one SeamCropCuda instance has been created and before any instances have been destroyed!
  // calculate the energy cost of each cropping window and store the value on the position of its left border
  static void calculateCostCroppingWindowTime(CudaImage32FDataDescriptor const& columnCost, CudaImage32FDataDescriptor const& croppingWindow, unsigned int tw);
  // find the path of cropping windows that contains the maximum energy
  static void calculateMaxEnergyPath(CudaImage32FDataDescriptor const& croppingWindow, CudaImage32FDataDescriptor const& predecessors, CudaVectorHandleDataDescriptor<unsigned int> const& cropLeft);

  // pass 2
  // image/energy cropping
  void cropImage8U(CudaImage8UDataDescriptor const& src, CudaImage8UDataDescriptor const& dst, unsigned int cropLeft);
  void cropImage32F(CudaImage32FDataDescriptor const& src, CudaImage32FDataDescriptor const& dst, unsigned int cropLeft);
  // computes the Forward Energy of image and stores it in fwdEnergy (channels: left, center, right)
  void computeForwardEnergy(CudaImage8UDataDescriptor const& image, CudaImage32FDataDescriptor const& fwdEnergy);
  // calculate temporal coherence cost and add them to a temp energy map. the temporal costs are based on
  // gradients and the distance of the pixel that should be removed to the according seam pixel from the last frame.
  bool addTemporalCoherenceCost(CudaImage32FDataDescriptor const& energy, CudaImage32FDataDescriptor const& tmpEnergy, CudaImage32FDataDescriptor const& previousSeams, unsigned int seamID, int cropOffset);
  // sum up the cheapest paths from top to bottom by dynamic programming. In this case,  in each position the cheapest value of the possible connected predecessors is added.
  void computeCostWidth(CudaImage32FDataDescriptor const& energy, CudaImage32FDataDescriptor const& fwdEnergy, CudaImage32FDataDescriptor const& costWidth, CudaImage32FDataDescriptor const& predecessors);
  // mark the position of the pixels of the found seam
  void markSeamWidth(CudaImage32FDataDescriptor const& costWidth, CudaImage32FDataDescriptor const& energy, CudaImage32FDataDescriptor const& predecessors, CudaImage32FDataDescriptor const& seams, unsigned int seamID);
  // carve out all seams
  void removeSeams(CudaImage8UDataDescriptor const& src, CudaImage8UDataDescriptor const& dst, CudaImage32FDataDescriptor const& seams);

private:
  // fills function pointer maps so that for each map the following is true: map[i] points to the implementation using the global texture memory reference for thread i
  static void setupFunctionPointerMaps();
  // uploads the specified kernel to constant memory (used by smooth(...)). Returns false if the kernel is too large.
  static bool uploadGaussKernel(float* kernel, unsigned int size);

  typedef void (*computeGradientFuncPtr)(CudaImage8UDataDescriptor const&, CudaImage32FDataDescriptor const&, unsigned int*, unsigned int*, float*, cudaStream_t*);
  typedef void (*computeFwdEnergyFuncPtr)(CudaImage8UDataDescriptor const&, CudaImage32FDataDescriptor const&, cudaStream_t*);

  computeGradientFuncPtr computeGradientFunc;
  computeFwdEnergyFuncPtr computeFwdEnergyFunc;
  cudaStream_t* stream;

  // maps thread IDs to their corresponding 'instance' of the CUDA code that uses the TMU
  static std::map<uint32, computeGradientFuncPtr> computeGradientFuncPtrMap;
  static std::map<uint32, computeFwdEnergyFuncPtr> computeFwdEnergyFuncPtrMap;

  static uint32 gaussKernelSize_CPU;

  // pass 1 buffers
  boost::shared_ptr<CudaImage8UHandle> smoothTmp;
  CudaImage8UDataDescriptor const* smoothTmpData;
  
  unsigned int* d_min;
  unsigned int* d_max;
  float* averageMax;

  // intermediate buffers
  static unsigned int* staticBlockCnt;

  // pass 2 buffers
  unsigned int* count;
  unsigned int* blockCnt;
};

#endif // SEAMCROPCUDA_KERNELS_H

