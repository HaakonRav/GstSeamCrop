#include "include/types/MocaTypes.h"
#include "include/types/Image8U.h"
#include "include/types/MocaException.h"
#include "include/filter/Filter.h"
#include "include/tools/Timing.h"
#include "include/tools/Thread.h"
#include "include/tools/Maths.h"

#include <algorithm>
#include <string>
#include <vector>

#include "include/cudaHDR/CudaImage8UHandle.h"
#include "include/cudaHDR/CudaImage32FHandle.h"
#include "include/cudaHDR/CudaVectorHandle.h"
#include "seamCropCuda_Kernels.h"
#include "seamCrop_commons.h"
#include "BufferWriter.h"

#define SIGMA 5.0 // smoothing
#define NUM_THREADS 4 // must be at least set to 2! A single thread will wait forever for the next frame.
// the maximum number of threads is currently 8. This is limited by the need for global texture memory references (see seamCropCuda_KernelsTex.cu included several times by seamCropCuda_Kernels.cu)

// stores all the information about each frame
struct FrameInfo
{
  uint32 vWidth; // source frame width
  uint32 tWidth; // target frame width
  uint32 eWidth; // size of the extended window before seam carving
  uint32 oWidth; // rest of width outside target width
  uint32 numSeams; // number of seams to carve out
  uint32 height; // frame height
  uint32 channels; // number of source frame channels
  uint32 frameCount; // number of frames
};

enum
{
  PASS_ONE,
  PASS_TWO
};

//##################################################################################|
// SeamCropPipeline							            |
//##################################################################################|

// repeatedly reads a frame and processes it. Supports two passes:
// 1st - calculate energy and prepare cropping window path computation
// 2st - calculate energy, crop frame, compute and carve out seams, store result in a file
// all objects of this class stay in first pass mode until nextPass() is called which switches
// all threads to second pass mode
class SeamCropPipeline : public Thread
{
public:
  SeamCropPipeline(uint32 threadID); //Note: setFrameInfo(...) must be called before creating objects of this class
  ~SeamCropPipeline();

  // setter methods for the static members
  static void setFrameInfo(FrameInfo const& fi);
  static void setBufferWriter(boost::shared_ptr<BufferWriter> writer);
  static void setColumnCost(boost::shared_ptr<CudaImage32FHandle> columnCost);
  static void setCropLeft(std::vector<uint32>* cropLeft);
  static void setPreSmoothCropLeft(uint32* preSmoothCropLeft);

  // Waits for at least one thread to finish
  static void wait();
  // swaps pass mode
  static void nextPass(int passMode);

  float getTime(); // returns the estimated time addColumnCost (the only kernel specific to the first pass) ran

  static uint32 lastFrameOfVideo; // index of the last frame in the frame window.
  static uint32 lastFrameOfStream; // index of the actual final frame received.
  static bool wraparound; // indicates whether we are doing multiple runs.
  static bool haltExecution;
  static bool firstPass;
  static bool stopped;  // Measurement.

  static std::vector<bool> imgPresent;  // indicates which frames are present in originalVideo.
  static std::vector<boost::shared_ptr<Image8U> > originalVideo; // preloaded video for measurement mode. 

protected:
  // repeatedly reads a single frame and processes it. After all frames are processed it waits
  // for the thread to be stopped
  void doStuff();

private:
  // reads the next frame in readFrameCPU and returns its frame number [1..fi.frameCount] (thread-safe)
  int32_t readNextFrame();

  // performs all steps necessary for the first pass
  void processImage_pass1(uint32 t);
  // calculates the energy (gradient + motion)
  void calculateEnergy(uint32 t);

  // performs all steps necessary for the second pass
  void processImage_pass2(uint32 t);

  static FrameInfo fi;
  static uint32 lastFrameNumber;
  static boost::shared_ptr<BufferWriter> writer; // used to store the result of the second pass

  static std::vector<boost::shared_ptr<CudaImage8UHandle> > imgBuffer; // stores all frames uploaded to the GPU

  static std::vector<bool> imgAvailable; // imgAvailable[t] == true iff frame number t has previously been uploaded to the GPU this pass
  static std::vector<bool> imgDone; // imgDone[t] == true iff frame number t has been fully processed this pass

  static boost::shared_ptr<CudaImage32FHandle> columnCost;
  static std::vector<uint32>* cropLeft;

  // these buffers are similar to imgBuffer and imgDone except they concern themselves with seams
  static std::vector<boost::shared_ptr<CudaImage32FHandle> > seams;
  static std::vector<uint32> seamsDone; // seamsDone[t] == n means the first n seams of frame t have been computed

  static boost::shared_ptr<CudaImage32FHandle> prevSeams;
  static uint32 *preSmoothCropLeft; // used to adjust seams after a buffer wraparound.

  static boost::mutex mutex;
  static boost::condition_variable cond;

  cudaStream_t stream;
  boost::shared_ptr<SeamCropCuda> cuda;
  uint32 threadID;

  // pass 1 (calculate energy) buffers
  float time; // used to estimate the extra time pass 2 would take if a single pass implementation was used instead
  boost::shared_ptr<Image8U> readFrameCPU;
  boost::shared_ptr<CudaImage32FHandle> gradient;
  boost::shared_ptr<CudaImage8UHandle> motionSaliency;
  boost::shared_ptr<CudaImage32FHandle> totalSaliency;

  // pass 2 (crop/seam carve) buffers
  boost::shared_ptr<CudaImage8UHandle> frame;
  boost::shared_ptr<CudaImage32FHandle> energy;
  boost::shared_ptr<CudaImage32FHandle> tmpEnergy;
  boost::shared_ptr<CudaImage32FHandle> fwdEnergy;
  boost::shared_ptr<CudaImage32FHandle> optimalCost;
  boost::shared_ptr<CudaImage32FHandle> predecessors;
  boost::shared_ptr<CudaImage8UHandle> finalFrame;
  boost::shared_ptr<Image8U> writeFrameCPU;
};

//##################################################################################|
// SeamCrop								            |
//##################################################################################|

// The class first calculates a cropping window of target size. It then adds x% to the borders and removes 
// them again through seam carving. For temporal coherence, more seams are searched in a next key frame 
// and the closest with the minimal costs are used.
class SeamCrop
{
public:
  // changes the video to match these parameters. Only used in measurement mode
  SeamCrop(uint32 videoWidth, uint32 videoHeight, uint32 numFrames, float retargetingFactor, float extendBorderFactor);
  //SeamCrop(uint32 scaleWidth, uint32 scaleHeight, uint32 numFrames);
  SeamCrop();
  ~SeamCrop();

  // perform all steps
  void run();
  // adds an image to the queue
  bool addFrame(unsigned int frameNum, const boost::shared_ptr<Image8U> &image);
  // signals end of stream
  void endOfStreamSignal(unsigned int lastFrameOfStream);
  void setWriter(boost::shared_ptr<BufferWriter> writer);
  void stopExecution();
private:
  // load the frames of the video and calculate the energy
  void run_pass1();
  // starts all threads and waits for them to finish
  void run_threads();
  // smooths the transition between the cropping windows of different passes
  void smoothTransition();
  // smooth the path of the cropping windows to prevent shaking movement
  void smoothSignal();
  // define the positions of the extended borders
  uint32 defineBorders(uint32 cropLeft, uint32 curFrame, float smoothWeight);
  // crop, seam carve and save the frames as video
  void run_pass2();
  // prints the collected timing information to std::cout
  void printTimingInfo();

  FrameInfo fi;
  boost::shared_ptr<SeamCropPipeline> scp[NUM_THREADS];

  boost::shared_ptr<CudaImage32FHandle> columnCost;
  boost::shared_ptr<CudaImage32FHandle> croppingWindowCost;
  boost::shared_ptr<CudaImage32FHandle> predecessors;

  uint32 prevCropLeft;
  uint32 firstPreSmoothCropLeft;
  uint32 totalRetargetedFrames;

  boost::shared_ptr<CudaVectorHandle<unsigned int> > cropLeftGPU;
  std::vector<uint32> cropLeft;
  uint32 pendingFrames; 
  bool endOfStream;  // set to true when the stream ends.
};
