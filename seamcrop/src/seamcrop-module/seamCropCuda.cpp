#include "seamCropCuda.h"

/* Global variables */
class SeamCrop;

//##################################################################################|
// SeamCropPipeline							            |
//##################################################################################|

// repeatedly reads a frame and processes it. Supports two passes:
// 1st - calculate energy and prepare cropping window path computation
// 2st - calculate energy, crop frame, compute and carve out seams, store result in a file
// all objects of this class stay in first pass mode until nextPass() is called which switches
// all threads to second pass mode

// static members of SeamCropCuda - BEGIN
std::map<uint32, SeamCropCuda::computeGradientFuncPtr> SeamCropCuda::computeGradientFuncPtrMap;
std::map<uint32, SeamCropCuda::computeFwdEnergyFuncPtr> SeamCropCuda::computeFwdEnergyFuncPtrMap;

uint32 SeamCropCuda::gaussKernelSize_CPU;
unsigned int* SeamCropCuda::staticBlockCnt;
// static members of SeamCropCuda - END

SeamCropCuda::SeamCropCuda(uint32 threadID, uint32 w, uint32 ew, uint32 h, double sigma, cudaStream_t* stream)
{
  static bool doOnce = true;
  if(doOnce)
  {
    setupFunctionPointerMaps();

    uint32 kSize = cvRound(sigma*6 + 1)|1; // imitates behavior of OpenCV 2.4.5
    cv::Mat cvGaussKernel = cv::getGaussianKernel(kSize, sigma, CV_32F);
    if(!uploadGaussKernel((float*)cvGaussKernel.ptr(), kSize))
      BOOST_THROW_EXCEPTION(ArgumentException("Gaussian kernel for smoothing is too large."));

    cudaMalloc(&staticBlockCnt, sizeof(unsigned int));
    doOnce = false;
  }

  if(computeGradientFuncPtrMap.size() <= threadID)
    BOOST_THROW_EXCEPTION(RuntimeException(std::string("Invalid thread ID ") + Maths::toString<uint32>(threadID)));

  computeGradientFunc = computeGradientFuncPtrMap[threadID];
  computeFwdEnergyFunc = computeFwdEnergyFuncPtrMap[threadID];

  this->stream = stream;

  smoothTmp = boost::shared_ptr<CudaImage8UHandle>(new CudaImage8UHandle);
  smoothTmp->allocate(h, w, 1, 1);
  smoothTmpData = smoothTmp->getDataDescPtr();

  cudaMalloc(&d_min, sizeof(unsigned int));
  cudaMalloc(&d_max, sizeof(unsigned int));
  cudaMalloc(&averageMax, sizeof(float));

  cudaMallocHost(&count, sizeof(unsigned int));
  cudaMalloc(&blockCnt, sizeof(unsigned int));
}


SeamCropCuda::~SeamCropCuda()
{
  cudaFree(blockCnt);
  cudaFreeHost(count);

  cudaFree(averageMax);
  cudaFree(d_max);
  cudaFree(d_min);

  static bool doOnce = true;
  if(doOnce)
  {
    cudaFree(staticBlockCnt);

    doOnce = false;
  }
}


// static members of SeamCropPipeline - BEGIN
FrameInfo SeamCropPipeline::fi;
bool SeamCropPipeline::firstPass;
bool SeamCropPipeline::wraparound;
bool SeamCropPipeline::haltExecution;
// Measurement
bool SeamCropPipeline::stopped;
uint32 SeamCropPipeline::lastFrameNumber;
boost::shared_ptr<BufferWriter> SeamCropPipeline::writer;
std::vector<boost::shared_ptr<Image8U> > SeamCropPipeline::originalVideo;

// custom variables - BEGIN
std::vector<bool> SeamCropPipeline::imgPresent;
uint32 SeamCropPipeline::lastFrameOfVideo;
uint32 SeamCropPipeline::lastFrameOfStream;
boost::shared_ptr<CudaImage32FHandle> SeamCropPipeline::prevSeams;
uint32 *SeamCropPipeline::preSmoothCropLeft;
// custom variables - END

std::vector<boost::shared_ptr<CudaImage8UHandle> > SeamCropPipeline::imgBuffer;
std::vector<bool> SeamCropPipeline::imgAvailable;
std::vector<bool> SeamCropPipeline::imgDone;

boost::shared_ptr<CudaImage32FHandle> SeamCropPipeline::columnCost;
std::vector<uint32>* SeamCropPipeline::cropLeft;

std::vector<boost::shared_ptr<CudaImage32FHandle> > SeamCropPipeline::seams;
std::vector<uint32> SeamCropPipeline::seamsDone;

boost::mutex SeamCropPipeline::mutex;
boost::condition_variable SeamCropPipeline::cond;
// static members of SeamCropPipeline - END


  SeamCropPipeline::SeamCropPipeline(uint32 threadID)
: threadID(threadID)
{
  cudaStreamCreate(&stream);
  cuda = boost::shared_ptr<SeamCropCuda>(new SeamCropCuda(threadID, fi.vWidth, fi.eWidth, fi.height, SIGMA, &stream));

  this->time = 0.0f;

  readFrameCPU = boost::shared_ptr<Image8U>(new Image8U(fi.vWidth, fi.height, fi.channels));

  gradient = boost::shared_ptr<CudaImage32FHandle>(new CudaImage32FHandle);
  gradient->allocate(fi.vWidth, fi.height, 1, 1);

  motionSaliency = boost::shared_ptr<CudaImage8UHandle>(new CudaImage8UHandle);
  motionSaliency->allocate(fi.vWidth, fi.height, 1, 1);

  totalSaliency = boost::shared_ptr<CudaImage32FHandle>(new CudaImage32FHandle);
  totalSaliency->allocate(fi.vWidth, fi.height, 1, 1);

  frame = boost::shared_ptr<CudaImage8UHandle>(new CudaImage8UHandle);
  frame->allocate(fi.eWidth, fi.height, fi.channels, fi.channels);

  energy = boost::shared_ptr<CudaImage32FHandle>(new CudaImage32FHandle);
  energy->allocate(fi.eWidth, fi.height, 1, 1);

  tmpEnergy = boost::shared_ptr<CudaImage32FHandle>(new CudaImage32FHandle);
  tmpEnergy->allocate(fi.eWidth, fi.height, 1, 1);

  fwdEnergy = boost::shared_ptr<CudaImage32FHandle>(new CudaImage32FHandle);
  //fwdEnergy->allocate(fi.eWidth, fi.height, 3, 3); // multi-channel float images seem not be handled correctly
  fwdEnergy->allocate(fi.eWidth*3, fi.height, 1, 1);

  optimalCost = boost::shared_ptr<CudaImage32FHandle>(new CudaImage32FHandle);
  optimalCost->allocate(fi.eWidth, fi.height, 1, 1);

  predecessors = boost::shared_ptr<CudaImage32FHandle>(new CudaImage32FHandle);
  predecessors->allocate(fi.eWidth, fi.height, 1, 1);

  finalFrame = boost::shared_ptr<CudaImage8UHandle>(new CudaImage8UHandle);
  finalFrame->allocate(fi.tWidth, fi.height, fi.channels, fi.channels);

  writeFrameCPU = boost::shared_ptr<Image8U>(new Image8U(fi.tWidth, fi.height, fi.channels));
}


SeamCropPipeline::~SeamCropPipeline()
{
  cudaStreamDestroy(stream);
}


void SeamCropPipeline::setFrameInfo(FrameInfo const& fi)
{
  SeamCropPipeline::fi = fi;
  firstPass = true;
  wraparound = false;
  lastFrameNumber = 0;
  lastFrameOfVideo = fi.frameCount;
  lastFrameOfStream = fi.frameCount+2;

  imgBuffer.resize(fi.frameCount+2);
  imgAvailable.resize(fi.frameCount+2);
  imgDone.resize(fi.frameCount+2);

  imgPresent.resize(fi.frameCount+2);

  seams.resize(fi.frameCount+2);
  seamsDone.resize(fi.frameCount+2);

  imgPresent[0]   = false;
  imgAvailable[0] = true;
  imgDone[0] = true;

  seams[0] = boost::shared_ptr<CudaImage32FHandle>(new CudaImage32FHandle);
  seams[0]->allocate(1, 1, 1, 1);
  seamsDone[0] = fi.numSeams;

  for(uint32 i = 1; i <= fi.frameCount; ++i)
  {
    imgAvailable[i] = false;
    imgPresent[i] = false;
    imgDone[i] = false;
    seamsDone[i] = 0;
  }

  imgAvailable[fi.frameCount+1] = true;
  imgDone[fi.frameCount+1] = true;
  seamsDone[fi.frameCount+1] = fi.numSeams;
}

void SeamCropPipeline::setBufferWriter(boost::shared_ptr<BufferWriter> writer)
{
  SeamCropPipeline::writer = writer;
}


void SeamCropPipeline::setColumnCost(boost::shared_ptr<CudaImage32FHandle> columnCost)
{
  SeamCropPipeline::columnCost = columnCost;
}


void SeamCropPipeline::setCropLeft(std::vector<uint32>* cropLeft)
{
  SeamCropPipeline::cropLeft = cropLeft;
}

void SeamCropPipeline::setPreSmoothCropLeft(uint32* preSmoothCropLeft)
{
  SeamCropPipeline::preSmoothCropLeft = preSmoothCropLeft;
}

void SeamCropPipeline::wait()
{
  boost::unique_lock<boost::mutex> lock(mutex);
  cond.wait(lock);
}

void SeamCropPipeline::nextPass(int passMode)
{
  switch(passMode)
  {
    case PASS_ONE:
      firstPass = true;
      wraparound = true;
      seamsDone[0] = fi.numSeams;
      break;
    case PASS_TWO:
      firstPass = false;
      lastFrameNumber = 0;
      break;
  }

  lastFrameNumber = 0;

  for(uint32 i = 1; i <= fi.frameCount; ++i)
  {
    imgAvailable[i] = false;
    imgDone[i] = false;
    if(passMode == PASS_ONE)
      seamsDone[i] = 0;
  }

}

float SeamCropPipeline::getTime()
{
  return time;
}


void SeamCropPipeline::doStuff()
{

  if(lastFrameOfVideo < fi.frameCount && !imgDone[lastFrameOfVideo+1])
  {   
    // Set the last actual frame on stream end. 
    // Otherwise, the last thread will wait forever
    // for a frame that never arrives.
    imgAvailable[lastFrameOfVideo+1] = true;
    imgDone[lastFrameOfVideo+1] = true;
  }

  // Are we finished or flushing?
  if(imgDone[1] || haltExecution)
  {
    cond.notify_one();
    boost::this_thread::yield();
    return;
  } 

  while(true)
  {
    int32_t t = readNextFrame();

    /* Attempt to read next frame until it is available. */
    while(t < 0) {
      boost::this_thread::yield();
      t = readNextFrame();
      if(haltExecution)
        t = 0;
    }
    if(t == 0 || haltExecution) {
      break;
    } 

    imgBuffer[t] = boost::shared_ptr<CudaImage8UHandle>(new CudaImage8UHandle);
    imgBuffer[t]->put(*readFrameCPU, stream);
    cudaStreamSynchronize(stream);
    imgAvailable[t] = true;

    if(!firstPass)
      imgPresent[t-1] = false;

    while(!(imgAvailable[t-1] && imgAvailable[t] && imgAvailable[t+1]))
    {
      boost::this_thread::yield();
      if(haltExecution)
        break;
    }

    if(firstPass)
      processImage_pass1((uint32)t);
    else
      processImage_pass2((uint32)t);

    imgDone[t] = true;

    for(uint32 i = 1; i <= lastFrameOfVideo; ++i)
      if(imgDone[i-1] && imgDone[i] && imgDone[i+1]) {
        imgBuffer[i] = boost::shared_ptr<CudaImage8UHandle>();
      }
  }
  // Clear reference to the last processed frame.
  readFrameCPU = boost::shared_ptr<Image8U>();
}

int32_t SeamCropPipeline::readNextFrame()
{
  boost::unique_lock<boost::mutex> lock(mutex);
  if(lastFrameNumber >= lastFrameOfVideo)
    return 0;
  else if(!imgPresent[lastFrameNumber])
    return -1;

  // Frame is available.
  readFrameCPU = originalVideo[lastFrameNumber];

  lastFrameNumber += 1;
  return lastFrameNumber;
}


void SeamCropPipeline::processImage_pass1(uint32 t)
{
  calculateEnergy(t);

  cudaEvent_t begin, end;
  float kernelTime;

  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin, 0);
  cuda->addColumnCost(*totalSaliency->getDataDescPtr(), *columnCost->getDataDescPtr(), t-1);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);

  cudaEventElapsedTime(&kernelTime, begin, end);

  cudaEventDestroy(begin);
  cudaEventDestroy(end);

  time += kernelTime;
}


void SeamCropPipeline::calculateEnergy(uint32 t)
{
  CudaImage8UDataDescriptor const* prevFrame = imgBuffer[t]->getDataDescPtr();
  cuda->computeGradient(*prevFrame, *gradient->getDataDescPtr());

  CudaImage8UDataDescriptor const* nextFrame = prevFrame;

  if(t < lastFrameOfVideo)
    nextFrame = imgBuffer[t+1]->getDataDescPtr();

  if(t > 1)
    prevFrame = imgBuffer[t-1]->getDataDescPtr();

  cuda->findMotionSaliency(*prevFrame, *nextFrame, *motionSaliency->getDataDescPtr());
  cuda->smooth(*motionSaliency->getDataDescPtr());

  cuda->mergeSaliency(*motionSaliency->getDataDescPtr(), *gradient->getDataDescPtr(), *totalSaliency->getDataDescPtr());
}


void SeamCropPipeline::processImage_pass2(uint32 t)
{
  uint32 const numSeams = fi.numSeams;

  seams[t] = boost::shared_ptr<CudaImage32FHandle>(new CudaImage32FHandle);
  seams[t]->allocate(fi.numSeams, fi.height, 1, 1);

  CudaImage8UDataDescriptor const& origFrame = *imgBuffer[t]->getDataDescPtr();
  CudaImage8UDataDescriptor const& frameData = *frame->getDataDescPtr();
  CudaImage32FDataDescriptor const& energyData = *energy->getDataDescPtr();
  CudaImage32FDataDescriptor const& tmpEnergyData = *tmpEnergy->getDataDescPtr();
  CudaImage32FDataDescriptor const& fwdEnergyData = *fwdEnergy->getDataDescPtr();
  CudaImage32FDataDescriptor const& optimalCostData = *optimalCost->getDataDescPtr();
  CudaImage32FDataDescriptor const& predecessorsData = *predecessors->getDataDescPtr();
  CudaImage32FDataDescriptor const& seamsData = *seams[t]->getDataDescPtr();

  /* Wait until previous seams are finished. */
  while(seamsDone[t-1] < numSeams)
  {
    boost::this_thread::yield();
    if(haltExecution)
      return;
  }
  // Suffers from some unknown synchronization issue.
  /*
     while(seamsDone[t-1] == 0)
     boost::this_thread::yield();
     */

  CudaImage32FDataDescriptor const& prevSeamsData = (wraparound && t == 1) ? *prevSeams->getDataDescPtr() 
    : *seams[t-1]->getDataDescPtr();

  calculateEnergy(t);

  unsigned int& curCropLeft = (*cropLeft)[t-1];

  cuda->cropImage8U(origFrame, frameData, curCropLeft);
  cuda->cropImage32F(*totalSaliency->getDataDescPtr(), energyData, curCropLeft);

  cuda->computeForwardEnergy(frameData, fwdEnergyData);

  int32 cropOffset = 0;
  if(t > 1)
    cropOffset = ((*cropLeft)[t-2] - curCropLeft);
  else if(wraparound)
    cropOffset = ((*preSmoothCropLeft) - curCropLeft);

  for(uint32 seamID = 0; seamID < numSeams; ++seamID)
  {
    bool copyEnergy = true;

    while(seamsDone[t-1] <= seamID)
    {
      boost::this_thread::yield();
      if(haltExecution)
        return;
    }

    if(t > 1 || wraparound) 
      copyEnergy = cuda->addTemporalCoherenceCost(energyData, tmpEnergyData, prevSeamsData, seamID, cropOffset);

    if(copyEnergy)
      *tmpEnergy = *energy;

    cuda->computeCostWidth(tmpEnergyData, fwdEnergyData, optimalCostData, predecessorsData);
    cuda->markSeamWidth(optimalCostData, energyData, predecessorsData, seamsData, seamID);
    cudaStreamSynchronize(stream); // necessary to prevent the next thread from trying to read the seam info too soon

    seamsDone[t] = seamID+1;
  }

  cuda->removeSeams(frameData, *finalFrame->getDataDescPtr(), seamsData);
  finalFrame->getImage8UData(*writeFrameCPU, stream);

  while(!imgDone[t-1])
    boost::this_thread::yield();

  // Maintain a pointer to the seam data of the final frame in this retargeting window.
  if(t == lastFrameOfVideo)
    prevSeams = boost::shared_ptr<CudaImage32FHandle>(seams[t]);

  seams[t-1] = boost::shared_ptr<CudaImage32FHandle>();

  // Pass the finished image to the writer.
  writer->putImage(*writeFrameCPU);

  // Stops the timer after the first frame is added to the output queue.
  if(!stopped){
    Timing::stop(5);
    stopped = true;
  }
}

//##################################################################################|
// SeamCrop								            |
//##################################################################################|

// The class first calculates a cropping window of target size. It then adds x% to the borders and removes 
// them again through seam carving. For temporal coherence, more seams are searched in a next key frame 
// and the closest with the minimal costs are used.

SeamCrop::SeamCrop()
{
}

SeamCrop::SeamCrop(uint32 videoWidth, uint32 videoHeight, uint32 numFrames, float retargetingFactor, float extendBorderFactor)
{
  // Set general variables.
  uint32 w = videoWidth;  // width
  uint32 h = videoHeight; // height
  uint32 c = 3;           // image channels (R,G,B)
  uint32 fc = numFrames;  // total framecount

  // Set frameinfo.
  fi.vWidth = w;                                                  // src width
  fi.tWidth = w * retargetingFactor;                              // target width
  fi.eWidth = fi.tWidth + ((w - fi.tWidth) * extendBorderFactor); // extend width
  fi.oWidth = w - fi.tWidth + 1;                                  // rest of width outside of target width
  fi.numSeams = fi.eWidth - fi.tWidth;                            // seams to produce
  fi.height = h;                                                  // height
  fi.channels = c;                                                // image channels
  fi.frameCount = fc;                                             // total framecount

  SeamCropPipeline::setFrameInfo(fi);
  SeamCropPipeline::originalVideo.resize(fi.frameCount);

  for(uint32 i = 0; i < NUM_THREADS; ++i) 
    scp[i] = boost::shared_ptr<SeamCropPipeline>(new SeamCropPipeline(i));

  columnCost = boost::shared_ptr<CudaImage32FHandle>(new CudaImage32FHandle);
  columnCost->allocate(fi.vWidth, fi.frameCount, 1, 1);

  croppingWindowCost = boost::shared_ptr<CudaImage32FHandle>(new CudaImage32FHandle);
  croppingWindowCost->allocate(fi.oWidth, fc, 1, 1);

  predecessors = boost::shared_ptr<CudaImage32FHandle>(new CudaImage32FHandle);
  predecessors->allocate(fi.oWidth, fc, 1, 1);

  cropLeftGPU = boost::shared_ptr<CudaVectorHandle<unsigned int> >(new CudaVectorHandle<unsigned int>);
  cropLeftGPU->resize(fc);

  cropLeft.resize(fc);

  prevCropLeft = 0;
  firstPreSmoothCropLeft = 0;
  totalRetargetedFrames = 0;
  pendingFrames = fc;

  SeamCropPipeline::setColumnCost(columnCost);

  endOfStream = false;
  SeamCropPipeline::haltExecution = false;
  SeamCropPipeline::stopped = false;
}


SeamCrop::~SeamCrop()
{
}

void SeamCrop::stopExecution()
{
  SeamCropPipeline::haltExecution = true;
}


void SeamCrop::run()
{
  SeamCropPipeline::setCropLeft(&cropLeft);
  SeamCropPipeline::setPreSmoothCropLeft(&firstPreSmoothCropLeft);

  while(true){
    
    if(!SeamCropPipeline::wraparound)
    {
      // First frame window. No additional measurements.
      Timing::start(5);
      if(SeamCropPipeline::haltExecution)
        break;
      totalRetargetedFrames += pendingFrames;
      run_pass1();
      if(SeamCropPipeline::haltExecution)
        break;
      SeamCropCuda::calculateCostCroppingWindowTime(*columnCost->getDataDescPtr(), 
          *croppingWindowCost->getDataDescPtr(), fi.tWidth);
      SeamCropCuda::calculateMaxEnergyPath(*croppingWindowCost->getDataDescPtr(), 
          *predecessors->getDataDescPtr(), *cropLeftGPU->getDataDescPtr());
      cropLeftGPU->getData(cropLeft);
      if(SeamCropPipeline::wraparound)
        smoothTransition();
      smoothSignal();
      SeamCropPipeline::nextPass(PASS_TWO);
      run_pass2();
      if((endOfStream && pendingFrames == 0) || SeamCropPipeline::haltExecution)
        break;
      SeamCropPipeline::nextPass(PASS_ONE);

      //Timing::stop(5);
      Timing::start(0); // TIME: total time for remaining frame windows
    } else {

      /* All subsequent frame windows. */

      if(SeamCropPipeline::haltExecution)
        break;

      totalRetargetedFrames += pendingFrames;

      /******* PASS 1: Calculating energy and column costs for each frame *********/
      Timing::start(1);
      run_pass1();
      Timing::stop(1);

      if(SeamCropPipeline::haltExecution)
        break;

      /******* CROPWINDOW: Calculate the best cropping window path and smooth the result. *********/
      Timing::start(2); 

      SeamCropCuda::calculateCostCroppingWindowTime(*columnCost->getDataDescPtr(), 
          *croppingWindowCost->getDataDescPtr(), fi.tWidth);
      SeamCropCuda::calculateMaxEnergyPath(*croppingWindowCost->getDataDescPtr(), 
          *predecessors->getDataDescPtr(), *cropLeftGPU->getDataDescPtr());
      cropLeftGPU->getData(cropLeft);

      /******* SMOOTHING:  Transitional smoothing between cropping windows. *********/
      if(SeamCropPipeline::wraparound)
      {
        Timing::start(3); 
        smoothTransition();
        Timing::stop(3);
      }

      smoothSignal();
      Timing::stop(2);

      /******* PASS 2: Perform window cropping and seam carving on frames. *********/
      Timing::start(4); 
      SeamCropPipeline::nextPass(PASS_TWO);

      run_pass2();

      if((endOfStream && pendingFrames == 0) || SeamCropPipeline::haltExecution)
      {
        Timing::stop(4);
        break;
      }

      SeamCropPipeline::nextPass(PASS_ONE);
      Timing::stop(4);
    }
  }
  
  if(!SeamCropPipeline::wraparound)
    Timing::stop(5);
  else
    Timing::stop(0);
  
  // Uncomment to print timing info on stream end.
  //printTimingInfo();

  // Reset all structures for a potential new run.
  for(int i = 0; i < (int)fi.frameCount; i++)
  {
    if(i < NUM_THREADS)
      scp[i] = boost::shared_ptr<SeamCropPipeline>();
    SeamCropPipeline::originalVideo[i] = boost::shared_ptr<Image8U>();
    SeamCropPipeline::imgPresent[i] = false;
    SeamCropPipeline::wraparound = false;
  }

  columnCost = boost::shared_ptr<CudaImage32FHandle>();
  croppingWindowCost = boost::shared_ptr<CudaImage32FHandle>();
  predecessors = boost::shared_ptr<CudaImage32FHandle>();
  cropLeftGPU = boost::shared_ptr<CudaVectorHandle<unsigned int> >();

}

bool SeamCrop::addFrame(unsigned int frameNum, const boost::shared_ptr<Image8U> &image)
{
  if(SeamCropPipeline::haltExecution)
    return true; // Module is flushing.

  if(SeamCropPipeline::imgPresent[frameNum])
    return false; // The image cannot be added yet. Wrapper must wait.

  if(!SeamCropPipeline::wraparound)
    SeamCropPipeline::originalVideo[frameNum] = boost::shared_ptr<Image8U>(new Image8U(fi.vWidth, fi.height, fi.channels));

  // Adds image to originalVideo[].
  Filter::resize(*image, *SeamCropPipeline::originalVideo[frameNum]);
  SeamCropPipeline::imgPresent[frameNum] = true;

  return true;
}

void SeamCrop::endOfStreamSignal(unsigned int lastFrameOfStream)
{
  // If there is only one frame left, pretend there are two.
  if(SeamCropPipeline::firstPass) {
    SeamCropPipeline::lastFrameOfVideo = (lastFrameOfStream == 0) ? 1 : lastFrameOfStream;
    pendingFrames = 0;
  } else {
    pendingFrames = (lastFrameOfStream == 0) ? 1 : lastFrameOfStream;
    if(lastFrameOfStream == 0) {
      if(!SeamCropPipeline::imgPresent[0]) {
        pendingFrames = 0;
      }
    }
  }
  endOfStream = true;
}


void SeamCrop::run_pass1()
{
  if(endOfStream) 
  {
    SeamCropPipeline::lastFrameOfVideo = pendingFrames;
    pendingFrames = 0;
  }
  run_threads();
}

void SeamCrop::run_threads()
{
  for(uint32 i = 0; i < NUM_THREADS; ++i) 
    scp[i]->start();

  SeamCropPipeline::wait();

  for(uint32 i = 0; i < NUM_THREADS; ++i) 
    scp[i]->stop();
}

void SeamCrop::smoothTransition()
{  

  int diff = prevCropLeft - cropLeft[0];
  if(diff <= 2 && diff >= -2)
  {
    // Difference is negligible. No need to smooth the transition.
    firstPreSmoothCropLeft = prevCropLeft;
    return;
  }

  float weightedPrevCrop;
  float weightedCurCrop;
  float importanceWeight = (1 / (float)SeamCropPipeline::lastFrameOfVideo);

  for(uint32 i = 0; i < SeamCropPipeline::lastFrameOfVideo; ++i)
  {
    // Gradient smoothing from the last frame of the previous window to the current window.
    weightedPrevCrop = prevCropLeft - (prevCropLeft * (i * importanceWeight));
    weightedCurCrop = cropLeft[i] * (i * importanceWeight);
    cropLeft[i] = weightedPrevCrop + weightedCurCrop;
  }
  firstPreSmoothCropLeft = cropLeft[0];
}

void SeamCrop::smoothSignal()
{
  uint32 const fc = SeamCropPipeline::lastFrameOfVideo;

  float *next = new float[fc];
  float *prev = new float[fc];

  if(next == NULL || prev == NULL)
    BOOST_THROW_EXCEPTION(RuntimeException("Out of memory."));

  for(uint32 i = 0; i < fc; ++i) 
    prev[i] = cropLeft[i];

  // gauss-based average, repeated;
  for(uint32 repeat = 0; repeat < 100; repeat++)
  {
    for(uint32 i = 0; i < fc; ++i)
      if(i == 0)
        if(!SeamCropPipeline::wraparound)
          next[i] = (prev[i] + prev[i+1])/2.0f; // (i + (i+1))/2
        else
          next[i] = prev[i];
      else if (i < fc-1)
        next[i] = 0.25f*prev[i-1] + 0.5f*prev[i] + 0.25f*prev[i+1]; //(0.25 + 0.5 + 0.25)
      else  
        next[i] = (prev[i] + prev[i-1])/2.0f;

    float* tmp = prev;
    prev = next;
    next = tmp;
  }

  float smoothWeight = (1 / (float)SeamCropPipeline::lastFrameOfVideo);

  // Ensure that the cropping does not exceed the maximum allowed (width - extended_width - 1). 
  for(uint32 i = 0; i < fc; ++i) 
    cropLeft[i] = defineBorders((uint32)(prev[i] + 0.5f),i, smoothWeight);

  // Remove small one-pixel jitters. Prevalent with substantial retargeting.
  for(uint32 i = 1; i < fc; ++i)
    if(cropLeft[i-1] == cropLeft[i+1] && cropLeft[i] != cropLeft[i+1])
      cropLeft[i] = cropLeft[i-1];

  // Store the last crop value to transition between this and the next pass.
  prevCropLeft = cropLeft[fc-1];

  delete prev;
  delete next;

}


uint32 SeamCrop::defineBorders(uint32 cropLeft, uint32 curFrame, float smoothWeight)
{
  uint32 const w = fi.vWidth;
  uint32 const ew = fi.eWidth;
  uint32 const tw = fi.tWidth;

  int extraSpace = (ew - tw)/2;

  // Granular border adjustment. If the cropLeft value is too high, lock to max value.
  if(SeamCropPipeline::wraparound)
  {
    int adjustedSpace = extraSpace * (curFrame * smoothWeight);

    if(((int) cropLeft) - adjustedSpace <=0)
      cropLeft = 0;
    else if((cropLeft - adjustedSpace) > (w - ew - 1))
      cropLeft = w - ew - 1;
    else
      cropLeft = cropLeft - adjustedSpace;
  } else
  {
    if(((int)cropLeft) - extraSpace <= 0) 
      cropLeft = 0;
    else if((cropLeft + tw + extraSpace) >= w-1)
      cropLeft = w - ew - 1;
    else
      cropLeft = cropLeft - extraSpace;
  }
  return cropLeft;
}

/** Sets the writer for SeamCropPipeline. **/
void SeamCrop::setWriter(boost::shared_ptr<BufferWriter> writer)
{
  SeamCropPipeline::setBufferWriter(writer);
  writer->start();
}

void SeamCrop::run_pass2()
{
  run_threads();
}


void SeamCrop::printTimingInfo()
{
  //std::cout << std::endl << "total_time energy cropping_window trans_smoothing seam_carving first_window total_frames" << std::endl;
  std::cout << "RESULTS " << Timing::sum(0) << " " << Timing::sum(1) << " " << Timing::sum(2) << " ";
  std::cout << Timing::sum(3) << " " << Timing::sum(4) << " " << Timing::sum(5) << " " << totalRetargetedFrames << std::endl;
  for (int i=0; i<6; ++i)
    Timing::reset(i);
}

