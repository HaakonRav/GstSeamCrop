/* 
 * This file is a part of GstSeamCrop.
 * 
 * GstSeamCrop is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * GstSeamCrop is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Suite 500,
 * Boston, MA 02110-1335, USA.
 */

#include "include/types/MocaException.h"
#include "seamCropWrapper.h"

// ==================== SeamCropWrapper ====================

// Static members of SeamCropWrapper.
SeamCropWrapper handler;
boost::thread seamExec;
static struct SwsContext *img_convert_ctx;

// Measurement variables.
GstClock *measure_clock;
GstClockTime addTime;
bool measurementMode;

SeamCropWrapper::SeamCropWrapper(gint inWidth, gint inHeight, double inFramerate, 
    float retargetFactor, float inExtendBorderFactor, gint inFrameWindowSize, GAsyncQueue *output_queue) 
: width(inWidth), height(inHeight), frameRate(inFramerate), retargetingFactor(retargetFactor), 
  extendBorderFactor(inExtendBorderFactor), frameWindowSize(inFrameWindowSize), started(false), currentFrame(0), currentMaxFrame(0)
{
  targetWidth     = width * retargetingFactor;

  // Must be divisible by two for correct image conversion.
  if(targetWidth % 2)
    targetWidth -= 1;
  targetFrameSize = av_image_get_buffer_size(AV_PIX_FMT_YUV420P, targetWidth, height, YUV420P_ALIGNMENT);

  // Allocate frames.
  pFrameYUV = av_frame_alloc();
  pFrameBGR = av_frame_alloc();

  // Allocate buffer of appropriate size for pFrameBGR.
  av_image_alloc(pFrameBGR->data,pFrameBGR->linesize, width, height, AV_PIX_FMT_RGB24, RGB24_ALIGNMENT);

  // Allocate an image to fill the received frames into. 
  image = boost::shared_ptr<Image8U>(new Image8U(width, height, 3));

  // Initialize seamCropCuda.
  seamCropCuda = SeamCrop(width, height, frameWindowSize, retargetingFactor, extendBorderFactor);   

  // Initialize the writer.
  writer = boost::shared_ptr<BufferWriter>(new BufferWriter(output_queue, targetWidth, height));

  // Pass writer to seamCropCuda.
  writer->start();
  seamCropCuda.setWriter(writer);

  // Set up conversion context.
  img_convert_ctx = sws_getContext(
      width,                          // src width
      height,                         // src height
      AV_PIX_FMT_YUV420P,             // src pixelformat
      width,                          // dst width
      height,                         // dst height
      AV_PIX_FMT_BGR24,               // dst pixelformat
      SWS_BICUBIC | SWS_CPU_CAPS_MMX, // flags
      NULL,                           // src filter
      NULL,                           // dst filter
      NULL);                          // rescale algorithm

  if(!img_convert_ctx)
    BOOST_THROW_EXCEPTION(IOException(std::string("Cannot initialize conversion context.")));

  if(measurementMode)
  {
    writer->setMeasurementClock(measure_clock);
    std::cout << "----- Video properties. -----" << std::endl;
    std::cout << "Input size: " << width << "x" << height << std::endl;
    std::cout << "Target size: " << targetWidth << "x" << height << std::endl;
    std::cout << "Retargetfactor: " << retargetFactor << std::endl;
    std::cout << "Extendfactor: " << extendBorderFactor << std::endl;
    std::cout << "Frame window size: " << frameWindowSize << std::endl;
  }
}

SeamCropWrapper::SeamCropWrapper() : started(false)
{

}
SeamCropWrapper::~SeamCropWrapper()
{
  if(started) {
    av_freep(&pFrameBGR->data[0]);
    av_free(pFrameBGR);
    av_free(pFrameYUV);
  }
}

gint initSeamCrop(GAsyncQueue *output_queue, gint width, gint height, double framerate, 
    float retargetingFactor, float extendBorderFactor, gint frameWindowSize)
{
  handler = SeamCropWrapper(width, height, framerate, retargetingFactor, 
      extendBorderFactor, frameWindowSize, output_queue);
  return handler.getTargetFrameSize();
}

GstClockTime passBuffers(GstBuffer *inbuf, GstBuffer *outbuf)
{
  handler.passFrame(inbuf);
  handler.passOutbuf(outbuf);

  if(measurementMode)
    return addTime;
  else
    return GST_CLOCK_TIME_NONE;
}


void startSeamCrop()
{
  handler.seamCropCuda.run();
}

void flushCurrentInstance()
{
  //std::cout << "Waiting for SeamCropCuda to finish." << std::endl;
  if(measurementMode)
    std::cout << "FLUSH" << std::endl;
  handler.stop(0);
  handler = SeamCropWrapper();
}

void signalEndOfStream() {

  if(measurementMode)
    std::cout << "EOS" << std::endl;
  handler.seamCropCuda.endOfStreamSignal(handler.getCurrentFrameCount());
  // Wait for SCC to finish.
  //std::cout << "Waiting for SeamCropCuda to finish." << std::endl;
  //seamExec.join();
  //std::cout << "SeamCropCuda finished." << std::endl;
  handler.stop(1);
  handler = SeamCropWrapper();
}

void setMeasurementClock(GstClock *cur_clock)
{
  measure_clock = cur_clock;
  measurementMode = true;
}

void SeamCropWrapper::stop(int type)
{
  if(type == 1)
  {
    // End of stream case.
    seamExec.join();
    seamCropCuda = SeamCrop();
    if(writer)
    {
      if(measurementMode)
        writer->writeMeasurements();
      writer->stop();
      writer = boost::shared_ptr<BufferWriter>();
    }
  } else {
    // Flush case.
    if(started)
    {
      seamCropCuda.stopExecution();
      img_convert_ctx = NULL;
      seamExec.join();
      seamCropCuda = SeamCrop();
      if(writer)
      {
        if(measurementMode)
          writer->writeMeasurements();
        writer->stop();
        writer = boost::shared_ptr<BufferWriter>();
      }
    }
  }
}

void SeamCropWrapper::passFrame(GstBuffer *inbuf)
{
  // Extract info from buffer.
  bool seamReturn;
  GstMapInfo info;
  gst_buffer_map(inbuf, &info, GST_MAP_READ);


  // Fill pFrameYUV with the raw YUV image (info.data).
  av_image_fill_arrays(
      pFrameYUV->data,                  // dst pointers
      pFrameYUV->linesize,              // dst linesize
      info.data,                        // video frame
      AV_PIX_FMT_YUV420P,               // pixel format
      width,                            // width
      height,                           // height
      YUV420P_ALIGNMENT);               // alignment

  // Convert image from YUV420P to RGB24.
  int ret = sws_scale(
      img_convert_ctx,                  // conversion context
      pFrameYUV->data,                  // src data
      pFrameYUV->linesize,              // src plane strides
      0,                                // src X-coord position
      height,                           // src Y-coord position
      pFrameBGR->data,                  // dst data
      pFrameBGR->linesize);             // dst plane strides

  if(ret <= 0)
    BOOST_THROW_EXCEPTION(IOException(std::string("Input frame conversion failed.")));

  // Set pointer to beginning of imagedata.
  uint8_t* dataPointer = pFrameBGR->data[0];
  Image8U& img = *image;

  // Copy contents into an Image8U frame.
  for(uint32 y = 0; y < height; y++)
    for(uint32 x = 0; x < width; x++)
    {
      img(x,y,0) = *dataPointer++;
      img(x,y,1) = *dataPointer++;
      img(x,y,2) = *dataPointer++;
    }

  if(measurementMode)
    addTime = gst_clock_get_internal_time(measure_clock);

  do {
    seamReturn = seamCropCuda.addFrame(currentMaxFrame, image);
  } while(!seamReturn);


  // Start SeamCropCuda when first frame has been added.
  if(!started) {
    seamExec = boost::thread(startSeamCrop);
    started = true;
  }

  // Update the frame counter. Will reset to 0 when frameWindowSize is reached.
  currentMaxFrame = (currentMaxFrame + 1) % frameWindowSize;

  gst_buffer_unmap(inbuf, &info);
}

void SeamCropWrapper::passOutbuf(GstBuffer *outbuf)
{
  GstMapInfo info;
  gst_buffer_map(outbuf, &info, GST_MAP_READ);
  /* Send output buffer to the writer. */
  writer->addBuffer(outbuf);
  gst_buffer_unmap(outbuf, &info);
}


void SeamCropWrapper::setTotalFrameCount(int64_t inFrameCount)
{
  frameCount = inFrameCount;
}

double SeamCropWrapper::getFrameRate()
{
  return frameRate;
}

int64_t SeamCropWrapper::getTotalFrameCount()
{
  return frameCount;
}

int64_t SeamCropWrapper::getCurrentFrameCount()
{
  return currentMaxFrame;
}

sizeType SeamCropWrapper::getWidth()
{
  return width;
}

sizeType SeamCropWrapper::getHeight()
{
  return height;
}

Rect SeamCropWrapper::getImageDimension()
{
  return Rect(0,0,width,height);
}

int32 SeamCropWrapper::getImageChannels()
{
  return 3;
}

bool SeamCropWrapper::getEndOfVideo()
{
  return (currentFrame == frameCount);
}

float SeamCropWrapper::getRetargetingFactor()
{
  return retargetingFactor;
}

float SeamCropWrapper::getExtendBorderFactor()
{
  return extendBorderFactor;
}

gint SeamCropWrapper::getTargetFrameSize()
{
  return targetFrameSize;
}



