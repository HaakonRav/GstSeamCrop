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

#include "BufferWriter.h"
#include "include/types/MocaException.h"
#include <queue>

// ==================== BufferWriter ====================

// Static members of BufferWriter
static struct SwsContext *img_convert_ctx;

// Measurement variables
GstClock *measureClock;
std::queue<GstClockTime> qAddTimes;
bool measurements;


  BufferWriter::BufferWriter(GAsyncQueue *output_queue, sizeType const& width, sizeType const& height)
: outputQueue(output_queue), width(width), height(height), currentFrame(0), started(false)
{
  measurements = false;
  /* Set up conversion context. */
  img_convert_ctx = sws_getContext(
      width,              // src width
      height,             // src height
      AV_PIX_FMT_BGR24,   // src pixel format
      width,              // dst width
      height,             // dst height
      AV_PIX_FMT_YUV420P, // dst pixel format
      SWS_BICUBIC,        // faster, but image quality suffers
      NULL,               // src filter
      NULL,               // dst filter
      NULL);              // rescale algorithm
  if (img_convert_ctx == NULL)
    BOOST_THROW_EXCEPTION(IOException(std::string("Cannot initialize the conversion context.")));
}

BufferWriter::~BufferWriter()
{
  if(started)
    stop();
}


void BufferWriter::start()
{
  pFrameYUV = av_frame_alloc();
  pFrameBGR = av_frame_alloc();

  // Allocate buffer of appropriate size for pFrameBGR.
  av_image_alloc(pFrameBGR->data, pFrameBGR->linesize, width, height, AV_PIX_FMT_RGB24, RGB24_ALIGNMENT);
}

void BufferWriter::stop()
{
  av_freep(&pFrameBGR->data[0]);
  av_free(pFrameBGR);
  av_free(pFrameYUV);
  //g_async_queue_unref(outputQueue);
}

void BufferWriter::addBuffer(GstBuffer *outbuf)
{
  outBuffers.push(outbuf);
}

void BufferWriter::putImage(Image8U const& image)
{
  GstBuffer *outbuf;
  GstMapInfo info;
  
  /* Retrieve output buffer. 
   * SeamCropCuda guarantees sequential image processing, so we do not need to manage indexes. */
  outbuf = outBuffers.front();
  outBuffers.pop();

  /* Retrieve buffer info. */
  gst_buffer_map(outbuf, &info, GST_MAP_WRITE);

  /* Use av_image_fill_arrays to write correct alignment for the YUV420P image to info.data.*/
  av_image_fill_arrays(
      pFrameYUV->data,      // dst pointers
      pFrameYUV->linesize,  // dst linesize
      info.data,            // storage buffer
      AV_PIX_FMT_YUV420P,   // pixel format
      width,                // width
      height,               // height
      YUV420P_ALIGNMENT);   // alignment

  /* Fill pFrameBGR with data from the provided image. */
  uint8_t* dataPointer = pFrameBGR->data[0];
  for(uint32 y=0;y<height;y++)
    for(uint32 x=0;x<width;x++)
      for(uint32 z=0;z<3;z++)
        *dataPointer++ = image(x, y, z);


  /* Convert the BGR24 image to YUV420P. */
  int ret = sws_scale(
      img_convert_ctx,      // conversion context
      pFrameBGR->data,      // src data
      pFrameBGR->linesize,  // src linesize
      0,                    // src X-coord position
      height,               // src Y-coord position
      pFrameYUV->data,      // dst data
      pFrameYUV->linesize); // dst plane strides

  if(ret <= 0)
    BOOST_THROW_EXCEPTION(IOException(std::string("Output frame conversion failed.")));

  /* Insert output buffer into queue. */
  g_async_queue_push(outputQueue, outbuf);

  if(measurements)
    qAddTimes.push(gst_clock_get_internal_time(measureClock));

  currentFrame++;

  gst_buffer_unmap(outbuf, &info);
}

void BufferWriter::setMeasurementClock(GstClock *cur_clock)
{
  measureClock = cur_clock;
  measurements = true;
}

void BufferWriter::writeMeasurements()
{
  while(!qAddTimes.empty())
  {
    std::cout << "QADD " << GST_TIME_AS_MSECONDS(qAddTimes.front()) << std::endl;
    qAddTimes.pop();
  }
}

sizeType BufferWriter::getImageWidth()
{
  return width;
}

sizeType BufferWriter::getImageHeight()
{
  return height;
}

int32 BufferWriter::getCurrentFrame()
{
  return currentFrame;
}

