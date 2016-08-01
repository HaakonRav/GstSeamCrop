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

#ifndef SEAMCROPWRAPPER_H
#define SEAMCROPWRAPPER_H

/**
  Class that passes buffers between the plugin and the retargeting module.
  Each call to passBuffers() will incur a getImage() function call.
 **/

/* C++ interpretation of header. */
#ifdef __cplusplus
#include "seamCropCuda.h"
#include "include/types/MocaTypes.h"
#include "include/types/Rect.h"
#include "include/types/Image8U.h"
#include "BufferWriter.h"
#include "seamCrop_commons.h"

#include <boost/shared_ptr.hpp>
#include <fstream>
#include <vector>
#include <iostream>

extern "C" {
#undef FF_API_FLAC_GLOBAL_OPTS
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <gst/gst.h>
#include <glib.h>
  gint initSeamCrop(GAsyncQueue *output_queue, gint width, gint height, double framerate, 
    float retargetingFactor, float extendBorderFactor, gint frameWindowSize);
  GstClockTime passBuffers(GstBuffer *inbuf, GstBuffer *outbuf);
  void startSeamCrop();
  void signalEndOfStream();
  void flushCurrentInstance();
  void setMeasurementClock(GstClock *cur_clock);
}

class SeamCropWrapper
{
  public:
    SeamCropWrapper();
    SeamCropWrapper(gint inWidth, gint inHeight, double inFramerate, 
        float retargetingFactor, float extendBorderFactor, gint frameWindowSize, GAsyncQueue *output_queue);
    // Destructor. Calls stop().
    ~SeamCropWrapper();

    // SeamCrop object.
    SeamCrop seamCropCuda;
    // Writer that handles output buffers.
    boost::shared_ptr<BufferWriter> writer;
    
    // Adds a received buffer to seamCropCudas internal queue.
    void passFrame(GstBuffer *inbuf);

    // Gives a writeable buffer to the writer.
    void passOutbuf(GstBuffer *outbuf);
  
    void stop(int type);

    // Needed to update total framecount on stream end.
    void setTotalFrameCount(int64_t inFrameCount);

    /* FIXME Trenger egentlig ingen utenom getCurrentFrameCount() og getTargetFrameSize() */

    // Returns the duration of the video.
    sizeType getDuration();
    // Returns the frames per second of the video.
    double getFrameRate();
    // Returns the number of frames of the video.
    int64_t getTotalFrameCount();
    // Returns the number of frames of the video.
    int64_t getCurrentFrameCount();
    // Returns the frame width.
    sizeType getWidth();
    // Returns the frame height.
    sizeType getHeight();
    // Returns Rect(0, 0, width(), height())
    Rect getImageDimension();
    // Returns the number of channels of the image returned by getImage().
    int32 getImageChannels();
    // Indicates whether there are more frames to process.
    bool getEndOfVideo();
    // Returns the retargeting factor passed to the plugin.
    float getRetargetingFactor();
    // Returns the border extension factor passed to the plugin.
    float getExtendBorderFactor();
    // Returns the size needed for an output frame.
    gint getTargetFrameSize();
    
  private:
    // Automatically set with values passed by external source.
    sizeType        width; 
    sizeType        height; 
    double	        frameRate;
    float           retargetingFactor;
    float           extendBorderFactor;
    gint            frameWindowSize;
    bool            started;
    int64_t         currentFrame;
    int64_t         currentMaxFrame;

    // Internal values.
    sizeType        targetWidth;
    sizeType        targetFrameSize;
    double		      duration;
    int64_t 		    frameCount;

    // Video frame storage.
    AVFrame         *pFrameYUV;
    AVFrame         *pFrameBGR;

    // Pointer to pass frame between wrapper and retargeting module.
    boost::shared_ptr<Image8U> image;
};

#else

/* C interpretation of header. */
#include <gst/gst.h>
gint initSeamCrop(GAsyncQueue *output_queue, gint width, gint height, double framerate, 
    float retargetingFactor, float extendBorderFactor, gint frameWindowSize);
GstClockTime passBuffers(GstBuffer *inbuf, GstBuffer *outbuf);
void signalEndOfStream();
void setMeasurementClock(GstClock *cur_clock);
// We need to reset everything.
void flushCurrentInstance();

typedef
struct SeamCropWrapper
SeamCropWrapper;
#endif

#endif
