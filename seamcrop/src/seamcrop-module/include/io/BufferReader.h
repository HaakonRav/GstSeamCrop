#ifndef IO_BUFFERREADER_H
#define IO_BUFFERREADER_H

/**
  Class that passes buffers between the plugin and the retargeting module.
  Each call to passBuffers() (or other appropriately named function) will incur a getImage() function call.
  ^-- Should be other way around.
  buffers (or their contents) sent by passBuffers should be stored, to be used when needed.
 **/

#include "io/Reader.h"
#include <iostream>

extern "C" 
{
  #include <libavcodec/version.h>
#undef FF_API_FLAC_GLOBAL_OPTS
  #include <libavcodec/avcodec.h>
  #include <libavformat/avformat.h>
  #include <libswscale/swscale.h>
  #include <gst/gst.h>
}

class BufferReader : public Reader
{
  public:
    /// Constructor.
    BufferReader();
    /// Destructor. Calls stop().
    ~BufferReader();

    /// Initializes the reader.
    void start();
    /// Closes the reader.
    void stop();

    /// Gets an image from the queue.
    void getImage(Image8U& image);

    /// Returns the duration of the video.
    sizeType getDuration();
    /// Returns the frames per second of the video.
    double getFrameRate();
    /// Returns the number of frames of the video.
    int64_t getTotalFrameCount();
    /// Returns the number of frames of the video.
    int64_t getCurrentFrameCount();
    /// Returns the frame width.
    sizeType getImageWidth();
    /// Returns the frame height.
    sizeType getImageHeight();
    /// Returns Rect(0, 0, width(), height())
    Rect getImageDimension();
    /// Returns the number of channels of the image returned by getImage().
    int32 getImageChannels();
    /// Indicates whether there are more frames to process.
    bool getEndOfVideo();

  private:
    // Values passed by external source.
    sizeType		    duration;
    double	        frameRate;
    int64_t 		    frameCount;
    sizeType        width; 
    sizeType        height; 

    // Internal values
    int64_t         currentFrame;
    int             videoStream;
    uint8_t         *buffer;
    bool            started, endOfVideo;

    AVFormatContext *pFormatCtx;
    AVCodecContext  *pCodecCtx;
    AVFrame         *pFrame;
    AVFrame         *pFrameBGR;
    AVPacket        packet;
};

#endif
