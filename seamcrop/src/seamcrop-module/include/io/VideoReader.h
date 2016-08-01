#ifndef IO_VIDEOREADER_H
#define IO_VIDEOREADER_H

#include "io/Reader.h"
#include <iostream>
extern "C" {
  #include <libavcodec/version.h>
#undef FF_API_FLAC_GLOBAL_OPTS
  #include <libavcodec/avcodec.h>
  #include <libavformat/avformat.h>
  #include <libswscale/swscale.h>
}


/**
   Class that loads a PGP video file from disk.
   Each call to getImage() will return the next frame of the video.
**/

class VideoReader : public Reader
{
 public:
  /// Constructor.
  VideoReader(std::string const& fileName);
  /// Destructor. Calls stop().
  ~VideoReader();
  
  /// Opens the video file and queries it's properties.
  void start();
  /// Closes the video file.
  void stop();

  void getImage(Image8U& image);
  
  /// Returns the estimated duration of the video [in seconds].
  sizeType getDuration();
  /// Returns the estimated frames per second of the video.
  double getFrameRate();
  /// Returns the estimated number of frames of the video.
  int64_t getTotalFrameCount();
  /// Returns the estimated number of frames of the video.
  int64_t getCurrentFrameCount();
  /// Returns the frame width.
  sizeType getImageWidth();
  /// Returns the frame height.
  sizeType getImageHeight();
  /// Returns Rect(0, 0, width(), height())
  Rect getImageDimension();
  /// Returns the number of channels of the image returned by getImage().
  int32 getImageChannels();

  bool getEndOfVideo();
  void setFileName(std::string const& fileName);
  
 private:
  std::string     fileName; /// Name of the video file.
  sizeType		  duration; // duration of the stream [in seconds] (this value might be estimated)
  double	      frameRate; // average framerate (this value might change in a video)
  int64_t 		  frameCount; // estimated number of frames (this value might be estimated)
  sizeType        width; /// Frame width.
  sizeType        height; /// Frame height.
  int64_t         currentFrame; // current Frame.
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
