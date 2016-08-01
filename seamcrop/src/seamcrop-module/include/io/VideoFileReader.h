#ifndef IO_VIDEOFILEREADER_H
#define IO_VIDEOFILEREADER_H

#include "io/Reader.h"
#include <opencv/highgui.h>


/**
   Class that loads a video file from disk.
   Each call to getImage() will return the next frame of the video.
   The supported file formats are system-dependant:
    when using OpenCV 1.0:
    - Windows: anything which can be read using Video for Windows
    - Linux: anything ffmpeg can read
    - Mac OS X: anything QuickTime can play
   Warning: some properties (width, height, frame count) may be inaccurate!
**/

class VideoFileReader : public Reader
{
 public:
  /// Constructor.
  VideoFileReader(std::string const& fileName);
  /// Destructor. Calls stop().
  ~VideoFileReader();
  
  /// Opens the video file and queries it's properties.
  void start();
  /// Closes the video file.
  void stop();
  /// Returns the current video frame and sets the frame pointer to the next frame. Restarts after the last frame was captured.
  void getImage(Image8U& image);
  
  /// Returns the frame width.
  sizeType getImageWidth();
  /// Returns the frame height.
  sizeType getImageHeight();
  /// Returns Rect(0, 0, width(), height())
  Rect getImageDimension();
  /// Returns the number of channels
  int32 getImageChannels();
  
  /// Returns the total number of rames in the video.
  sizeType getFrameCount();
  /// Returns the current position of the frame pointer (number of the current frame, starting from 0).
  indexType getCurrentFrame();
  /// Skips to the specified frame number.
  void setCurrentFrame(indexType frame);

  /// Loads a new video file instead of the initially specified one. (the reader has to be stopped in order to do this)
  void loadVideo(std::string const& fileName);
  
 private:
  std::string fileName; /// Name of the video file.
  CvCapture* file; /// File that contains the video.
  sizeType width; /// Frame width.
  sizeType height; /// Frame height.
  int32 channels; /// Number of channels.
  sizeType frameCount; /// Total number of frames.
};

#endif
