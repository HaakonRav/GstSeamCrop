#ifndef IO_VIDEOPPMWRITER_H
#define IO_VIDEOPPMWRITER_H

#include "io/Writer.h"
#include <fstream>
#include <iostream>


/**
   Class that loads a PGP video file from disk.
   Each call to getImage() will return the next frame of the video.
**/

class VideoPPMWriter : public Writer
{
 public:
  /// Constructor.
  VideoPPMWriter(std::string const& fileName, sizeType const& width, sizeType const& height);
  /// Destructor. Calls stop().
  ~VideoPPMWriter();
  
  /// Opens the video file and queries it's properties.
  void start();
  /// Closes the video file.
  void stop();

  void putImage(Image8U const& image);
  
  /// Returns the frame width.
  sizeType getImageWidth();
  /// Returns the frame height.
  sizeType getImageHeight();
  /// Returns Rect(0, 0, width(), height())
  Rect getImageDimension();
  /// Returns currentFrame.
  int32 getCurrentFrame();
  
 private:
  std::string fileName; /// Name of the video file.
  std::ofstream file; /// File that contains the video.
  sizeType width; /// Frame width.
  sizeType height; /// Frame height.
  int32 currentFrame; /// current Frame.
  bool started;
};

#endif
