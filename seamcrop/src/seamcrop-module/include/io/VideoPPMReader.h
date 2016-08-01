#ifndef IO_VIDEOPPMREADER_H
#define IO_VIDEOPPMREADER_H

#include "io/Reader.h"
#include <fstream>
#include <iostream>


/**
   Class that loads a PGP video file from disk.
   Each call to getImage() will return the next frame of the video.
**/

class VideoPPMReader : public Reader
{
 public:
  /// Constructor.
  VideoPPMReader(std::string const& fileName);
  /// Destructor. Calls stop().
  ~VideoPPMReader();
  
  /// Opens the video file and queries it's properties.
  void start();
  /// Closes the video file.
  void stop();

  void getImage(Image8U& image);
  
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
  std::string fileName; /// Name of the video file.
  std::ifstream file; /// File that contains the video.
  sizeType width; /// Frame width.
  sizeType height; /// Frame height.
  int32 currentFrame; // current Frame.
  char* data;
  bool started;
  bool endOfVideo;

  /// removes the white Space.
  void rmWhiteSpace(); 
  void rmHeader();
};

#endif
