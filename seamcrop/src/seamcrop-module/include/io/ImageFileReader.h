#ifndef IO_IMAGEFILEREADER_H
#define IO_IMAGEFILEREADER_H


#include "io/Reader.h"


/**
   Reader that loads a file from disk and returns it each time getImage() is called.
   The core of this class is a call to Imaging::loadImage() which is a wrapper of an OpenCV functions.
   Supported file formats are therefore anything that OpenCV can read.
**/
class ImageFileReader : public Reader
{
 public:
  /// Constructor. The image is already loaded here.
  ImageFileReader(std::string fileName);
  /// Destructor. Frees the loaded image.
  ~ImageFileReader();
  /// Does nothing since the image is loaded in the constructor.
  void start();
  /// Does nothing.
  void stop();
  /// Returns the loaded image.
  boost::shared_ptr<Image8U> getImage();
  void getImage(Image8U& img);
  /// Returns the width of the loaded image.
  sizeType getImageWidth();
  /// Returns the height of the loaded image.
  sizeType getImageHeight();
  /// Returns Rect(0, 0, width(), height())
  Rect getImageDimension();
  /// Returns the number of channels
  int32 getImageChannels();
  /// Loads a new image than the initially specified one. Subsequent calls to getImage will return this new image.
  void loadImage(std::string fileName);

 private:
  boost::shared_ptr<Image8U> theImage; /// The loaded image that will be return by getImage().
};

#endif

