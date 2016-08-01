#ifndef IO_READER_H
#define IO_READER_H

/**
   Contains definitions of most reader related classes.
   Readers are classes that produce images in one way or another
   (for example by loading image/video files or capturing images from a camera:)
**/

#include "types/MocaTypes.h"
#include "types/Rect.h"
#include "types/Image8U.h"

#include <boost/shared_ptr.hpp>
#include <fstream>
#include <vector>


/**
   Base class for all readers in the VAOI project.
   Readers are classes that produce images.
   They can be implemented as image/video file readers or
   camera readers that capture live images from a camera.
**/
class Reader
{
 public:
  /// Destructor.
  virtual ~Reader(){}
  /// Does all the initialization like loading the image/video or starting the camera's capturing. Must be called before getImage().
  virtual void start() = 0;
  /// Stops the image acquisition process. After stopping, getImage() can not be called anymore until start() is called.
  virtual void stop() = 0;
  /// Acquires a single image from the reader. start() must be called before getImage().
  virtual void getImage(Image8U& img) = 0;
  /// Returns the width of the image that will be returned by getImage().
  virtual sizeType getImageWidth() = 0;
  /// Returns the height of the image that will be returned by getImage().
  virtual sizeType getImageHeight() = 0;
  /// Returns the currently set region of interest. This includes image width and height.
  virtual Rect getImageDimension() = 0;
  /// Returns the number of channels of the image returned by getImage().
  virtual int32 getImageChannels() = 0;
  
 private:

};

#endif

