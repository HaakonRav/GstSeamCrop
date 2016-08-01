#ifndef IO_IO_H
#define IO_IO_H


#include "types/MocaTypes.h"
#include "types/ImageBase.h"
#include "types/Image8U.h"
#include "types/Rect.h"
#include <boost/shared_ptr.hpp>


class IO
{
public:
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static boost::shared_ptr<Image8U> loadImage(std::string const& filename);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void saveImage(std::string const& fileName, ImageBase const& image);
  
  /// adds an OS-specific directory seperator character at the end of path if it doesn't already end with one
  static void checkPath(std::string& path);
  /// creates the directories path points to (must end in a directory seperator or filename) if they don't exist already
  static void createDirectories(std::string const& path);

private:  
  static char const dirSeperator[2];
};

#endif

