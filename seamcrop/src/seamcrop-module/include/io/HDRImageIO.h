#ifndef IO_HDRIMAGEIO_H
#define IO_HDRIMAGEIO_H

#include "types/MocaTypes.h"
#include "types/Image32F.h"
#include <boost/shared_ptr.hpp>


class HDRImageIO
{
public:
  /// Saves Image32F to disk. Image is assumed to be BGR (bgr==true) or Yxy (bgr==false)
  static void saveHDRImage(std::string const& fileName, Image32F const& image, bool bgr=true);
  /// Saves Image32F to disk. Image will be BGR (bgr==true) or Yxy (bgr==false)
  static boost::shared_ptr<Image32F> loadHDRImage(std::string const& fileName, bool bgr=true);
};

#endif
