#ifndef TYPES_EXPOSURE_H
#define TYPES_EXPOSURE_H

#include "types/MocaTypes.h"
#include "types/Vector.h"
#include <boost/shared_ptr.hpp>

class Image8U;


class Exposure
{
 public:
  enum CheckedInvalids {TOO_BRIGHT, TOO_DARK, TOO_BOTH};

  Exposure();
  Exposure(Exposure const& other);
  Exposure(boost::shared_ptr<Image8U> image, VectorI topLeft, float shutter, CheckedInvalids direction, uint32 parent);

  boost::shared_ptr<Image8U> image;
  VectorI topLeft;
  float shutter;
  CheckedInvalids direction;
  uint32 parent;
};


std::ostream& operator<<(std::ostream& stream, Exposure const& exp);


#endif
