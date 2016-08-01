#ifndef TYPES_SIZE_H
#define TYPES_SIZE_H

#include <sstream>
#include "types/MocaTypes.h"
#include "types/Rect.h"


class Size
{
 public:
  Size() : w(0), h(0) {}
  Size(int32 w, int32 h) : w(w), h(h) {}
  Size(Rect const& rect) : w(rect.w), h(rect.h) {}
  int32 w, h;
};
  

std::ostream& operator<<(std::ostream& stream, Size const& s);


#endif
