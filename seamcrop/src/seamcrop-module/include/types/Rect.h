#ifndef TYPES_RECT_H
#define TYPES_RECT_H

#include <sstream>
#include "types/MocaTypes.h"


/// Representation of a rectangle using the position of the top-left corner, width and height.
class Rect
{
 public:
  Rect() : x(0), y(0), w(0), h(0) {}
  Rect(int32 x, int32 y, int32 w, int32 h) : x(x), y(y), w(w), h(h){}
  int32 x, y, w, h;
};


/// Representation of a rectangle using the position of the top-left corner, width and height as double values.
class RectD
{
public:
  RectD() : x(0), y(0), w(0), h(0) {}
  RectD(double x, double y, double w, double h) : x(x), y(y), w(w), h(h){}
  double x, y, w, h;
};


std::ostream& operator<<(std::ostream& stream, Rect const& r);


#endif
