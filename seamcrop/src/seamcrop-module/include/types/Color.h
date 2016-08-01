#ifndef TYPES_COLOR_H
#define TYPES_COLOR_H


#include "types/MocaTypes.h"


class Color
{
 public:
  enum Channel
  {
    LUMINANCE = 0,
    RED = 0,
    GREEN,
    BLUE
  };


  Color(uint8 r, uint8 g, uint8 b);
  Color(uint8 lumi);
  Color(const Color &other);


  Color& operator=(const Color &other);
  uint8& operator[](Channel i);
  const uint8& operator[](Channel i) const;
  
  void convertToGrayscale();
  void convertToRGB();

  bool isGrayscale() const;
  bool isRGB() const;

 private:
  uint8 c[3];
  bool isGray;
};

#endif

