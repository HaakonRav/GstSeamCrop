#ifndef TOOLS_DRAWING_H
#define TOOLS_DRAWING_H

#include "types/ImageBase.h"
#include "types/Color.h"
#include "types/Rect.h"
#include "types/Vector.h"


class Drawing
{
 public:
  /// Draws a cross into an image.
  static void drawCross(ImageBase& image, VectorI const& pos, int32 const size, int32 const width, Color color);
  /// Draws a line between two points and of given width into an image.
  static void drawLine(ImageBase& image, VectorI const& pos1, VectorI const& pos2, int32 width, Color color);
  /// Draws a non-solid rectangle with given line width into an image.
  static void drawRect(ImageBase& image, Rect const& rect, Color color, int thickness=1);
  /// Draws a Circle into an image.
  static void drawCircle(ImageBase &image, Vector const& pos, int32 const radius, Color &color, int thickness=1);
  /// Draws a arrow from pos1 to pos2 with given width into an image.
  static void drawArrow(ImageBase& image, VectorI const& pos1, VectorI const& pos2, int32 width, Color color);
  /// Draws a Dot into an image.
  static void drawDot(ImageBase& image, Vector const& pos, int32 const radius, Color &color);
  /// Draws a Text into an image.
  static void drawText(ImageBase& image, Vector const& pos, char const* text, Color& color, float const size=1.0f, int font_face=CV_FONT_HERSHEY_SIMPLEX);
  /// Draws a Number into an image.
  static void drawNumber(ImageBase& image, Vector const& pos, int const number, Color& color, float const size=1.0f, int font_face=CV_FONT_HERSHEY_SIMPLEX);
  /// Draws an array (e.g. a histogram) as bars into an image.
  static void arrayToImage(ImageBase& image, std::vector<double> const& array);
};


#endif // TOOLS_DRAWING_H
