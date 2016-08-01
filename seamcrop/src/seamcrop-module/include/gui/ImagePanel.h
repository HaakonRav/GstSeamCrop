#ifndef GUI_IMAGEPANEL_H
#define GUI_IMAGEPANEL_H

#include "types/MocaTypes.h"

#ifdef HAVE_LIBFLTK

#include <boost/shared_ptr.hpp>
#include "types/MocaTypes.h"
#include "types/Rect.h"
#include "gui/FLTKHeaders.h"

class Image8U;
class ImageFileWriter;


class ImagePanel : public Fl_Double_Window
{
 public:
  ImagePanel(int32 x, int32 y, int32 w, int32 h, const char* title);

  void showImage(boost::shared_ptr<Image8U const> image);
  void captureImage();
  
 protected:
  void draw();
  int32 handle(int32 event);

  boost::shared_ptr<Image8U const> image;
  boost::shared_ptr<ImageFileWriter> imgWriter;
  Rect rectL, rectR;
  bool leftButton;
  bool rightButton;
 
 private:
  static void drawRGB(void* panel, int32 x, int32 y, int32 w, uint8* dst);
  static void drawGrayscale(void* panel, int32 x, int32 y, int32 w, uint8* dst);
};

#endif // HAVE_LIBFLTK

#endif
