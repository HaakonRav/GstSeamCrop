#ifndef GUI_IMAGEWINDOW_H
#define GUI_IMAGEWINDOW_H

#include "types/MocaTypes.h"

#ifdef HAVE_LIBFLTK

#include "gui/DisplayWindow.h"
#include "io/ImageFileReader.h"


class ImageWindow : public DisplayWindow
{
 public:
  ImageWindow(Rect rect, std::string title, std::string fileName);
  ~ImageWindow();

 protected:
  boost::shared_ptr<ImageFileReader> reader;
};

#endif // HAVE_LIBFLTK

#endif

