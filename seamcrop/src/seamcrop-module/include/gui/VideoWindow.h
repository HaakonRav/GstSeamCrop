#ifndef GUI_VIDEOWINDOW_H
#define GUI_VIDEOWINDOW_H

#include "types/MocaTypes.h"

#ifdef HAVE_LIBFLTK

#include "gui/DisplayWindow.h"
#include "io/VideoReader.h"

class VideoWindow : public DisplayWindow
{
 public:
  VideoWindow(Rect rect, std::string title, std::string fileName);
  ~VideoWindow();

 protected:
  boost::shared_ptr<VideoReader> reader;
};

#endif // HAVE_LIBFLTK

#endif
