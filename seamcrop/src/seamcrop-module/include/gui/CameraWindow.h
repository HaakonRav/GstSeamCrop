#ifndef GUI_CAMERAWINDOW_H
#define GUI_CAMERAWINDOW_H

#include "types/MocaTypes.h"

#ifdef HAVE_CAMERA
#ifdef HAVE_LIBFLTK

#include "gui/DisplayWindow.h"
#include "io/CameraReader.h"

class CamControl;


class CameraWindow : public DisplayWindow
{
 public:
  CameraWindow(Rect rect, std::string title, Rect camera,
               CameraReader::BusSpeed speed, CameraReader::ColorMode color,
               int selectedCamera = -1, bool freeRunning = true);
  ~CameraWindow();

 protected:
  void mainLoopFunc();
  boost::shared_ptr<CameraReader> reader;
  boost::shared_ptr<CamControl> camControl;
  
 private:
  static void camControlsCB(DisplayWindow* wnd);
};


#endif // HAVE_LIBFLTK
#endif // HAVE_CAMERA

#endif

