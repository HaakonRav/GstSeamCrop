#ifndef GUI_CAMSELECT_H
#define GUI_CAMSELECT_H

#include "types/MocaTypes.h"

#ifdef HAVE_CAMERA
#ifdef HAVE_LIBFLTK

#include "types/Rect.h"
#include "gui/FLTKHeaders.h"
#include <vector>


class CamSelect : public Fl_Double_Window
{
public:
  CamSelect(Rect wndSize, int32* selectedCamera);
  ~CamSelect();

private:
  static void callbackFunc(Fl_Widget* w, void* data);
    
  static const int32 buttonOffset;
  static const int32 buttonHeight;

  int32* selectedCamera;
  std::vector<std::string> cameraList;
};

#endif // HAVE_LIBFLTK
#endif // HAVE_CAMERA

#endif

