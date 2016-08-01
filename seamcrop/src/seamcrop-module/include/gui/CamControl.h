#ifndef GUI_CAMCONTROL_H
#define GUI_CAMCONTROL_H

#include "types/MocaTypes.h"

#ifdef HAVE_CAMERA
#ifdef HAVE_LIBFLTK

#include "types/Rect.h"
#include "gui/FLTKHeaders.h"
#include <boost/shared_ptr.hpp>
#include <vector>


class CameraReader;
class FeatureControl;


class CamControl : public Fl_Double_Window
{
public:
  CamControl(boost::shared_ptr<CameraReader> &camReader, Rect wndSize);
  
private:
  void draw();
  void showPage(indexType pageNum);

  static void prevButtonCallback(Fl_Widget *_widget, void *_window);
  static void nextButtonCallback(Fl_Widget *_widget, void *_window);

  Fl_Output *pageLabel;
  indexType curPage;
  sizeType numPages, numFeatures;
  std::vector<FeatureControl *> controls;

  static const uint32 featuresPerPage = 5;
  static const uint32 borderSize = 2;
  static const uint32 spacing = 8;
};

#endif // HAVE_LIBFLTK
#endif // HAVE_CAMERA

#endif

