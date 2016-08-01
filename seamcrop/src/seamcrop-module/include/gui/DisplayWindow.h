#ifndef GUI_DISPLAYWINDOW_H
#define GUI_DISPLAYWINDOW_H

#include "types/MocaTypes.h"

#ifdef HAVE_LIBFLTK
#include <boost/shared_ptr.hpp>
#include <map>
#include <vector>
#include "gui/FLTKHeaders.h"
#include "types/Rect.h"
#include "types/Size.h"
#include "types/Vector.h"

class Image8U;
class ImagePanel;
class HistogramWindow;


class DisplayWindow : public Fl_Double_Window
{
 public:
  DisplayWindow(Rect rect, std::string title);
  ~DisplayWindow();

  void mainLoop();
  virtual void clickedPoint(VectorI p) {};
  virtual void clickedLine(VectorI p1, VectorI p2) {};
  virtual void clickedRect(Rect rect) {};

  Fl_Scrollbar *vScrollbar, *hScrollbar;

 protected:
  virtual void mainLoopFunc();
  virtual void doStuff() = 0;
  void resize(Rect size);
  void addMenuEntry(std::string name, char shortcut, void (*callback)(DisplayWindow*));
  void showImage(boost::shared_ptr<Image8U const> image);
  
  boost::shared_ptr<HistogramWindow> histogram;
  boost::shared_ptr<Image8U> image;

  static const int32 menuHeight;
  static const int32 scrollbarSize;

 private:
  enum Scrollbars
  {
    SCROLLBARS_NONE,
    SCROLLBARS_VERTICAL,
    SCROLLBARS_HORIZONTAL,
    SCROLLBARS_BOTH
  };
 
  Scrollbars checkForScrollbars(Size wndSize);
  void updateScrollbars(Size wndSize);
  static void cbFunction(Fl_Widget* widget, void* func);
  static void histogramCB(DisplayWindow* wnd);
  static void captureImageCB(DisplayWindow* wnd);
  
  uint32 shownImageWidth, shownImageHeight; // the image dimension of the image which is displayed
  // through the panel doesn't need to match the dimension of the member 'image'. Thus these are needed for the scrollbars
  
  Fl_Menu_Bar* menuBar; // those are automatically destroyed by FLTK. no smart pointer needed
  ImagePanel* panel;
};

#endif // HAVE_LIBFLTK

#endif

