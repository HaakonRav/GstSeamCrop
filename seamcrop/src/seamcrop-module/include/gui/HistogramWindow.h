#ifndef GUI_HISTOGRAMWINDOW_H
#define GUI_HISTOGRAMWINDOW_H

#include "types/MocaTypes.h"

#ifdef HAVE_LIBFLTK

#include "types/Vector.h"
#include "gui/FLTKHeaders.h"
#include <boost/shared_ptr.hpp>


class Image8U;


class HistogramWindow : public Fl_Double_Window
{
 public:
  HistogramWindow(int32 x, int32 y, int32 w = 256, int32 h = 200);
  ~HistogramWindow();

  void computeHist(Image8U const& source);

 protected:
  void draw();

  boost::shared_ptr<Image8U> histImage;
  VectorI histogram;
};

#endif // HAVE_LIBFLTK

#endif
