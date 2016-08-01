#ifndef GUI_FEATURECONTROL_H
#define GUI_FEATURECONTROL_H

#include "types/MocaTypes.h"

#ifdef HAVE_CAMERA
#ifdef HAVE_LIBFLTK

#include "types/Rect.h"
#include "gui/FLTKHeaders.h"
#include "io/CameraFeature.h"


class FeatureControl : public Fl_Double_Window
{
public:
  FeatureControl(CameraFeaturePtr feature, Rect wndSize);
  ~FeatureControl();

  void update();

  static const uint32 minWidth = 480;
  static const uint32 height = 68;

private:
  void update(bool power);
  void update(CameraFeature::Mode mode);
  void update(float value);
  void update(bool power, CameraFeature::Mode mode, float value);

  static void setText(Fl_Output* widget, uint32 value);
  static void setText(Fl_Output* widget, float value);
  static void sliderCallback(Fl_Widget* _widget, void* _window);
  static void enableSwitchCallback(Fl_Widget* _widget, void* _window);
  static void absoluteSwitchCallback(Fl_Widget* _widget, void* _window);
  static void autoSwitchCallback(Fl_Widget* _widget, void* _window);
  static void oneShotButtonCallback(Fl_Widget* _widget, void* _window);

  CameraFeaturePtr feature;
  Fl_Check_Button *enableSwitch, *absoluteSwitch, *autoSwitch;
  Fl_Output *minLabel, *maxLabel, *valueLabel;
  Fl_Slider* slider;
};

#endif // HAVE_LIBFLTK
#endif // HAVE_CAMERA

#endif
