#ifndef IO_PSEUDOCAMERAFEATURE_H
#define IO_PSEUDOCAMERAFEATURE_H


#include "io/CameraFeature.h"


/**
   Implements the camera features of a pseudo camera (see class PseudoCameraReader for details).
   This class does pretty much nothing, since a pseudo camera doesn't have any features but its shutter setting.
**/
class PseudoCameraFeature : public CameraFeature
{
public:
  PseudoCameraFeature(Type feature);

  void setPower(bool newPow) {};
  bool getPower() const {return true;};

  void setValue(uint32 newVal);
  uint32 getValue() const {return shutterVal;};

  void setMode(Mode newMode) {};
  Mode getMode() const {return MODE_Manual;};
  
  void setAbsControl(bool newCMode) {};
  bool getAbsControl() const {return false;};

private:
  uint32 shutterVal;
};

#endif

