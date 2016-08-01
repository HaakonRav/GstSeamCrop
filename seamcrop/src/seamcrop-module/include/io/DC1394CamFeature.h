#ifndef IO_DC1394CAMFEATURE_H
#define IO_DC1394CAMFEATURE_H


#include "types/MocaTypes.h"
#include "io/CameraFeature.h"


#ifdef HAVE_LIBDC1394

#include <dc1394/control.h>

class DC1394CamFeature : public CameraFeature
{
public:
  DC1394CamFeature(dc1394camera_t *camera, Type feature);
  
  void setPower(bool newPow);
  bool getPower() const;

  void setValue(uint32 newVal);
  uint32 getValue() const;

  void setAbsValue(float newVal);
  float getAbsValue() const;
  
  void setAbsControl(bool newCMode);
  bool getAbsControl() const;

  void setMode(Mode newMode);
  Mode getMode() const;

private:
  static void initIDMap();

  static std::map<CameraFeature::Type, dc1394feature_t> idMap;

  dc1394camera_t *camera;
  dc1394feature_t id;
};


#else

// this code is only required to avoid linker warning in MSVC (LNK4221)
class DC1394CamFeature 
{
public:
	DC1394CamFeature();
};




#endif // HAVE_LIBDC1394

#endif

