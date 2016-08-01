#ifndef IO_CAMERAFEATURE_H
#define IO_CAMERAFEATURE_H


#include "types/MocaTypes.h"
#include <string>
#include <map>
#include <boost/shared_ptr.hpp>


/**
   Base class for features of a camera.
   This class is implemented twice, once for windows and once for linux.
   A camera feature is any read-/writable attribute of a camera.
   Examples include shutter setting, trigger parameters, gain control, etc.
   Camera features can usually be checked for presence, permission to change and their particular value.
**/
class CameraFeature
{
public:
  /// See the FireGrab or libdc1394 documentation for details on these camera features.
  enum Type
  {
    FEATURE_Shutter = 0,
    FEATURE_Brightness,
    FEATURE_Exposure,
    FEATURE_Gain,
    FEATURE_Gamma,    
    FEATURE_FrameRate,
    FEATURE_Sharpness,
    FEATURE_Hue,
    FEATURE_Saturation,
    FEATURE_Iris,    
    FEATURE_Focus,
    FEATURE_Trigger,
    FEATURE_TriggerDelay,
    FEATURE_WhiteBalanceUB,
    FEATURE_WhiteBalanceVR,
    FEATURE_WhiteShading,
    FEATURE_Zoom,
    FEATURE_Pan,
    FEATURE_Tilt,
    FEATURE_OpticalFilter,
    FEATURE_CaptureSize,    
    FEATURE_CaptureQuality,
  }; // the order here specifies the order of the features in the std::vector later on

  static const sizeType numFeatures = 22; //!!! must be equal to the number of features in Type

  /// Operation mode of a feature. Some features allow automatic setting. See the FireGrab or libdc1394 docs for more info.
  enum Mode
  {
	MODE_Manual = 0,
	MODE_Auto,
	MODE_OneShotAuto
  };

  CameraFeature(Type feature);
  virtual ~CameraFeature() {};

  /// Returns the feature type as a string.
  std::string getName() const;
  /// Returns the feature type as a CameraFeature::Type.
  Type isFeature() const {return feature;};
  /// Returns true if the feature type matches the given type.
  bool isFeature(Type feature) const {return feature==isFeature();};
  
  /// Returns true if the camera supports this feature. The behavior of all other methods is undefined if this is false.
  bool isPresent() const {return bIsPresent;};
  /// Returns true if this feature can be read out from the camera.
  bool isReadable() const {return bIsReadable;};

  
  // ==================== Power on/off ====================
  /// Returns true if this feature can be turned on and off.
  bool isSwitchable() const {return bIsSwitchable;};
  /// Switches a feature off (false) and on (true). Feature must be switchable.
  virtual void setPower(bool newPow);
  /// Returns the current power state of the feature: off (false) / on (true).
  virtual bool getPower() const;

  // ==================== uint value ====================
  /// Returns true if this feature has a value other than on/off.
  bool hasValue() const {return bHasValue;};
  /// Returns the minimum settable value.
  uint32 getMin() const {return uiMin;};
  /// Returns the maximum settable value.
  uint32 getMax() const {return uiMax;};
  /// Returns the range of settable values for this camera feature.
  void getBoundaries(uint32 &min, uint32 &max) const {min=getMin(); max=getMax();};
  /// Sets the feature's value. Feature must be enabled.
  virtual void setValue(uint32 newVal);
  /// Queries the current value of the feature in the camera.
  virtual uint32 getValue() const;

  // ==================== float value ====================
  /// Returns true if an absolute (real-world) value can be set for this feature
  bool hasAbsValue() const {return bHasAbsValue;}
  /// Sets whether or not the feature shall be controlled by means of an absolute value
  virtual void setAbsControl(bool newCMode);
  /// Returns whether or not the feature is controlled by means of an absolute value
  virtual bool getAbsControl() const;
  /// Returns the minimum settable absolute value.
  float getAbsMin() const {return fMin;};
  /// Returns the maximum settable absolute value.
  float getAbsMax() const {return fMax;};
  /// Returns the range of settable absolute values for this camera feature.
  void getAbsBoundaries(float &min, float &max) const {min=getAbsMin(); max=getAbsMax();};
  /// Sets the feature's absolute value. Feature must be enabled.
  virtual void setAbsValue(float newVal);
  /// Queries the current absolute value of the feature in the camera.
  virtual float getAbsValue() const;

  // ==================== Modes ====================
  /// Returns true if the feature can be set to manual mode
  bool hasManualMode() const {return bHasManualMode;};
  /// Returns true if the feature can be set automatically in the camera.
  bool hasAutoMode() const {return bHasAutoMode;};
  /// Returns true if the camera supports one-shot-auto mode for this feature.
  bool hasOneShotAutoMode() const {return bHasOneShotAutoMode;};
  /// Sets the mode of the feature (manual, auto, one-shot-auto). Camera must support this mode for this feature.
  virtual void setMode(Mode newMode);
  /// Queries the current mode of the feature in the camera.
  virtual Mode getMode() const;

protected:
  Type feature;

  bool bIsPresent, bIsReadable, bIsSwitchable, bHasValue, bHasAbsValue;
  bool bHasManualMode, bHasAutoMode, bHasOneShotAutoMode;
  uint32 uiMin, uiMax;
  float fMin, fMax;
  
private:
  static void initNameMap();

  static std::map<Type, std::string> featureNames;  
};

typedef boost::shared_ptr<CameraFeature> CameraFeaturePtr;

/// prints all available information about the feature
std::ostream& operator<<(std::ostream& stream, CameraFeature const& feature);

#endif

