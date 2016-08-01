#ifndef IO_DC1394READER_H
#define IO_DC1394READER_H

#include "types/MocaTypes.h"

#ifdef HAVE_LIBDC1394

#include <dc1394/control.h>
#include "io/CameraReader.h"


class DC1394Reader : public CameraReader
{
 public:
  DC1394Reader(BusSpeed speed, ColorMode color, indexType selectedCamera = 0);
  virtual ~DC1394Reader();
  
  void start();
 
  typedef dc1394camera_t* CameraPtr;
  typedef dc1394video_frame_t* FramePtr;
  
  sizeType getImageWidth();
  sizeType getImageHeight();
  Rect getImageDimension();
  int32 getImageChannels();
  void getMaxImageSize(sizeType& width, sizeType& height);
  void getUnitSize(sizeType& unitSizeX, sizeType& unitSizeY);
  void getUnitPos(uint32& unitPosX, uint32& unitPosY);
 
  virtual void captureImage(Rect roi, uint32 shutter, float gain);
  virtual void returnImagePtr(boost::shared_ptr<CameraImage> image);
  void getImage(Image8U& img);
  
  static void enumerateCameras(std::vector<std::string> &cameraList);
  static void resetBus(indexType cameraId);
  
 protected:
  void printFormat7Info();
  void printCameraFeaturesInfo();
  void setImageDimension(Rect roi, uint32 forcedBPP = 0, uint32* usedBPP = NULL);
  boost::shared_ptr<CameraImage> getImagePtr();

  CameraPtr camera;
  dc1394_t* contextPtr;
  dc1394camera_id_t cameraId;
  dc1394color_coding_t colorCoding;
  indexType selectedCamera;
  BusSpeed speed;

  sizeType unitX, unitY;
  sizeType maxWidth, maxHeight;
  uint32 unitPosX, unitPosY;
  uint32 unitBPP;

  bool initialized;
};

#endif // HAVE_LIBDC1394

#endif

