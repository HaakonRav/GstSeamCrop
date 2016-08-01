#ifndef IO_DC1394READER_TRIGGERED_H
#define IO_DC1394READER_TRIGGERED_H

#include "io/DC1394Reader.h"

#ifdef HAVE_LIBDC1394


struct DC1394ReaderParameters
{
  Rect roi;
  uint32 shutter;
  float gain;
  uint32 bpp; // for internal use by DC1394Reader_Triggered only
};


class DC1394Reader_Triggered : public DC1394Reader
{
 public:
  DC1394Reader_Triggered(BusSpeed speed, ColorMode color, indexType selectedCamera = 0);
  ~DC1394Reader_Triggered();

  void start();
  void stop();

  // single shot methods:
  void captureImage(Rect roi, uint32 shutter, float gain=-10034);
  
  // sequence mode methods:
  void captureImages(std::vector<DC1394ReaderParameters> const& params);
  void captureImages(std::vector<uint32> const& newShutter, std::vector<float> const& newGain);
  void captureImages();

  uint32 getNumRemainingImages() const { return params.size() - paramsIndex; };
  DC1394ReaderParameters getImageParams() const; // returns parameters of the image returned by the next getImage() call

  // methods for both use cases:
  void getImage(Image8U& img);

 protected:
  void enableSequenceMode();
  void disableSequenceMode();

  void setImageDimension(Rect roi);
  void setMaxROI();
  void setupSequence();

  void triggerImage();  
  void triggerSequence();

  Rect roi; // [single shot] stores the last ROI used ||| [sequence mode] biggest region of interest (.w, .h)
  
  // sequence mode attributes:
  bool hasSequenceMode, inSequenceMode;
  uint32 seqRegisters[3]; // used for storing the content of the sequence mode registers
  std::vector<DC1394ReaderParameters> params; // stores the parameters used for the last captured sequence
  uint32 paramsIndex; // stores the index of the parameter set for the next image to be retrieved
  uint32 maxBPP; // maximum number of bytes per packet
};

#endif // HAVE_LIBDC1394

#endif
