#ifndef FILTER_OPTHDR_H
#define FILTER_OPTHDR_H

#include "types/MocaTypes.h"

#ifdef HAVE_LIBDC1394

#include "filter/HDR.h"
#include <boost/shared_ptr.hpp>

class DC1394Reader_Triggered;
class DC1394ReaderParameters;


class OptHDR : public HDR
{
 public:
  OptHDR(boost::shared_ptr<CameraReader> reader, uint32 darkest, uint32 brightest, uint32 minInvalidPerLine);

  /// Uses the shutters given in the exposure vector. Image pointers in Exposure must be allocated.
  /// Captures a full HDR sequence if exposures is empty (for bootstrapping).
  void captureHDR(std::vector<Exposure>& exposures);
  
 private:
  void createDefSeq(std::vector<Exposure>& exposures, float startShut, float incFact, uint32 numExps);
  bool isSimilarParams(std::vector<Exposure> const& exposures);
  
  boost::shared_ptr<DC1394Reader_Triggered> seqReader;
  std::vector<DC1394ReaderParameters> captureParams;
  bool sequenceChanged;
  uint32 diffSequence; // Number of frames in a row with differing parameters
  static uint32 const maxDiffSequence = 5; // maximum number before changing camera parameters
  static double const maxShutterDiff = 0.2; // maximum shutter difference in percent
};


#endif // HAVE_LIBDC1394

#endif

