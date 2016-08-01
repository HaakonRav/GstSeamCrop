#ifndef FILTER_SEQUENCEHDR_H
#define FILTER_SEQUENCEHDR_H

#include "types/MocaTypes.h"

#ifdef HAVE_LIBDC1394

#include "io/DC1394Reader_Triggered.h"
#include "filter/HDR.h"
#include <boost/shared_ptr.hpp>


class SequenceHDR : public HDR
{
 public:
  SequenceHDR(boost::shared_ptr<CameraReader> reader, uint32 darkest, uint32 brightest, uint32 minInvalidPerLine, uint32 addDelThreshold);

  /// Clears all exposures and fills them with an HDR sequence.
  void captureHDR(std::vector<Exposure>& exposures);
  
 private:
  std::vector<DC1394ReaderParameters> createParameterSet(std::vector<Exposure>& exposures);
  void analyzeSeqLength(std::vector<uint32>const& tooDark, std::vector<uint32> const& tooBright);
  
  boost::shared_ptr<DC1394Reader_Triggered> seqReader;
  uint32 baseShutter, numExposures, addThreshold, removeThreshold;
  double expFactor;
  bool sequenceChanged;
};


#endif // HAVE_LIBDC1394

#endif

