#ifndef IO_HDREXPOSUREIO_H
#define IO_HDREXPOSUREIO_H

#include "types/MocaTypes.h"
#include "filter/HDR.h"
#include <boost/shared_ptr.hpp>

class Exposure;


class HDRExposureIO
{
public:
  ~HDRExposureIO();

  /** Creates an object which can read multiple Exposure vectors. As parameter it expects a path which contains a file
      called index.txt which contains the number of exposures. Each exposure vector must be in a subdirectory of that path
      called exposureN where N is a number. getNext() will read the next vector. It automatically starts over if the end
      is reached. **/
  static boost::shared_ptr<HDRExposureIO> createReader(std::string const& pathPrefix);
  void getNext(std::vector<Exposure>& exposures);
  void getPrev(std::vector<Exposure>& exposures);
  
  /** Creates an object which can write multiple Exposure vectors. The meanings of the parameters correspond to the
      meanings of the parameters for readers. The created directory structure is compatible to reader objects. All required
      index files are created automatically (to ensure everything was written to disc call close() or destroy the object). **/
  static boost::shared_ptr<HDRExposureIO> createWriter(std::string const& pathPrefix);
  void putNext(std::vector<Exposure> const& exposures);

  void close();

  /// loads one Exposure vector. path must point to a directory containg a properly formatted index file
  static void loadVector(std::string const& path, std::vector<Exposure>& exposures);
  /** saves one Exposure vector to disc. Everything will be written to directory pointed to by path. The formatting
      is compatible to loadVector(). **/
  static void saveVector(std::string const& path, std::vector<Exposure> const& exposures);

  int32 getCurPos();
  int32 getNumExposures();

private:
  HDRExposureIO(std::string const& pathPrefix);

  bool reading, writing;
  std::string pathPrefix;
  int32 curPos; // index of the exposure set that was least recently loaded/saved
  int32 numExposures;
};

#endif

