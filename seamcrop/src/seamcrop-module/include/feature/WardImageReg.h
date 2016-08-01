#ifndef FEATURE_WARD_IMAGE_H
#define FEATURE_WARD_IMAGE_H

#include "types/MocaTypes.h"
#include "types/Image8U.h"
#include "types/Vector.h"
#include <vector>
class BinImage;
class Rect;
typedef std::vector<BinImage> BinImagePyr;


class WardImageReg
{
 public:
  WardImageReg();

  // semi-static method computing the shift between two grayscale images (requires bitCount[])
  // 2^maxBits is the maximum shift. offset is the initial guess _and_ the result
  void computeShift(Image8U const& img1, Image8U const& img2, int32 maxBits, VectorI& offset);

 private:
  void computeCountLUT();
  int32 findMean(VectorI const& hist, int32 percentage);
  void createBinImagePyr(Image8U const& img, BinImagePyr& imgPyr, uint8 mean, int32 maxBits);
  void createBinImage(Image8U const& img, BinImage& binimg, uint8 mean, int32 pyrLevel = 0);
  void createGreyImage(BinImage const& binimg, Image8U& img); // debugging onry

  // shifts an image (used for the exhaustive search)
  //void shiftBinImageVer(BinImage const& binimg, BinImage& result, int32 offset);
  void shiftBinImageHor(BinImage const& binimg, BinImage& result, int32 offset);
  // computes the XOR (difference) of two binary images
  void binImageDiff(BinImage const& binimg1, BinImage const& binimg2, int32 verOff, BinImage& result);
  // counts the white bits in a difference image
  int32 countOnes(BinImage const& binimg);

  // computes the shift between two bitmap images recursively.
  // first vector is the approximate shift, second vector is the _additional_ fine offset.
  void computeShiftRec(BinImagePyr const& img1, BinImagePyr const& img2, uint32 level, VectorI const& approx, VectorI& offset);

  std::vector<uint8> bitCount;
};


// local class to represent a bitmap image (binary image)
class BinImage
{
public:
  int32 width, height; // real width/height in pixels
  int32 widthStep; // number of uint64s to skip per line (approx: width/64)
  std::vector<uint64> image, exclusion;
  
  void setParms(BinImage const& o); // sets sizes, but doesn't copy data
};


std::ostream& operator<<(std::ostream& stream, BinImage const& img);


#endif
