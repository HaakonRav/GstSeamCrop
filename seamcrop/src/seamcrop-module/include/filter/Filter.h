#ifndef FILTER_FILTER_H
#define FILTER_FILTER_H


#include "types/ImageBase.h"
#include "types/Image8U.h"
#include "types/Image16S.h"
#include "types/Image32F.h"
#include "types/Color.h"
#include "types/Rect.h"
#include "types/Vector.h"
#include "types/Matrix.h"


class Filter
{
 public:
  /// Computes a binary image using an adaptive threshold of the given window size.
  static void adaptiveThreshold(ImageBase const& image, ImageBase& result, int32 windSize=17);
  /// limits the channels of an image to the lowClip or highClip.
  static void clip(Image8U const& image, Image8U& result, uint32 const lowClip, uint32 const highClip);
  /// Same as above with Image32F. Can be used in-place
  static void clip(Image32F const& image, Image32F& result, float const lowClip, float const highClip);
  /// Computes two images representing a gradient (orientation using atan() and magnitudes using euclidean length).
  static void computeGradients(Image32F const& image, Image32F& orientations, Image32F& magnitudes);
  /// changes the contrast of the image. cValue has to be between -1.0 and 1.0
  static void contrast(Image8U const& image, Image8U& result, double const& cValue);
  /// Computes arbitrary derivatives of a given image. Horizontal and vertical derivative order and mask size for smoothing must be specified.
  static void derivative(ImageBase const& image, ImageBase& result, uint32 hOrder, uint32 vOrder, uint32 maskSize);
  /// flips an image horizontal or vertical.
  static void flip(Image8U const& image, Image8U& result, bool horizontal, bool vertical);
  /// inverts the color of an image.
  static void inverse(Image8U const& image, Image8U& result);
  /// changes the luminance of the image. value has to be between -1.0 and 1.0
  static void luminance(Image8U const& image, Image8U& result, double const& value);
  /// shears an image horizontal and/or vertical.
  static void shear(Image8U const& image, Image8U& result, bool left = false, bool top = false);
  /// transpose the image.
  static void transpose(Image8U const& image, Image8U& result);
  /// Transforms an image into a new image according to an affine transformation (2x3 matrix).
  static void warpAffine(ImageBase const& image, ImageBase& result, Matrix const& transMat, bool inverse = false);
  /// Transforms an image into a new image according to an affine or projective transformation warpAffine() or warpPerspective() are chosen according to the matrix size.
  static void warpImage(ImageBase const& image, ImageBase& result, Matrix const& transMat, bool inverse = false);  
  /// Transforms an image into a new image according to a projective transformation (3x3 matrix).
  static void warpPerspective(ImageBase const& image, Image8U& result, Matrix const& transMat);
  /// The input has to be in the HSV or Y format.
  static void gammaCorrection(Image8U const& image, Image8U& result, double const& gamma);

  /// Copies the specified image area into the destination image so that the top left corner of the source area lies at the target position.
  static void copyImage(ImageBase const& source, ImageBase &dest, Rect const& area, VectorI const& target);
  /// Copies the given image into the destination image so that the top left corner of the source image lies at the target position.
  static void copyImage(ImageBase const& source, ImageBase& dest, VectorI const& target);

  // opencv wrapper functions
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void canny(ImageBase const& image, ImageBase& result, double thresh1, double thresh2);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void closing(ImageBase const& image, ImageBase& result, int maskSize);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void dilate(ImageBase const& image, ImageBase& result, int maskSize);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void erode(ImageBase const& image, ImageBase& result, int maskSize);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void findStereoCorrespondenceBM(Image8U const& left, Image8U const& right, Image16S& disparity, uint32 numDisps);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void findStereoCorrespondenceSGBM(Image8U const& left, Image8U const& right, Image16S& disparity, uint32 numDisps);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void floodFill(ImageBase& image, Rect& rect, double& area, VectorI const& seed, Color color = Color(0,0,0));
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void matchTemplate(ImageBase const& image, ImageBase& result, ImageBase const& templ);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void mul(ImageBase const& image1, ImageBase const& image2, ImageBase& result);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void opening(ImageBase const& image, ImageBase& result, int maskSize);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void orFilter(ImageBase const& one, ImageBase const& two, ImageBase& result);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void pyramidDown(ImageBase const& image, ImageBase& result);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void pyramidUp(ImageBase const& image, ImageBase& result);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void resize(ImageBase const& image, ImageBase& result, int32 mode = CV_INTER_LINEAR);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void set(ImageBase& image, double value);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void smooth(ImageBase const& image, ImageBase& result, double sigma);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void sub(ImageBase const& image1, ImageBase const& image2, ImageBase& result);
  /// OpenCV wrapper function. See the OpenCV documentation for details.
  static void threshold(ImageBase const& image, ImageBase& result, double thresh);

  // conversion functions
  /// Converts from one image format to another (e.g. 8U to 16S). Values saturate. Use abs==true for absolute values.
  static void convert(ImageBase const& src, ImageBase& dst, bool abs = false);
  /// Same as convert() but scales the source image to fit into dest.
  static void convertScale(ImageBase const& src, Image8U& dst);
};

#endif
