#ifndef FILTER_SEAMCARVINGIMAGE_H
#define FILTER_SEAMCARVINGIMAGE_H

#include "types/Image8U.h"
#include "types/Image32F.h"

#include "filter/Filter.h"
#include "tools/Timing.h"
#include "io/IO.h"

/// IN THE CURRENT STATE, THE CLASS CAN ONLY REDUCE OR ENLARGE THE WIDTH OF AN IMAGE 

class SeamCarvingImage
{
 public:
  
  /// Reduces or enlarges the width of an image based on the difference between width and target width.
  static void changeWidth(Image8U const& image, Image8U& result, int targetWidth);
   
  
  /// Computes an energy value for each pixel based on gradient energy.
  static void computeEnergy(Image8U const& image, Image32F& energy, int width);
  /// Calculates a cost value for each pixel. The costs are the minimum sum of energy values in 8-connected paths from top to the current pixel position.
  static void computeCostWidth(Image8U const& image, Image32F& energy, Image32F& costWidth, Image32F& predecessors, int width); 
  /// Marks the pixels on the path (seam) with the lowest overall cost from bottom to top.
  static void markSeamWidth(Image32F& costWidth, Image32F& energy, Image32F& predecessors, Image32F& seams, int width);
  /// Removes the marked pixels from the image and closes the resulting gap by moving up all the pixels on the right side.
  static void removeSeamsWidth(Image8U& image, Image32F& seams, int width);
  /// Duplicates the marked pixels. All pixels on the right side of the seam are moved to the left and the duplicated pixels are inserted.
  static void duplicateSeamsWidth(Image8U& image, Image32F& seams, int width);
  
  /// Copies an image in full height until the specified width. Used for simple copy operations as well as cutting of the black parts after reduction.
  static void copyWidth (Image8U const& image, Image8U& result, int width);
  /// Gets the minimum of the summed up costs in the last row of costWidth.
  static void getMinCostWidth(Image32F& costWidth, int& min, int& minPosition, int width);
  /// Gets the maximum of the summed up costs in the last row of costWidth.
  static void getMaxCostWidth(Image32F& costWidth, int& max, int width);
  /// Clears an image by writing 0 in each position (only width and heigth, no channels).
  static void clearImage(Image32F& image);
  
  // Helper classes for debugging and illustration
  /// Draws the marked seams into an image.
  static void drawSeams(Image8U& image, Image8U& drawImage, Image32F& seams, int width);
  
  protected:
  // These classes are used for straight edge protection
  /// Detects lines in an image by using the Canny Edge Detector and Hough Transformation (OpenCV)
  static void detectLines();
  
};

#endif
