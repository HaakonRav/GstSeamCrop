#ifndef FILTER_SEAMCROPIMAGE_H
#define FILTER_SEAMCROPIMAGE_H

// MOCA headers 
#include "types/MocaTypes.h"
#include "types/Image8U.h"
#include "types/Image32F.h"
#include "io/IO.h"
#include "types/MocaException.h"
#include "feature/Feature.h"
#include "filter/SeamCarvingImage.h"

#include "tools/Timing.h"
 
// C++ headers 
#include <string>
#include <math.h>

/// IN THE CURRENT STATE, THE CLASS CAN ONLY REDUCE THE WIDTH OF AN IMAGE

class SeamCropImage: public SeamCarvingImage
{
  public:
    
    /// Reduces the width of an image to the target width.
    static void reduceWidth(Image8U const& image, Image8U& result, int targetWidth);
    
  //private:
    
   /// Sums up the energy values downward each column. 
   static void computeCostCropWidth(Image32F& costCrop, int width);
   /// Marks and removes seams until a threshold is met and if a condition is fullfilled.
   static void seamCarvImage(Image8U& tmpImage, int& tmpWidth, int& targetWidth);
   /// Finds the best cropping window and crops the image if a condition is fullfilled.
   static void cropImage(Image8U& tmpImage, int& tmpWidth, int& targetWidth);
   /// Summarizes the costs. Is used to find the total energy of an image (all energy values summed up).
   static void summarize(Image32F cost, int& totalCost, int width);
   
   // overwritten
   /// Computes an energy value for each pixel based on a saliency map (include/feature/Feature.h).
   static void computeEnergy(Image8U image, Image32F& energy, int tmpWidth);
   /// Removes the marked pixels from the image and closes the resulting gap by moving up all the pixels on the right side.
   static void removeSeamsWidth(Image8U& image, Image32F& energy, Image32F& seams, int width);
};

#endif