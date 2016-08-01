#ifndef FEATURE_HISTOGRAM_H
#define FEATURE_HISTOGRAM_H

/**
   \file Histogram.h
   Contains only the definition of the Histogram class.
**/

#include "types/MocaTypes.h"
#include <vector>


/**
   Allows the fast computation of histograms.
   A histogram is created by specifying the range of double values that will be counted and the number of accumulator bins.
   The class' main purpose is to perform the mapping between double values to discrete histogram bins.
**/
class Histogram
{
public:
  /// Creates a histogram with the given number of bins (binCount) that represent values in the given range.
  Histogram(double minVal, double maxVal, indexType binCount);
  
  /// Returns the value of the bin with the given integer index.
  double& bin(indexType index);
  /// Returns the value of the bin with the given integer index.
  double const& bin(indexType index) const;
  /// Returns the value of the bin corresponding to the given double value. Value must be in the histogram's range.
  double& bin(double value);
  /// Returns the value of the bin corresponding to the given double value. Value must be in the histogram's range.
  double const& bin(double value) const;

  /// Interpolates between the two bins surrounding 'value' and returns a mix of both bins.
  double binLerp(double value) const;

  /// Returns the value which the bin with the given index represents.
  double binToValue(indexType bin) const;
  /// Returns the index to which the given double value will get mapped.
  indexType valueToBin(double value) const;

  /// Getter methods
  inline double getMinVal() const { return minVal; }
  inline double getMaxVal() const { return maxVal; }

  /// Smoothes the histogram bins with a [1/4, 1/2, 1/4] filter mask.
  void smooth();
  /// Returns the double value represented by the bin with the highest entry.
  double findMaxPos() const;
  /// Returns the number of histogram bins.
  indexType size() const;
  /// Clears bins
  void clear();
  /// Returns sum of all bins
  double binSum() const;
  /// Returns the average of all bins
  double avg() const;

private:
  std::vector<double> bins; /// Histogram bins
  double minVal; /// Lowest possible value to store in the histogram
  double maxVal; /// Highest possible value to store in the histogram
};


#endif

