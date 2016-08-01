/*
* @author Timo Sztyler
* @version 02.05.2012
*/

#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#include "types/Image8U.h"
#include "types/Image32F.h"

using namespace cv;

/*
* This class implements the histogram based contrast algorithm
*/
class HistogramSaliency 
{ 
	private: 
		void quantify();	// quantizes the input image
		void measure();		// calculate color distance
		void smooth();		// smoothing of the color space
		void build();		// create saliency map as Mat object
		Mat rawImage;		// original image
		Mat tmpImage;		// color pointer (describes the real color of each pixel)
		Mat resImage;		// color palette
		Mat couImage;		// frequency of each color
		Mat disImage;		// weighted distance
		Mat resultHC;		// result of algorithm
		std::vector<std::vector<std::pair<float, int> > > collection;	// contains each color distance
	public:
		void hContrast(Image8U const& image, Image32F& result);	// constructor (path to file)		
};