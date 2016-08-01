#ifndef IO_IMAGEFILEWRITER_H
#define IO_IMAGEFILEWRITER_H

#include "io/Writer.h"

/**
   Writer that saves single images to disk.
   The file extension will determine the format (supplied formats are whatever OpenCV supplies since this is mostly a wrapper).
   Each call to putImage() will write a separate image file.
   Image files will be numbered sequentially.
**/
class ImageFileWriter : public Writer
{
 public:
  ImageFileWriter(std::string fileName, std::string ext);
  ~ImageFileWriter();

  void start();
  void stop();
  void putImage(Image8U const& image);

 protected:
  std::string fileName;
  std::string ext;
  uint32 count;
};

#endif

