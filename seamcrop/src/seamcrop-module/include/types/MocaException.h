#ifndef TYPES_MOCA_EXCEPTION_H
#define TYPES_MOCA_EXCEPTION_H

#include "MocaTypes.h"
#include <boost/exception/all.hpp>

#include "types/Vector.h"
#include "types/ImageBase.h"
#include <string>
class Rect;
class CameraFeature;

// define "attributes" that can be added to an exception
// the struct part is just a tag. important are the parts "int" and "ErrNo"
typedef boost::error_info<struct TagErrNo, int> ErrNo;
typedef boost::error_info<struct TagErrMsg, std::string> ErrMsg;
typedef boost::error_info<struct TagErrInt, int32> ErrInt;
typedef boost::error_info<struct TagErrInt2, int32> ErrInt2;
typedef boost::error_info<struct TagErrDoub, double> ErrDouble;
typedef boost::error_info<struct TagErrRect, Rect> ErrRect;
typedef boost::error_info<struct TagErrVectorI, VectorI> ErrVectorI;
typedef boost::error_info<struct TagErrImg1, ImageBase> ErrImg1;
typedef boost::error_info<struct TagErrImg2, ImageBase> ErrImg2;
typedef boost::error_info<struct TagErrIplImg, IplImage> ErrIplImg;
typedef boost::error_info<struct TagErrCamFeature, CameraFeature> ErrCamFeature;


class MocaException : public boost::exception, public std::exception
{
 public:
  MocaException() {}
  MocaException(std::string const& message)
    {
      *this << ErrMsg(message);
    }
  MocaException(int32 errNo, std::string const& message)
    {
      *this << ErrNo(errNo) << ErrMsg(message);
    }
};


class ArgumentException : public MocaException
{
 public:
 ArgumentException() : MocaException() {}
 ArgumentException(std::string const& message) : MocaException(message) {}
 ArgumentException(int32 errNo, std::string const& message) : MocaException(errNo, message) {}
};


class IOException : public MocaException
{
 public:
 IOException() : MocaException() {}
 IOException(std::string const& message) : MocaException(message) {}
 IOException(int32 errNo, std::string const& message) : MocaException(errNo, message) {}
};


class RuntimeException : public MocaException
{
 public:
 RuntimeException() : MocaException() {}
 RuntimeException(std::string const& message) : MocaException(message) {}
 RuntimeException(int32 errNo, std::string const& message) : MocaException(errNo, message) {}
};


class NotImplementedException : public MocaException
{
 public:
 NotImplementedException() : MocaException() {}
 NotImplementedException(std::string const& message) : MocaException(message) {}
 NotImplementedException(int32 errNo, std::string const& message) : MocaException(errNo, message) {}
};


#endif
