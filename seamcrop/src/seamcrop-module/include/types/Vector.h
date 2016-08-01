#ifndef TYPES_VECTOR_H
#define TYPES_VECTOR_H

#include "MocaTypes.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

typedef boost::numeric::ublas::vector<double> Vector;
typedef boost::numeric::ublas::vector<int32> VectorI;

namespace Vector2D
{
  Vector create(double x, double y);
  Vector create(Vector v);
  VectorI create(int32 x, int32 y);
  VectorI create(VectorI v);
};

namespace Vector3D
{
  Vector create(double x, double y, double z);
  Vector create(Vector v, double z);
  VectorI create(int32 x, int32 y, int32 z);
  VectorI create(VectorI v, int32 z);
};


#endif

