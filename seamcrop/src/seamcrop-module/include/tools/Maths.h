#ifndef TOOLS_MATHS_H
#define TOOLS_MATHS_H

#include "types/MocaTypes.h"
#include "types/Vector.h"
#include "types/Matrix.h"
#include <math.h>
#include <algorithm> // contains: T std::min<T>(T a, T b)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef INFINITY
#define INFINITY MOCA_DOUBLE_MAX
#endif

#ifndef NAN
#define NAN sqrt(-1)
#endif


/*
Note:

You can add/subtract/multiply-by-scaler vectors and matrices just like primitive data types. E.g.

Vector v1(3), v2(3), v;
// initialize v1, v2 here; v doesn't need any initialization or even a specified size!
v = v1 + v2;

works. If you want to multiply matrices with each other or a matrix and a vector use

result = prod(m1, m2);
result = prod(m, v);

where m1, m2, m are matrices and v is a vector (result should be a matrix or a vector depending
on the multiplication performed). The euclidean length of a vector v can be calculated by using

norm_2(v)

.
*/

namespace Maths
{
  /// Conversion function that converts a string to a generic data type using a stringstream.
  template <typename T> T fromString(std::string str)
  {
    T result;
    std::stringstream sstr(str);
    sstr >> result;
    return result;  
  }
  /// Conversion function that converts a generic data type to a string using a stringstream.
  template <typename T> std::string toString(T t)
  {
    std::stringstream sstr("");
    sstr << t;
    return sstr.str();
  }
  
  /// Returns if a value is Not a Number (e.g. 0/0).
  bool isNaN(double value);
  /// Returns the euclidean distance between vectors of any dimension.
  double distance(Vector const& p1, Vector const& p2);
  /// Returns the angle to the x-axis of a 2D vector in degrees.
  double angle(Vector const& p);
  /// Returns the angle between two 2D vectors.
  double angle(Vector const& p1, Vector const& p2);
  /// Returns the value of a gaussian function with mean mu and variance sigma at position x.
  double gaussian(double const x, double const mu, double const sigma);

  #ifdef HAVE_LIBGSL
  // =============== matrix estimation ===============
  /**
     Estimates an affine transformation from two sets of corresponding points.
     modelPoints and imagePoints must have the same size and both must contain at least 3 points (point means three dimensional vector).
     The resulting transformation will be a matrix that will transform the model points (via multiplication) into the image points.
     The given matrix must be of size 2x3.
  **/
  void leastSquaresAffine(std::vector<Vector> const& modelPoints, std::vector<Vector> const& imagePoints, Matrix& transform);
  /// Estimates an euclidean transformation (see leastSquaresAffine() for details). Matrix size 2x3, requires at least 2 point pairs.
  void leastSquaresEuclid(std::vector<Vector> const& modelPoints, std::vector<Vector> const& imagePoints, Matrix& transform);
  /// Estimates a similarity transformation (see leastSquaresAffine() for details). Matrix size 2x3, requires 2 points.
  void leastSquaresSimilar(std::vector<Vector> const& modelPoints, std::vector<Vector> const& imagePoints, Matrix& transform);
  /// Estimates a projective transformation (see leastSquaresAffine() for details). Matrix size 3x3, requires 4 points.
  void leastSquaresProj(std::vector<Vector> const& modelPoints, std::vector<Vector> const& imagePoints, Matrix& transform);
  /// Computes an exact euclidean transformation matrix (2x3). Requires exactly 2 point correspondences.
  void linSolveEuclid(std::vector<Vector> const& modelPoints, std::vector<Vector> const& imagePoints, Matrix& transform);
  /// Computes an exact affine transformation matrix (2x3). Exactly the first 3 point correspondences are considered.
  void linSolveAffine(std::vector<Vector> const& modelPoints, std::vector<Vector> const& imagePoints, Matrix& transform);
  #endif // HAVE_LIBGSL

  // =============== matrix and vector stuff ===============
  /// Computes a 2x2 rotation matrix. Matrix size must be at least 2x2.
  void rotationMatrix2D(double angle, Matrix& matrix);
  /// Multiplies two affine transformation matrices. Second matrix matrix will be converted to 3x3 for this purpose.
  Matrix concatAffineTrans(Matrix const& matrix1, Matrix const& matrix2);
  /// Computes the determinat of a 3 by 3 matrix.
  double determinant3x3(Matrix const& mat);
  /// Inverts the matrix s and stores the result in d. Matrices must be n by n
  void invertMatrix(Matrix const& s, Matrix& d);
  /// Solves the linear equation system A*x = b
  void linSolveQR(Matrix const& a, Vector const& b, Vector& x);
  /// Creates a 3x3 matrix from a 2x3 matrix by adding 0, 0, 1 as the last row.
  Matrix squareMatFromAffine(Matrix const& matrix);
};


#endif // TOOLS_MATHS_H

