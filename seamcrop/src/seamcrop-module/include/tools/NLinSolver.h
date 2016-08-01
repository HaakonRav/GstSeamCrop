#ifndef TOOLS_NLINSOLVER_H
#define TOOLS_NLINSOLVER_H

#include "types/MocaTypes.h"
#include "types/Vector.h"
#include "types/Matrix.h"
#include <boost/shared_ptr.hpp>


#ifdef HAVE_LIBGSL

/**
   This class wraps the GSL functions regarding the nonlinear least-squares fitting.
   Once function() and derivative() are implemented and a set of point correspondences is supplied,
   the solve() function will return the function parameter that match the point correspondences best.
**/
class NLinSolver
{
 public:
  /**
     Creates all the necessary GSL structures and stores the given point correspondences.
     @param modelPoints source points for the transformation. Applying the function to a vector in modelPoints will result in the corresponding vector in imagePoints.
     @param imagePoints destination points.
     @param paramCnt Specifies the number of parameters (variables) of the function to be estimated.
     @param funcCnt Sepcifies the number of components of the function to be estimated. Components are being minimized.
  **/
  NLinSolver(std::vector<Vector> const& modelPoints, std::vector<Vector> const& imagePoints, sizeType paramCnt, sizeType funcCnt);
  virtual ~NLinSolver();

  /**
     Estimates the parameters of a function and returns them in the vector given.
     @param params Output parameter. Will contain the function parameter. Vector size must match paramCnt.
  **/
  void solve(Vector& params);

 protected:
  /**
     Computes the function to be estimated at a given position. This method will be called in each processing step of solve().
     @param position The currently estimated set of parameters. The Vector size will match paramCnt. [IN]
     @param values Values of the function components to be minimized at the given position. Size will match funcCnt. [OUT]
  **/
  virtual void function(Vector const& position, Vector& values) = 0;
  /**
     Computes the derivative of the function to be estimated.
     @param position The currently estimated set of parameters of the function. Vector size will match paramCnt. [IN]
     @param jacobi Jacobi matrix (derivative) at the given position. The matrix size is funcCnt by paramCnt.
     Each row represents the paramCnt derivatives of the function.
     jacobi[i][j] is the partial derivative to parameter j of error function i.
  **/
  virtual void derivative(Vector const& position, Matrix& jacobi) = 0;
  /**
     Computes both function and derivative at the same time. May be more efficient sometimes.
     Usually this is implemented by calling both function() and derivative().
  **/
  virtual void both(Vector const& position, Vector& values, Matrix& jacobi) = 0;

  /// Sets of observations. These are constant throughout the estimation process.
  std::vector<Vector> modelPoints, imagePoints;

 private:
  class Impl;
  boost::shared_ptr<Impl> pImpl;
};

#endif // HAVE_LIBGSL


/*
  How to use an NLinSolver
  ========================

  Step 1) Don't panic!
  Step 2) RTFM! (see below)

  FM:
  NLinSolver estimates the paramCnt (p) parameters of an funcCnt-dimensional (n-D) error function that minimize the error.
  In the scenario of estimating a transformation matrix with p degrees of freedom, the parameters of the functions are the parameters of the matrix.
  In the 2D case, each point correspondence provides two equations (one for each dimension) resulting in two error functions.
  So n/2 point correspondences result in n equations (= error functions).

  function(Vector position, Vector values)
  In each iteration of the solving process, this function is called once.
  [position] is the p dimensional vector containing the current estimation of the parameters.
  [values] is the n dimensional vector that will store the values of the n error functions.

  derivative(Vector position, Matrix jacobi)
  Computes the jacobi matrix of the function given the current parameters in [position].
  [position] is the p dimensional vector containing the current parameters.
  [jacobi] is the n x p jacobi matrix of the function.
    Each row represents the p derivatives of one error function.
    jacobi[i][j] is the partial derivation for parameter j of the ith error function.

  both(...)
  Computes both jacobi matrix and error values in one step.
  Sometimes this can be more efficient.

  In each iteration, the model and image points are available for the computation.
  Those are to be considered constant.
*/


#endif // TOOLS_NLINSOLVER_H

