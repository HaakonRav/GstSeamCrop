#ifndef TOOLS_KALMANFILTER_H
#define TOOLS_KALMANFILTER_H

#include "types/MocaTypes.h"
#include "types/Matrix.h"
#include "types/Vector.h"


class KalmanFilter
{
 public:
  KalmanFilter(uint32 stateDim, uint32 obsDim);

  // setters. Probably used only once
  void setInitialState(Vector const& vect);
  void setInitialCov(Matrix const& mat);

  // setters. May be used before each update
  void setStateTrans(Matrix const& mat);
  void setObsModel(Matrix const& mat);
  void setProcCov(Matrix const& mat);
  void setObsCov(Matrix const& mat);

  // include the latest observations into the state estimation
  void updateState(Vector const& obs);

  // get current state/accuracy
  Vector getStateEstimate();
  Matrix getStateCov();

 private:
  // returns an identity matrix. Helper function.
  Matrix identity(uint32 size1, uint32 size2);

  // dimension of hidden state and observations
  uint32 stateDim, obsDim;
  // estimation of the hidden variables
  Vector state;
  // covariance matrix of the random state variable
  Matrix stateCov;
  // state transition matrix: state_t = stateTrans * state_{t-1}
  Matrix stateTrans;
  // observation model: obs_t = obsModel * state_t
  Matrix obsModel;
  // covariance matrix of the process noise
  Matrix procCov;
  // covariance matrix of the observation noise
  Matrix obsCov;
  // control input model: currently not implemented
  // Matrix contModel;
};


#endif
