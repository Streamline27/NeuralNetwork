//=======================================================
// include guard defined
#ifndef __NUMGRAD_H__
#define __NUMGRAD_H__

//=======================================================
// used dependencies
#include <iostream>
#include <armadillo>
#include <vector>
#include <math.h>

#include "hfunc.h"
#include "NNModel.h"
#include "MethodBP.h"
using namespace std;
using namespace arma;

/*****************************************************************************************
Numerical gradient is a way to get sure that your back propagation is computing derivatives
correctly. Back propagation and Numerical gradient output results should be quite similar.
Differences is a good indicator of bugs in implementation of back propagation.
Back propagation works correctly if difference is less than 1e-9.
Computing derivatives using numerical gradient is a lot more
expensive operation compared to back propagation method. That is the reason why it
is not used for derivative computation during learning algorithm.

*****************************************************************************************/
void computeNumericalGradient(NNModel *model, mat &trainingSet, mat &resultSet, int lambda);



#endif //__NUMGRAD_H__