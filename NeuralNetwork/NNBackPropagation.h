//===========================================================
// include guard defined
#ifndef __NNBP_H__
#define __NNBP_H__
//===========================================================
// used dependencies
#include "stdafx.h"
#include "optimization.h"

#include"NNModel.h"
#include"MethodBP.h"

using namespace alglib;
using namespace arma;

/**********************************************************************************************************
NNBackPropagation is main class responsible for Neural Network learning.
It uses MethodBP(Back Propagation method) to compute derivatives and minimization objective(cost function)
of trained model.

Call optimize to start process of learning paraeters(synapse weighs) of Neural Network model.
oprimize method will run LBFGS optimization method for defined number of iterations.
Running LBFGS for more iterations usually results in better convergence but however may cause
the model no overfit the training set.
***********************************************************************************************************/
class NNBackPropagation{
	mat& trainingSet;
	mat& resultSet;
	double lambda;		// Regularization parameter.

	NNModel* model;     // Trained model.
	MethodBP *methodBP; // Training method.
public:
	// ====================   Constructor   ===================== 
	NNBackPropagation(NNModel *model, mat& trainingSet, mat& resultSet, double lambda);

	// ====================   Optimization method  ==============
	double optimize(int numIterations);

	// ====================   Misc. methods  ====================
	NNModel* getUpdatedModel();

};
#endif //__NNBP_H__