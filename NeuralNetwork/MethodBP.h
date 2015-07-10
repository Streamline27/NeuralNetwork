//===========================================================
// include guard defined
#ifndef __BPMETH_H__
#define __BPMETH_H__

//===========================================================
// used dependencies

#include <iostream>
#include <armadillo>
#include <vector>
#include <math.h>

#include "NNModel.h"
#include "hfunc.h"

using namespace std;
using namespace arma;
/*********************************************************************************************
MethodBP is a class that implements back propagation method to compute derivatives
of neural network cost function. These derivatives later will be supplied to optimization
algorithm(To LBFGS in this case) to find the optimal parameters for model with respect to
training set.
MethodBP implements regularization tactic as a technique to deal with overfitting.
Increasing regularization parameter lambda will lead to shrinking learned parameters
not allowing model to overfit the training set. However to large lambda bay cause
underfitting.
Back propagation cold be implemented as a part of model but it was not because
model can be used for pedicting without need of allocating additional memory 
that is needed to compute derivatives.
Separate of model implementation also decreases the complexity of project.
*********************************************************************************************/
class MethodBP{
	NNModel* model;          // Model that is getting trained

	vector<vec> activations; // Neuron activation container
	vector<vec> deltas;	     // deltas container

	vector<mat> weights;     // Synapses - Neuron mapping matrices
	vector<mat> derivatives; // Derivative container

	double cost;  			 //Error of model over dataSet

	/* Useful variables */
	int layerCount;
	int synapseCount;
	int inputUnitCount;
	int outputUnitCount;


public:
	// ====================   Constructor   ===================== 
	MethodBP(NNModel* model);

	//   ------------------  Main interface methods  ------------ 
	void computeCostGradient(mat &trainingSet, mat &resultSet, double lambda);
	double getCost();
	void getDerivatives(double* derArray);

	void getParams(double* paramArray);
	void setParams(double* paramArray);

	//   --------------  Misc. interface methods  -----------------
	int getParameterCount();

	NNModel* getModel();
	void updateModel();

private:
	/* ==  --------------------------------------------------  == */
	/* ==                                                      == */
	/* ==     Each method corresponds a phase of computing     == */
	/* ==     cost and gradient of passed into class model.    == */
	/* ==                                                      == */
	/* ==  --------------------------------------------------  == */

	vec forwardPropagation(vec& trainingExample);
	double costFunction(vec& anticipatedResult, vec& actualResult);

	void backPropagation(vec& error);
	void accumulateDerivatives(vector<mat>& derivatives);

	void regularizeCost(double &cost, double lambda);
	void regularizeDerivatives(vector<mat> &derivatives, double &lambda);
	void averageCostDerivatives(double &cost, vector<mat> &derivatives, int& m);

	/*   --------------     Helper methods ----------------------   */
	void constructActivations();
};

#endif //__BPMETH_H__