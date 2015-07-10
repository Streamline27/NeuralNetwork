#include "MethodBP.h"

/* ============================================================= */
/* =========              Constructor                =========== */
/* ============================================================= */

MethodBP::MethodBP(NNModel *model)
: model(model)
{
	/* Initializing model related attributes */
	this->layerCount = model->getLayerCount();
	this->synapseCount = model->getSynapseCount();
	this->inputUnitCount = model->getInputUnitCount();
	this->outputUnitCount = model->getOutputUnitCount();

	/* Initializing containers */
	this->weights = (model->getWeights());
	this->derivatives = *getEmptyStructure(weights);

	/* Initialize activation containers */
	constructActivations();
}

void MethodBP::constructActivations(){
	int numUnits = 0;

	/* Allocating neuron signal containers */
	for (int i = 0; i < layerCount; i++) {
		numUnits = model->getLayerSize(i); // getting size of currently processed layer

		vec layer = zeros<vec>(numUnits + 1);
		layer(0) = 1;

		deltas.push_back(layer);	  // Adding bias delta vector
		activations.push_back(layer); // Adding bias layer vector
	}
}


/* ============================================================= */
/* ===                                                       === */
/* ===        Back Propagation and Cost function.            === */
/* ===                                                       === */
/* ============================================================= */



void MethodBP::computeCostGradient(mat &trainingSet, mat &resultSet, double lambda){
	/* Reseting cost and gradient variables. */
	cost = 0;
	derivatives = *getEmptyStructure(weights);
	int m = trainingSet.n_rows;

	/* Iterating through every training example */
	for (int i = 0; i < m; i++)
	{

		/*  --- --- ---  Feed forward propagation  --- --- ---  */
		vec trainingExample = trainingSet.row(i).t(); // Training example in vector form

		vec& anticipatedResult = forwardPropagation(trainingExample); // FP computes activations and anticipation
		vec  actualResult = resultSet.row(i).t(); // Getting actual result
		

		/*  --- --- ---  Computing cost function   --- --- ---  */
		cost += costFunction(anticipatedResult, actualResult); 


		/* --- --- ---     Back propagation         --- --- --- */
		vec error = anticipatedResult - actualResult;
		backPropagation(error); // BP computes delta terms needed to compute derivatives


		/*   --- --- ---   Computing derivatives   --- --- ---  */
		accumulateDerivatives(derivatives);
	}

	// -- Regularization
	if (lambda != 0){
		regularizeCost(cost, lambda);
		regularizeDerivatives(derivatives, lambda);
	}


	averageCostDerivatives(cost, derivatives, m);
	
};

double MethodBP::costFunction(vec& anticipatedResult, vec& actualResult){
	//  Cost function defines how well model performs on training example
	//  with respect to actual result. This function is NN minimization objective

	vec& hTheta = anticipatedResult;
	vec& Y = actualResult;

	mat J = (-(Y.t()*log(hTheta)) - (1 - Y).t() * log(1 - hTheta));
	return J(0, 0);
}


vec MethodBP::forwardPropagation(vec& trainingExample){
	//    Forward propagation - method for computing prediction over input
	//  and computing activations of neurons that will be used later to compute derivatives 

	activations.at(0).tail(inputUnitCount) = trainingExample;
	for (int i = 0; i < layerCount-1; i++)
	{
		/* Getting required variables */
		mat& theta = weights.at(i);
		vec& a = activations.at(i);
		vec& z = activations.at(i + 1);

		/* computing activations in layer i+1*/
		z.tail(z.n_rows - 1) = theta*a;
		z.tail(z.n_rows - 1) = sigmoid(getBiasOf(z));
	}

	/* Last activation without bias unit will contain the result */
	vec anticipation = activations.at(layerCount - 1).tail(outputUnitCount);
	return anticipation;

}

void MethodBP::backPropagation(vec& error){
	//   Back propagation - method for computing delta terms
	//   that later will be used to compute derivatives 

	/* Assigning initial delta */
	deltas.at(layerCount - 1) = error;

	/* Computing delta terms for hidden layers */
	for (int i = layerCount-2; i > 0; i--)
	{
		/* Getting needed variables */
		vec &deltaCurrent = deltas.at(i+1);
		vec &a= activations.at(i);
		vec &deltaNew = deltas.at(i);
		mat &theta = weights.at(i);

		/* Computing delta term off i'th layer */
		deltaNew = (theta.t()*deltaCurrent);
		deltaNew = deltaNew % sigmoidGradient(a);

		/* Assigning this term */
		deltas.at(i) = getBiasOf(deltaNew);
	}
}

void MethodBP::accumulateDerivatives(vector<mat> &derivatives){
	/*  This method computes derivatives and accumulate them to derivative matrices
	sum of these derivatives will be averaged later on.
	Regularization derivative part will be also added later.
	*/
	for (int i = 0; i < weights.size(); i++)
	{
		vec& a = activations.at(i);
		vec& d = deltas.at(i + 1);
		mat& synapseDerivatives = derivatives.at(i);

		synapseDerivatives += d*a.t();
	}
}

void MethodBP::regularizeCost(double &cost, double lambda){
	double regTerm = 0;
	// Computing sum of all weight squares except bias weights
	for (int i = 0; i < weights.size(); i++)
	{
		mat& thetaBias = weights.at(i);

		mat theta = thetaBias.tail_cols(thetaBias.n_cols - 1);
		regTerm += accu(pow(theta, 2));

	}
	// Computing cost regularization and adding it to cost
	cost += (regTerm*lambda)/2;
}

void MethodBP::regularizeDerivatives(vector<mat> &derivatives, double &lambda){
	/* Adding regularization derivative part */
	for (int i = 0; i < weights.size(); i++)
	{
		mat& w = derivatives.at(i);
		mat& theta = weights.at(i);
		
		mat synapseRegularization = lambda*theta;
		// Adding to all excepti bias terms
		w.tail_cols(w.n_cols - 1) += synapseRegularization.tail_cols(w.n_cols - 1);
	}
}

void MethodBP::averageCostDerivatives(double &cost, vector<mat> &derivatives, int &m){
	/* Averaging cost */
	cost = cost / m;
	
	/* Averaging derivatives */
	for (int i = 0; i < weights.size(); i++) derivatives.at(i) = derivatives.at(i) / m;
}


/* ============================================================= */
/* =========          Interface functions            =========== */

double MethodBP::getCost(){
	return cost;
}


int MethodBP::getParameterCount(){
	return synapseCount;
}

void MethodBP::updateModel(){
	model->setWeights(weights);
}

NNModel* MethodBP::getModel(){
	return model;
}

void MethodBP::getParams(double* paramArray){
	unsigned int index = 0;
	for (unsigned int k = 0; k < weights.size(); k++){
		mat& m = weights.at(k);
		for (unsigned int i = 0; i < m.n_rows; i++) {
			for (unsigned int j = 0; j < m.n_cols; j++) {
				paramArray[index++] = m(i, j);
			}
		}
	}
}

void MethodBP::getDerivatives(double* derArray){
	unsigned int index = 0;
	for (unsigned int k = 0; k < derivatives.size(); k++){
		mat& m = derivatives.at(k);
		for (unsigned int i = 0; i < m.n_rows; i++) {
			for (unsigned int j = 0; j < m.n_cols; j++) {
				derArray[index++] = m(i, j);
			}
		}
	}
}

void MethodBP::setParams(double* paramArray){
	unsigned int index = 0;
	for (unsigned int k = 0; k < weights.size(); k++){
		mat& m = weights.at(k);
		for (unsigned int i = 0; i < m.n_rows; i++) {
			for (unsigned int j = 0; j < m.n_cols; j++) {
				m(i, j) = paramArray[index++];
			}
		}
	}

}