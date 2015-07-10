#include "NNModel.h"




/* ================================================================== */
/* Constructor */
NNModel::NNModel(){
	this->layerCount = 0;
	this->hiddenLayerCount = 0;
	this->inputUnitCount = 0;
	this->outputUnitCount = 0;
	this->synapseCount = 0;
	this->isInitialized = false;
}

NNModel::NNModel(const NNModel& other){
	this->layerCount = other.layerCount;
	this->hiddenLayerCount = other.hiddenLayerCount;
	this->inputUnitCount = other.inputUnitCount;
	this->outputUnitCount = other.outputUnitCount;
	this->synapseCount = other.synapseCount;
	this->isInitialized = other.isInitialized;
	this->weights = other.weights;
	this->layerSizes = other.layerSizes;
}


/* ================================================================== */
/* =====                                                        ===== */
/* ===      This methods will be used to predict result with      === */
/* ===      respect to "weights" and "input" signals of ANN.      === */
/* =====                                                        ===== */
/* ================================================================== */

vec NNModel::predict(vec& input){
	mat i = input.t();
	return predict(i);
}

mat NNModel::predict(mat& input){
	/* Initializing input neuron signal containers*/
	mat a = input;
	
	for (int i = 0; i < layerCount-1; i++)
	{
		/* Feed forward computation */
		a.insert_cols(0, ones(input.n_rows, 1)); // Inserting bias units
		mat &theta = weights.at(i); // Getting synapse layer - theta

		mat z = a * trans(theta);
		a = sigmoid(z);
	}
	return a;
}

/* ================================================================== */
/* =====                                                        ===== */
/* =====              Methods for model evaluation.             ===== */
/* =====                                                        ===== */
/* ================================================================== */

double NNModel::getCostOver(mat& predictions, mat& labels){
	double cost = 0;
	int m = predictions.n_rows;

	for (int i = 0; i < m; i++)
	{
		vec anticipatedResult = predictions.row(i).t();
		vec actualResult = labels.row(i).t();

		cost += costFunction(anticipatedResult, actualResult);
	}
	return cost / m;

}



double NNModel::costFunction(vec& anticipatedResult, vec& actualResult){
	//  Cost function defines how well model performs on training example
	//  with respect to actual result. This function is training minimization objective.

	vec& hTheta = anticipatedResult;
	vec& Y = actualResult;

	mat J = (-(Y.t()*log(hTheta)) - (1 - Y).t() * log(1 - hTheta));
	return J(0, 0);
}

/* ================================================================== */
/* =====                                                        ===== */
/* ====        Methods for ANN atrchitecture initialization      ==== */
/* =====                                                        ===== */
/* ================================================================== */

void NNModel::addInputLayer(int numUnits){
	/* Exception handling */
	if (isInitialized) return;
	
	/* Setting up some useful variables */
	this->layerCount++;

	/* Storing layer size*/
	this->inputUnitCount = numUnits;
	this->layerSizes.insert(layerSizes.begin(), numUnits);
}

void NNModel::addHiddenLayer(int numUnits){

	/*  Exception handling  */
	if (this->inputUnitCount == 0) return;
	if (isInitialized) return;

	/*  Changing some variables */
	this->hiddenLayerCount++;
	this->layerCount++;

	/* Storing layer size*/
	layerSizes.push_back(numUnits);
}

void NNModel::addOutputLayer(int numUnits){
	/* Exception handling */
	if (hiddenLayerCount == 0) return;
	if (isInitialized) return;
	
	/* Changing variables */
	this->layerCount++;

	/* Storing layer size*/
	this->outputUnitCount = numUnits;
	layerSizes.push_back(numUnits);
}

NNModel* NNModel::constructWeights(){
	/* Exception handling */
	/*
	if (this->isInitialized) return;
	if (this->inputUnitCount == 0) return;
	if (this->hiddenLayerCount == 0) return;
	if (this->outputUnitCount == 0) return;
	*/
	/* Constructing weight layers */
	int in;  // Number of units on the left of weight layer (including bias)
	int out; // Number of units to the right (excluding bias)

	for (int i = 0; i < layerCount - 1; i++) {
		in = getLayerSize(i) + 1; out = getLayerSize(i + 1); // Getting needed dimensions
		this->weights.push_back(mat(out, in)); // Constructing synapse layer
	}

	finalizeStructure();
	return this;
}

void NNModel::randomlyInitialize(){
	/* For every matrix in weight container */
	for (int i = 0; i < weights.size(); i++)
	{
		/* epsilon - random treshold*/
		mat &m = weights.at(i); 
		double epsilon = sqrt(6) / sqrt(m.n_cols + m.n_rows);
		m.randu();
		m = m * 2 * epsilon - epsilon;
	}
}

/* ================================================================== */
/* ================================================================== */
/* =====                                                        ===== */
/* =====                   Interface methods.                   ===== */
/* =====                                                        ===== */
/* ================================================================== */
/* ================================================================== */


void NNModel::show(){
	for (int i = 0; i < weights.size(); i++)
		cout << "Layer num: " << i << endl << weights.at(i);
}

void NNModel::setWeights(vector<mat> &weightsNew){
	/* Reasigning weights*/
	if (isWeightValid(weightsNew)) weights = weightsNew;
}

vector<mat> NNModel::getWeights(){
	return weights;
}

int NNModel::getInputUnitCount(){
	return inputUnitCount;
}

int NNModel::getOutputUnitCount(){
	return outputUnitCount;
}

int NNModel::getLayerCount(){
	return layerCount;
}

int NNModel::getLayerSize(int layer){
	return layerSizes.at(layer);
}

int NNModel::getSynapseCount(){
	return synapseCount;
}

void NNModel::getParams(double* paramArray){
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

void NNModel::setParams(double* paramArray){
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




/* ================================================================== */
/* ======             Private helper functions                 ====== */
/* ================================================================== */

void NNModel::finalizeStructure(){
	/* This method is invoked after architecture construction */
	isInitialized = true;

	// Constructing neuron signal containers
	for (auto pLayerSize = layerSizes.begin(); pLayerSize!= layerSizes.end(); ++pLayerSize)
	{
		int ls = *pLayerSize; // Size of layer
		vec layer = vec(ls+1);
		layer(0) = 1; // Adding bias unit
	}

	// Counting synapses
	for (auto pWeightLayer = weights.begin(); pWeightLayer != weights.end(); ++pWeightLayer)
	{
		this->synapseCount += (pWeightLayer)->n_elem;
	}
	// Initializing weight manager
}

bool NNModel::isWeightValid(vector<mat> &weightsNew){
	/* Validating weights for setWeights method */
	if (weightsNew.size() != this->weights.size()){
		cout << "Illegal size: " << weightsNew.size() << " != " << this->weights.size() << endl;
		return false;
	}
	for (int i = 0; i < weights.size(); i++){
		/* Getting alliases for code consiseness */
		int r = weights.at(i).n_rows;
		int rn = weightsNew.at(i).n_rows;
		int c = weights.at(i).n_cols;
		int cn = weightsNew.at(i).n_cols;

		/* Validating*/
		if ((c != cn) || (r != rn)) {
			cout << "Illegal weights...";
			return false;
		}
	}
	return true;
}