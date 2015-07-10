//========================
// include guard defined
#ifndef __NNMODEL_H__
#define __NNMODEL_H__

//========================
// used dependencies
#include <armadillo>
#include <vector>
#include <math.h>

#include "hfunc.h"

using namespace std;
using namespace arma;

/*****************************************************************************************
NNModel is class that corresponds to a Neural Network model.
Model is defined by it inner architecture (Number and size of layers)
and weights of neuron synapses(parameters of the model).
Training neural network means finding the weight values so
the error of predictions over training examples compared to actual results
of these examples will be minimal.
*****************************************************************************************/

class NNModel{
	vector<int> layerSizes; // Vector containing sizes of neuron layers

	vector<mat> weights;    // Vector of neural network parameters (Synapse weights).

	/* Some useful variables */
	int layerCount, hiddenLayerCount;
	int inputUnitCount, outputUnitCount;
	int synapseCount;
	bool isInitialized;
public:
	/* Constructor */
	NNModel();
	NNModel(const NNModel& other);

	/* Prediction methods */
	mat predict(mat& a);
	vec predict(vec& inut);

	/* Model evaluation methods */
	double getCostOver(mat& predictions, mat& labels);

	double costFunction(vec& anticipatedResult, vec& actualResult);

	/* Model architecture construction methods*/
	void addInputLayer(int numUnits);
	void addHiddenLayer(int numUnits);
	void addOutputLayer(int numUnits);

	NNModel* constructWeights();
	void randomlyInitialize();

	/* Misc interface methods*/
	void show();

	void setParams(double* paramArray);
	void getParams(double* paramArray);

	void setWeights(vector<mat>& weightsNew);
	vector<mat> getWeights();

	int getLayerSize(int layer);
	int getInputUnitCount();
	int getOutputUnitCount();
	int getLayerCount();
	int getSynapseCount();

private:
	/* Helper functions */
	void finalizeStructure();
	bool isWeightValid(vector<mat>& weightsNew);

}; // end NNModel

#endif //__NNMODEL_H__