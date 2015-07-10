#include "NeuralNetwork.h"

/*  ===============================================================  */
/*  =====   Method for reading model from the console window  =====  */
/*  ===============================================================  */


NNModel* readModel(){
	int numLayers;
	NNModel* model = new NNModel();
	/* Reading general info */
	cout << endl;
	cout << " ======== Model initialization ===== " << endl;
	cout << " == Enter number of hidden layers: ";	cin >> numLayers;
	cout << " == Input layer size:  " << numFeatures << endl;
	model->addInputLayer(numFeatures);

	/* Checking if input is valid */
	if (numLayers < 1) numLayers = 1;
	if (numLayers > 3) numLayers = 3;


	/* Reading info about layers */
	for (int i = 0; i < numLayers; i++)
	{
		int hiddenLayerSize;
		cout << " == Hidden layer size " << (i + 1) << ": "; cin >> hiddenLayerSize;
		model->addHiddenLayer(hiddenLayerSize);
	}
	cout << " == Output layer size: " << 10 << endl;
	cout << endl;
	cout << endl;

	/* Finalizing structure */
	model->addOutputLayer(10);
	model->constructWeights()->randomlyInitialize();
	return model;
}