#include "ModelLoader.h"

ModelLoader::ModelLoader(){

}

void ModelLoader::saveModel(NNModel* model, char* path){
	// Gathering necessary data from model

	int numLayers = model->getLayerCount();
	int *layerSizes = new int[numLayers];
	for (int i = 0; i < numLayers; i++)
	{
		layerSizes[i] = model->getLayerSize(i);
	}

	int synapseCount = model->getSynapseCount();
	double* parameters = new double[synapseCount];
	model->getParams(parameters);

	// Writing data to binary file
	ofstream f(path, ios::binary | ios::trunc);

	f.write((char*)&numLayers, sizeof(numLayers));
	f.write((char*)layerSizes, sizeof(int)*numLayers);
	f.write((char*)&synapseCount, sizeof(int));
	f.write((char*)parameters, sizeof(double)*synapseCount);

	f.close();

	cout << "--Model saved successfully." << endl;
}

NNModel* ModelLoader::loadModel(char* path){
	// Reading necessary data from file
	ifstream f(path, ios::binary);
	if (!f.is_open()){
		cout << "Wrong model name." << endl;
		cout << "Can't find model in specified dirrection." << endl;
	}

	int numLayers;
	f.read((char*)&numLayers, sizeof(int));
	
	int* layerSizes = new int[numLayers];
	f.read((char*)layerSizes, sizeof(int)*numLayers);

	int synapseCount;
	f.read((char*)&synapseCount, sizeof(int));

	double* parameters = new double[synapseCount];
	f.read((char*)parameters, sizeof(double)*synapseCount);

	f.close();

	// Creating model from gathered data
	NNModel* model = new NNModel();
	int g = layerSizes[0];
	model->addInputLayer(layerSizes[0]);
	for (int i = 1; i < numLayers-1; i++)
	{
		model->addHiddenLayer(layerSizes[i]);
	}
	model->addOutputLayer(layerSizes[numLayers - 1]);
	model->constructWeights();
	model->setParams(parameters);

	cout << "Model loaded successfully." << endl;

	return model;
}