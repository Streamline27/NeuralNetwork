#include "dataParser.h"

DataParser::DataParser(int numExamples, int numParams, int numLabels, char* fileName){
	if (numLabels <= 0) parseWithoutLabels(numExamples, numParams, fileName);
	else				parseWithLabels(numExamples, numParams, numLabels, fileName);
}

void DataParser::parseWithLabels(int numExamples, int numParams, int numLabels, char* fileName){

	cout << "Reading data from .csv file..."<< endl;
	CSVReader * reader = new CSVReader();
	cout << "Constructing exampleSet and labelSet..." << endl;
	double** data = reader->read(numExamples, numParams + 1, fileName);

	results = zeros(numExamples, numLabels);
	examples = zeros(numExamples, numParams);

	for (int i = 0; i < numExamples; i++)
	{
		// Marking result
		int label = data[0][i];
		results(i, label) = 1;
		// Copying params to param matrix
		for (int n = 1; n < numParams + 1; n++)
		{
			examples(i, n - 1) = data[n][i];
		}
	}
	cout << "done." << endl;
}

void DataParser::parseWithoutLabels(int numExamples, int numParams, char* fileName){
	cout << "Reading data from .csv file..." << endl;
	CSVReader * reader = new CSVReader();
	cout << "Constructing exampleSet..." << endl;

	double** data = reader->read(numExamples, numParams, fileName);

	results = zeros(1, 1);
	examples = zeros(numExamples, numParams);

	for (int i = 0; i < numExamples; i++)
	{
		// Copying params to param matrix
		for (int n = 0; n < numParams; n++)
		{
			examples(i, n) = data[n][i];
		}
	}
	cout << "done." << endl;
}

mat& DataParser::getExampleSet(){
	return examples;
}

mat& DataParser::getLabelSet(){
	return results;
}

mat DataParser::getExampleSet(int numExamples){
	return examples.head_rows(numExamples);
}

mat DataParser::getLabelSet(int numExamples){
	return results.head_rows(numExamples);
}

