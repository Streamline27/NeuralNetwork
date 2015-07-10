//==================================================================================
// include guard defined
#ifndef __DPARS_H__
#define __DPARS_H__

//==================================================================================
// used dependencies

#include <armadillo>
#include "CSVReader.h"

using namespace arma;
using namespace std;
/***********************************************************************************
Data parser is a class that allows easy acces to CSV datasets.
The necessary condition is that the firs column of CSV file contains contains
the labels of dataset examples and that numLabels is supplied correctly.
***********************************************************************************/
class DataParser{
	mat examples, results;
public:
	DataParser(int numExamples, int numParams, int numLabels, char* fileName);

	mat& getExampleSet();
	mat& getLabelSet();

	mat getExampleSet(int numExamples);
	mat getLabelSet(int numLabels);
private:
	void parseWithLabels(int numExamples, int numParams, int numLabels, char* fileName);
	void parseWithoutLabels(int numExamples, int numParams, char* fileName);

};

#endif //__DPARS_H__