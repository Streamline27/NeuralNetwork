#include"hfunc.h"


double sigmoid(double d){
	return 1 / (1 + exp(-d));
}

mat sigmoid(mat d){
	return 1/ (1 + exp(-d));
}

vec sigmoidGradient(vec& activation){
	/* Sigmoid gradient of regular elements */
	vec res = (activation % (1 - activation));

	/* Computing sigmoid gradient for bias unit specially */
	double sigmoidOne = sigmoid(1);
	res(0) = sigmoidOne * (1 - sigmoidOne);

	return res;
}

vector<mat>* getEmptyStructure(vector<mat>& from){
	/* Returns vector of zero matrices of corresponging size */
	vector<mat> *to = new vector<mat>(0);
	for (int i = 0; i < from.size(); i++)
	{
		int rows = from.at(i).n_rows;
		int cols = from.at(i).n_cols;
		to->push_back(zeros<mat>(rows, cols));
	};
	return to;
}

vec getBiasOf(vec a){
	return a.tail(a.n_rows - 1);
}