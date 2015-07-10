
//======================================================
// include guard defined
#ifndef __HFUNC_H__
#define __HFUNC_H__

//======================================================
// used dependencies

#include <iostream>
#include <armadillo>
#include <vector>
#include <math.h>


using namespace std;
using namespace arma;

//======================================================
// functions that are used somewhere during computations

double sigmoid(double);
mat sigmoid(mat);
vec sigmoidGradient(vec& activation);

vector<mat>* getEmptyStructure(vector<mat>& from);
vec getBiasOf(vec a);

#endif //__HFUNC_H__