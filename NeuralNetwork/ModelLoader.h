//========================
// include guard defined
#ifndef __MLOAD_H__
#define __MLOAD_H__

//========================
// used dependencies

#include "NNModel.h"
#include <fstream>

using namespace std;
using namespace arma;

//========================
// used functionality
class ModelLoader{
public:
	ModelLoader();

	void saveModel(NNModel* model, char* path);
	NNModel* loadModel(char* path);
};

#endif //__MLOAD_H__