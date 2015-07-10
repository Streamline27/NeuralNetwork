//========================
// include guard defined
#ifndef __ANN_H__
#define __ANN_H__

//========================
// used dependencies

#include <iostream>
#include <armadillo>
#include <vector>
#include <math.h>
#include <conio.h>

#include "NNModel.h"
#include "MethodBP.h"
#include "NNBackPropagation.h"
#include "CSVReader.h"
#include "dataParser.h"
#include "hfunc.h"
#include "ModelLoader.h"
#include "ModelEvaluation.h"
#include "Interface.h"

using namespace std;
using namespace arma;

//========================
// used functionality

/* ---------------------------------------- */
/* --------   Global variables   ---------- */
extern const int numFeatures;
extern const int numCrossValidationExamples;

/* == Signatures == */
// In this mode user will be able to train custom model
void trainingMode();
// In this mode user will be able to get predictions over test set in file
void predictionMode();
// In this mode small window will appear and user will be able to draw digits
void manualTest();

/***************************************/
/* Working methods from
---from ModelEvaluation.h
// In this mode user will be able to see performance of model with different lambdas
-----void lambdaSelection();
// In this mode user will be able to plot learning curves
-----void learningCurves();
*/


#endif //__ANN_H__