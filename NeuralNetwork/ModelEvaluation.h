#include "NeuralNetwork.h"
#include "Visualizer.h"
#include "NumericalGradient.h"


/***********************************************************************************
These methods are ment to test Neural Neural network model and diagnose
high bias or high variance problems.

High variance means that Neural network is overfitting training set.
Overfitting will cause model to show small error for training set but
it will fail to generalize on examples that it haven't seen.

High bias corresponds to underfitting problem.
Underfitting means that model shows bad fit to the data.
Underfitting is usually caused by huge regularization parameter lambda,
too small training set or lack of features.
When underfitting takes place model shows huge error over training set
and also huge error over cross validation set.
************************************************************************************/

//   Use this method if you are not sure what regularization parameter will fit the best for your data.
void lambdaSelection();
void plotLambdaSelection(NNModel* model, DataParser* trainingSet, DataParser* cvSet);

//   Plotting learning curves is a good way to diagnose if your model suffers from high bias or high variance.
void learningCurves();
void plotLearningCurves(NNModel* model, DataParser* trainingSet, DataParser* cvSet, double lambda);

//   Shows precentage of correct predictions
double getPredictionAccuracy(mat& predictions, mat& labels);