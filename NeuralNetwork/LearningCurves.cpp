#include "ModelEvaluation.h"

/* ---------------------------------------- */
/* ---------   Learning curves   ---------- */
int stepSize;
double testedLambda;
int numTrainingIterations;


void learningCurves(){
	/* *********************      Interface part      **********************/
	/* ********************* **** ************** **** **********************/
	int numTestedExamples;
	cout << endl << endl;
	cout << " ================== Learning curves  ===========" << endl;
	cout << " == Enter maximum size of training set: "; 	cin >> numTestedExamples;
	cout << " == Enter tested lambda: "; 	cin >> testedLambda;
	cout << " == Enter step size: "; 	cin >> stepSize;
	cout << " == Enter number of LBFGS iterations: "; 	cin >> numTrainingIterations;

	if (numTestedExamples < 150) numTestedExamples = 150;
	if (numTestedExamples > 40000) numTestedExamples = 40000;

	NNModel* model = readModel();

	/* *********************          Logic           **********************/
	/* ********************* **** ************** **** **********************/
	DataParser *traininSet = new DataParser(numTestedExamples, numFeatures, 10, "trainingSet.csv");
	DataParser *crossValidationSet = new DataParser(numCrossValidationExamples, numFeatures, 10, "crossValidation.csv");


	plotLearningCurves(model, traininSet, crossValidationSet, testedLambda);


	system("PAUSE");
	exit(0);
}




void plotLearningCurves(NNModel* model, DataParser* trainingSet, DataParser* cvSet, double lambda){


	/*  getting training set and cross validation set */
	mat& xTrain = trainingSet->getExampleSet();
	mat& yTrain = trainingSet->getLabelSet();

	mat& xCross = cvSet->getExampleSet();
	mat& yCross = cvSet->getLabelSet();

	/* Initializing some usefull variables */
	int m = xTrain.n_rows;
	int numEvaluations = ((m-stepSize) / stepSize)+1;
	
	/* Initializing data containers */
	double* errorCV = new double[numEvaluations];
	double* errorT = new double[numEvaluations];
	double* foldSizes = new double[numEvaluations];

	int currentFoldSize = stepSize;
	mat predictions;
	for (int i = 0; i < numEvaluations; i++)
	{
		/* Initializing example fold(size is dependent on iteration) */
		mat xFold = xTrain.head_rows(currentFoldSize);
		mat yFold = yTrain.head_rows(currentFoldSize);

		/* Training model over current example fold */
		NNBackPropagation * backPropagation
			= new NNBackPropagation(model, xFold, yFold, lambda);
		backPropagation->optimize(numTrainingIterations);

		model = backPropagation->getUpdatedModel();

		/* Getting error over current fold and cross validation set */
		predictions = model->predict(xFold);
		errorT[i] = model->getCostOver(predictions, yFold);

		predictions = model->predict(xCross);
		errorCV[i] = model->getCostOver(predictions, yCross);

		foldSizes[i] = currentFoldSize;

		/* Showing the results of iteration */
		cout << "   Cost for crossVal. set: " << errorCV[i] << endl;
		cout << "   Cost for training fold: " << errorT[i] << endl;
		cout << "   Training finished for fold of: " << currentFoldSize << endl << endl;

		/* Changing variables */
		currentFoldSize += stepSize;
		model->randomlyInitialize();
		delete backPropagation;
	}

	cout << " == Computation finished." << endl;
	cout << " == Visualizing data." << endl;

	/* ==================   Plotting the results    ==================  */
	ostringstream plotNameStream;
	plotNameStream << "\"" << "Learning curves: lambda = " << lambda << "\"";
	string plotName = plotNameStream.str();

	Visualizer *visualizer = new Visualizer("");
	visualizer->addSeries("\"Crossvalidation set\"", numEvaluations, foldSizes, errorCV);
	visualizer->addSeries("\"Training set\"", numEvaluations, foldSizes, errorT);
	visualizer->visualize("\"Learning curves\"", (char*)plotName.c_str(), "\"Fold size\"", "error");
}



