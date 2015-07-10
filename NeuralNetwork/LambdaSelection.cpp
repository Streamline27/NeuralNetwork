#include "ModelEvaluation.h"

/* ---------------------------------------- */
/* --------   Lambda selection   ---------- */
int numIterations;
int numAverage;

int numLambdas;
double *lambdas;



void lambdaSelection(){
	/* *********************      Interface part      **********************/
	/* ********************* **** ************** **** **********************/
	int numTestedExamples;
	cout << endl << endl << " ================== Lambda selection  ==========" << endl;
	cout << " == Enter number of training examples: ";	cin >> numTestedExamples;
	cout << " == Enter number of LBFGS iterations: "; cin >> numIterations;
	cout << " == Enter number of computations over fixed model: "; cin >> numAverage;
	cout << " == Enter number of lambdas: "; 	cin >> numLambdas;

	/* ===================   Validating input data    ================*/
	if (numTestedExamples > 40000) numTestedExamples = 40000;
	if (numIterations <= 0) numIterations = 8;
	if (numAverage <= 1) numAverage = 1;

	/* =============================================================== */
	/* =====   Entering array of lambdas that will be checked   ====== */
	/* =============================================================== */
	lambdas = new double[numLambdas];
	for (int i = 0; i < numLambdas; i++)
	{
		cout << " == tested lambda " << i + 1 << ": ";
		cin >> lambdas[i];
	}
	cout << endl;


	/* =========================   Reading model  ====================  */
	NNModel* model = readModel();


	/* *********************          Logic           **********************/
	/* ********************* **** ************** **** **********************/
	cout << " ========   Computation started.  ========" << endl << endl;
	DataParser *traininSet = new DataParser(numTestedExamples, numFeatures, 10, "trainingSet.csv");
	DataParser *crossValidationSet = new DataParser(numCrossValidationExamples, 784, 10, "crossValidation.csv");
	cout << endl;


	plotLambdaSelection(model, traininSet, crossValidationSet);


}


void plotLambdaSelection(NNModel* model, DataParser* trainingSet, DataParser* cvSet){
	/* ========================    Initializing error containers    =============== */
	double *errorT = new double[numLambdas];
	double *errorCV = new double[numLambdas];


	mat &xTrain = trainingSet->getExampleSet();
	mat &yTrain = trainingSet->getLabelSet();
	
	mat &xCv = cvSet->getExampleSet();
	mat &yCv = cvSet->getLabelSet();

	mat predictions;

	for (int i = 0; i < numLambdas; i++)
	{
		double lambda = lambdas[i];
		// Computing 3 times and averaging
		for (int j = 0; j < numAverage; j++)
		{
			/* =========  Training model with current lambda ======================= */
			NNBackPropagation* backPropagation
				= new NNBackPropagation(model, xTrain, yTrain, lambda);
			backPropagation->optimize(numIterations);
			model = backPropagation->getUpdatedModel();

			/* =========  Getting prediction error over training set ================ */
			predictions = model->predict(xTrain);
			errorT[i] = model->getCostOver(predictions, yTrain);

			/* =========  Getting prediction error over cross validation set ========= */
			predictions = model->predict(xCv);
			errorCV[i] = model->getCostOver(predictions, yCv);

			/* ===   Ending iteration   === */
			model->randomlyInitialize();
			delete backPropagation;
		}
		errorCV[i] /= 3;
		errorT[i] /= 3;
		
		/* ==============  Showing the results of current iteration   ================ */
		cout << "   Cost for crossVal. set: " << errorCV[i] << endl;
		cout << "   Cost for training set: " << errorT[i] << endl;
		cout << "   Training finished for lambda: " << lambda << endl << endl;
	}

	/* ==================   Visualizing data   ======================================== */
	Visualizer *visualizer = new Visualizer("");
	visualizer->addSeries("Crossvalidation-set", numLambdas, lambdas, errorCV);
	visualizer->addSeries("Training-set", numLambdas, lambdas, errorT);
	visualizer->visualize("Lambda-selection", "Lambda-selection", "lambda", "error");
}



