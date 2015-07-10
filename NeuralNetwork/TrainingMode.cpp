#include"NeuralNetwork.h"

/* ---------------------------------------- */
/* ------------   Training   -------------- */

int numTrainingExamples;// = 40001;
int numLbfgsIterations;// = 35;
double lambda;// = 0;

/* ---------------------------------------- */
/* ------------   --------   -------------- */

void showResults(mat&, mat&, mat&, mat&, NNModel*);
void saveResults(NNModel*);
void trainModel(NNModel*);

void trainingMode(){
	/* *********************      Interface part      **********************/
	/* ********************* **** ************** **** **********************/
	cout << endl << endl;

	cout << " ================== Training mode  =============" << endl;
	cout << " == Enter number of training examples: "; 	cin >> numTrainingExamples;
	cout << " == Enter number of LBFGS iterations: " ; 	cin >> numLbfgsIterations;
	cout << " == Enter lambda: "                     ; 	cin >> lambda;

	/* Checking if input is valid */
	if (numTrainingExamples > 40000) numTrainingExamples = 40000;
	if (numTrainingExamples <= 0) numTrainingExamples = 150;
	if (numLbfgsIterations < 1) numLbfgsIterations = 5;

	/* Reading model */
	//NNModel* model = readModel();
	ModelLoader l;
	NNModel* model = l.loadModel("modelL56.nn");

	trainModel(model);
}


void trainModel(NNModel* model){
	/* ===============       Reading data from files       =============== */
	DataParser *traininSet = new DataParser(numTrainingExamples, numFeatures, 10, "trainingSet.csv");
	mat xTrain = traininSet->getExampleSet();
	mat yTrain = traininSet->getLabelSet();

	DataParser *crossValidationSet = new DataParser(numCrossValidationExamples, numFeatures, 10, "crossValidation.csv");
	mat xCross = crossValidationSet->getExampleSet();
	mat yCross = crossValidationSet->getLabelSet();

	/* ===============================   Training model   =============== */
	NNBackPropagation* backPropagation = new NNBackPropagation(model, xTrain, yTrain, lambda);
	double f = backPropagation->optimize(numLbfgsIterations);
	model = backPropagation->getUpdatedModel();

	cout << "Training finished with cost: " << f << endl;

	/* ========  Showing performance ========= */
	showResults(xTrain, yTrain, xCross, yCross, model);

	/* ========  Saving results  ============= */
	saveResults(model);

	system("PAUSE");
	exit(0);
}


void showResults(mat& xTrain, mat& yTrain, mat& xCross, mat& yCross, NNModel* model){
	/* ========  Getting prediction accuracy ==========*/
	mat pred;

	/* =================   Showing result over training set =========== */
	pred = model->predict(xTrain); // Showing accuracy over training Set

	double accuracy = getPredictionAccuracy(pred, yTrain);
	double cost = model->getCostOver(pred, yTrain);

	cout << endl << "Training set prediction accuracy: " << accuracy << "%" << " (with cost: " << cost << ")" << endl;

	/* =================   Showing result over cross validation set === */
	pred = model->predict(xCross); // Showing accuracy ovet crossValidation set

	accuracy = getPredictionAccuracy(pred, yCross);
	cost = model->getCostOver(pred, yCross);

	cout << "CrossValidation set prediction accuracy: " << accuracy << "%" << " (with cost: " << cost << ")" << endl;
}

/* Asking user if he would like to save his model */
void saveResults(NNModel* model){
	cout << endl << "Would you like to save this model? (Press y/n)";

	/* Wait for y or n click*/
	char key = ' ';
	do{
		key = _getch();
	} while ((key != 'n') && (key != 'y'));
	 
	/* Saving model if needed */
	if (key == 'y'){
		cout << endl << endl << endl;
		string fileName;
		cout << "--Enter file name: "; cin >> fileName;
		
		ModelLoader m;
		m.saveModel(model, (char*)fileName.c_str());
	}
	cout << endl;
}