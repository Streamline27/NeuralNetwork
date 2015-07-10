#include "NeuralNetwork.h"

using namespace std;

/* ---------------------------------------- */
/* ------------   Training   -------------- */
//int numTrainingExamples;// = 40001;
//int numLbfgsIterations;// = 35;
//double lambda;// = 0;
//int numCrossValidationExamples = 2000;

/* ---------------------------------------- */
/* ------------   Evaluation   ------------ */


/* ---------------------------------------- */
/* ------------   Testing      ------------ */
//int testSetSize = 28000;
//char* loadedModelFileName = "model40k.nn";
//char* predictionFileName = "result.csv";
//char* testSetFileName = "test.csv";

/* ---------------------------------------- */
/* ------------   Generak      ------------ */
const int numFeatures = 784;
const int numCrossValidationExamples = 2000;


void showMenu();
void parseActions();

void main(){
	arma_rng::set_seed_random();

	showMenu();

	system("PAUSE");
}


void showMenu(){
	cout << " ================== Main menu ==================" << endl;
	cout << " ==== 1) Manual test" << endl;
	cout << " ==== 2) Train new model" << endl;
	cout << " ==== 3) Get predictions" << endl;
	cout << " ==== 4) Learning curves" << endl;
	cout << " ==== 5) Lambda selection" << endl;
	cout << " ==== 6) Exit" << endl;

	parseActions();
}


void parseActions(){
	cout << " Press number of menu to chose it." << endl;
	for (;;)
	{
		char key = _getch();
		switch (key)
		{
		case '1': 
			manualTest();
			break;
		case '2':
			trainingMode();
			break;
		case '3': 
			predictionMode();
			break;
		case '4':
			learningCurves();
			break;
		case '5':
			lambdaSelection();
			break;
		case '6':
			exit(0);
			break;
		default:
			break;
		}

	}
}