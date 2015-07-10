#include"NeuralNetwork.h"


/* ---------------------------------------- */
/* --------    Prediction mode   ---------- */
int testSetSize = 28000;
string loadedModelFileName;
string predictionFileName = "result.csv";
char* testSetFileName = "test.csv";

/* =================    Signatures   =====================*/
void saveTestResults(int* predFormated, int numPredictions);
int* getFormatedPredictions(mat&);
void predictOverTestSet();



void predictionMode(){
	/* *********************      Interface part      **********************/
	/* ********************* **** ************** **** **********************/
	cout << endl << endl;
	cout << " ================== Predicting mode  ===========" << endl;
	cout << " ==== Enter model file: ";   getline(cin, loadedModelFileName);
	cout << " ==== Enter result file: ";  getline(cin, predictionFileName);
	cout << endl << endl;
	ifstream f(loadedModelFileName);
	if (f.is_open()) predictOverTestSet();
	else {
		cout << endl << " Error: Unable to open model file." << endl << endl;
		system("PAUSE");
		exit(0);
	}	
}

void predictOverTestSet(){
	/* *********************          Logic           **********************/
	/* ********************* **** ************** **** **********************/
	DataParser testSet = DataParser(testSetSize, numFeatures, 0, testSetFileName);
	mat xTest = testSet.getExampleSet();

	/* ==== Creating model ==== */
	ModelLoader m;
	NNModel* model = m.loadModel((char*)loadedModelFileName.c_str());

	mat predictions = model->predict(xTest);
	cout << endl << "Predictions made succesfully.";

	int* predFormated = getFormatedPredictions(predictions);
	saveTestResults(predFormated, testSetSize);
	cout << endl << "Results saved successfully." << endl;
	cout << "You can find the result in file: " << predictionFileName << endl;
	system("PAUSE");
	exit(0);
}


/*   ========================================================  */
int* getFormatedPredictions(mat& predictions){
	int* predFormated = new int[predictions.n_rows];

	uword a;
	for (int i = 0; i < predictions.n_rows; i++)
	{
		predictions.row(i).max(a);
		predFormated[i] = a;
	}
	return predFormated;
}

/*   ========================================================  */
void saveTestResults(int* predFormated, int numPredictions){
	/* Opening file */
	ofstream f(predictionFileName, ios::out);

	/* Writting all the results into a file */
	if (f.is_open()){
		f << "ImageId,Label" << endl;
		for (int i = 0; i < numPredictions; i++)
			f << i+1 << ',' << predFormated[i] << endl;
	}
	else{
		cout << "Error: Unable to open result file." << endl;
	}
	f.close();

}