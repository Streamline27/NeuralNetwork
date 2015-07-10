#include "NeuralNetwork.h"

void validateModel(string&);

void manualTest(){
	cout << endl << endl;
	cout << " ================== Manual Testing =============" << endl;
	cout << " == Enter name of model file: ";
	string name;
	getline(cin, name);
	validateModel(name);

	stringstream command;
	command << "java -jar ManualTest.jar ";
	command << name;
	cout << endl << "Launching manual test.jar..." << endl << endl;

	/* Launching ManualTest.jar */
	system(command.str().c_str());

	system("PAUSE");
	exit(0);
}

void validateModel(string &name){
	ifstream f(name);
	if (!f.is_open()){
		cout << endl << "    Error: Unable to open model file." << endl << endl;
		system("PAUSE");
		exit(0);
	}
	f.close();
}