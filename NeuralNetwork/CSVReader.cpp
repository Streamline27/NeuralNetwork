#include "CSVReader.h"

CSVReader::CSVReader(){
	
}

double** CSVReader::read(int rows, int cols, char* filename){
	ifstream csvFile(filename);
	double** resultArray = getArray(rows, cols);

	/* If it is possible to open file */
	if (csvFile.is_open()){
		string line;
		/* Getting string rows */
		for (int lineIndex = 0; lineIndex < rows; lineIndex++)
		{
			if (!getline(csvFile, line)){
				break; // Dont read if file is empty
				cout << "parameters don't match dimensions of file." << endl;
				throw new exception();
			}
			istringstream lineStream(line);
			string field;
			/* Parsing fields of string */
			for (int fieldIndex = 0; fieldIndex < cols; fieldIndex++)
			{
				if (!getline(lineStream, field, ',')){
					break; // Dont read if file is empty
					cout << "parameters don't match dimensions of file." << endl;
					throw new exception();
				}
				/* Adding readed data to corresonding array entry */
				double d = atof(field.c_str());
				resultArray[fieldIndex][lineIndex] = d;
			}
		}
	}
	else{
		cout << "Error: File " << filename << " not found." << endl << endl;
		system("PAUSE");
		exit(0);
	}
	return resultArray;
}

double** CSVReader::getArray(int rows, int cols){
	double **r = new double*[cols];
	for (int i = 0; i < cols; i++) r[i] = new double[rows];
	return r;
}