//==================================================================================
// include guard defined
#ifndef __CSVR_H__
#define __CSVR_H__

//==================================================================================
// used dependencies

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

using namespace std;
/***********************************************************************************
CSV reader is a class that can read and parse CSV files.
***********************************************************************************/
class CSVReader{
public:
	CSVReader();
	double** read(int rows, int cols, char* filename);

private:
	double** getArray(int rows, int cols);
};

#endif //__CSVR_H__