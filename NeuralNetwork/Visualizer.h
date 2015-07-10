//===========================================================
// include guard defined
#ifndef __VIS_H__
#define __VIS_H__

//===========================================================
// used dependencies

#include <iostream>
#include <string>
#include <sstream>
#include <vector>

using namespace std;
/*****************************************************************************************
Visualizer is the class corresponding to interface between "visualizer.jar"
and this c++ program.
This class can show up to two charts in a special chart window.

To use add series first, then call visualize() method.
*****************************************************************************************/

/* ========  Series structure ============*/
struct Series{
	int size;
	double* seriesX;
	double* seriesY;
	string seriesName;
};

/* ========  Visuallizer class ===========*/

/* This class is ment to visualize data with
   line charts using visualizer.jar */
class Visualizer{
	string path;

	string windowCaption;
	string chartName;
	string labelX;
	string labelY;


	vector<Series> series;
public:
	Visualizer(char*  path);

	void visualize(char* windowCaption, char* chartName, char* labelX, char* labelY);

	void addSeries(char*  name, int size, double* seriesX, double* seriesY);

private:
	void addSeriesToCommand(ostream& executionCommand);
};


//============================================================
// used functionality

#endif //__VIS_H__