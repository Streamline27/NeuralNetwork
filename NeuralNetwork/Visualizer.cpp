#include "Visualizer.h"


/*  ===============================================================  */
/*  =================         Constructor       ===================  */

Visualizer::Visualizer(char* path){
	this->path = path;
}


/*  ===============================================================  */
/*  =====   Method for adding series that will be dispayerd   =====  */
/*  ===============================================================  */

void Visualizer::addSeries(char* name, int size, double* seriesX, double* seriesY){
	Series s;
	s.seriesName = name;
	s.size = size;
	s.seriesX = seriesX;
	s.seriesY = seriesY;
	series.push_back(s);
}

/*  ===============================================================  */
/*  ==================     Visualize method      ==================  */
/*  ===============================================================  */

void Visualizer::visualize(char* windowCaption, char* chartName, char* labelX, char* labelY){
	string visualizerName = path+"visualizer.jar ";
	ostringstream executionCommand;

	// Adding labels
	executionCommand << visualizerName << " " << windowCaption << " " << chartName;
	executionCommand << " " << labelX << " " << labelY;

	// Adding series
	addSeriesToCommand(executionCommand);
	// Executing visualizer.jar
	system(executionCommand.str().c_str());
}


/*  ===============     Private helper methods.     ===============  */

void Visualizer::addSeriesToCommand(ostream& executionCommand){
	// if there is more than 2 series show only two
	size_t seriesToShow = ((series.size() >= 2) ? 2 : series.size());

	// Adding info about every series to execution command
	for (size_t i = 0; i < seriesToShow; i++)
	{
		Series& s = series.at(i);
		// Adding name and size
		executionCommand << " " << s.seriesName << " " << s.size;
		// Adding X-Y values
		for (int i = 0; i < s.size; i++)
		{
			executionCommand << " " << s.seriesX[i] << " " << s.seriesY[i];
		}
	}

	
	
}

