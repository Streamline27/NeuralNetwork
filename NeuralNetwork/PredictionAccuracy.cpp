#include "ModelEvaluation.h"

/***************************************************************
*********   Method for getting prediction accuracy in %
****************************************************************/

double getPredictionAccuracy(mat& predictions, mat& labels){
	int m = predictions.n_rows;
	double accuracy = 0;
	mat anticipations = round(predictions);

	/* For each prediction */
	for (int i = 0; i < m; i++){
		/* If tere is no difference between rounded prediction and label*/
		if (accu(labels.row(i) - anticipations.row(i)) == 0) accuracy += 1;
	}
	return (accuracy / m) * 100;
}