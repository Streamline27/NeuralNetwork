#include "NumericalGradient.h"


void computeNumericalGradient(NNModel *model, mat &trainingSet, mat &resultSet, int lambda){
	
	// Allocating stuff
	int numParams = model->getSynapseCount();

	double* parameters = new double[numParams];
	model->getParams(parameters);

	MethodBP *costCalculator = new MethodBP(model);
	costCalculator->computeCostGradient(trainingSet, resultSet, lambda);
	
	// Getting actual gradient that will be evaluated
	double*actualGradient = new double[numParams];
	costCalculator->getDerivatives(actualGradient);

	// Numgrad will contain numerical gradient
	double* numGrad = new double[numParams];
	double epsilon = 0.0001F;	// Epsilon defines precision

	// Computing numerical gradient for every parameter
	for (int i = 0; i < numParams; i++)
	{
		double JPlus = 0;
		double JMinus = 0;

		// Getting plust term
		parameters[i] += epsilon;
		costCalculator->setParams(parameters);
		costCalculator->computeCostGradient(trainingSet, resultSet, lambda);
		JPlus = costCalculator->getCost();

		// Getting minus term
		parameters[i] -= 2*epsilon;

		costCalculator->setParams(parameters);
		costCalculator->computeCostGradient(trainingSet, resultSet, lambda);
		JMinus = costCalculator->getCost();

		numGrad[i] = (JPlus - JMinus) / (2 * epsilon);

		parameters[i] += epsilon;
	}

	// Showing numGrad
	double difference = 0;
	cout << "Numerical gradient: " << endl << endl;
	for (int i = 0; i < numParams; i++)
	{
		cout << "  "<<numGrad[i] << "   " << actualGradient[i] << endl;
		difference += abs(numGrad[i] - actualGradient[i]);
	}
	cout << endl << "Difference: " << difference << endl;;
}

