#include "NNBackPropagation.h"

struct PassedInfo{
	mat& x;
	mat& y;
	double lambda;
	MethodBP * methodBP;

	PassedInfo(MethodBP* methodBP, mat& x, mat& y, double lambda)
		:methodBP(methodBP), x(x), y(y), lambda(lambda){
	};
};

int iteration = 1;
/* Function that will be performed on every iteration of optimization algorithm */
void NNGrad_function(const real_1d_array &x, double &func, real_1d_array &grad, void *ptr){

	// Getting necessary parameters
	PassedInfo* info = (PassedInfo*)ptr;
	MethodBP* methodBP = info->methodBP;
	mat& trainingSet = info->x;
	mat& resultSet = info->y;
	double lambda = info->lambda;

	// Copying parameters to a non const container
	real_1d_array parametersNew = x;

	// Computing needed values with respect to new parameters
	methodBP->setParams(parametersNew.getcontent());
	methodBP->computeCostGradient(trainingSet, resultSet, lambda);

	// Setting values for further computations
	func = methodBP->getCost();
	methodBP->getDerivatives(grad.getcontent());

	cout << "Iteration number: " << iteration++ << " | with cost: " << func << endl;
}



/* =============================================== */
/* ======        Class functionality        ====== */
/* ================  Constructor  ================ */
NNBackPropagation::NNBackPropagation(NNModel *model, mat& trainingSet, mat& resultSet, double lambda)
:model(model), trainingSet(trainingSet), resultSet(resultSet), lambda(lambda)
{
	methodBP = new MethodBP(model);
}

double NNBackPropagation::optimize(int numIterations){
	/* Getting parameter count */
	int paramCount = methodBP->getParameterCount();

	/* Getting initial parametes*/
	double* initialX = new double[paramCount];
	methodBP->getParams(initialX);

	/* Storing parameters in const container*/
	const double* d = initialX;
	real_1d_array x;
	x.setcontent(paramCount, d);

	/* Allocating data that will be passed to optimization function */
	PassedInfo *info = new PassedInfo(methodBP, trainingSet, resultSet, lambda);

	/* Tunning and executing algorithm */
	double epsg = 0;
	double epsf = 0.000001;
	double epsx = 0;
	ae_int_t maxits = numIterations;;
	minlbfgsstate state;
	minlbfgsreport rep;

	iteration = 1;
	cout << "=================  Training started =================" << endl << endl;

	minlbfgscreate(3, x, state);
	minlbfgssetcond(state, epsg, epsf, epsx, maxits);
	alglib::minlbfgsoptimize(state, NNGrad_function, NULL, info);

	minlbfgsresults(state, x, rep);
	methodBP->updateModel();
	return methodBP->getCost();

}

NNModel* NNBackPropagation::getUpdatedModel(){
	methodBP->updateModel();
	return methodBP->getModel();
}

