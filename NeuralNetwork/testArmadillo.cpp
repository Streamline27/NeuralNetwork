#include <iostream>
#include <armadillo>
#include <chrono>

using namespace std;
using namespace arma;
using namespace std::chrono;

const int ITERATIONS = 3;

void getPerformance(mat);
void testPerformance();

void testArmadillo(){
	mat matrix = mat(3, 3);
	matrix  << 1 << 2 << 3 << endr
			<< 4 << 5 << 6 << endr
			<< 7 << 8 << 9 << endr;

	
	// Matrix transpose
	cout << "Initial matrix" << endl << matrix << endl;

	cout << "transposed matrix" << endl << trans(matrix) << endl;

	// Matrix reassigment
	/*
	mat &B = matrix;
	matrix(1, 1) = 15;

	cout << "matrix Initial" << endl << matrix << endl;
	cout << "matrix assigned" << endl << B << endl; */

	// Matrix operations
	/*vec b = vec(3);
	b << 1 << 2 << 3;

	cout << "Product" << endl << matrix * b; */

	// Submatrix
	/*mat sub = matrix.submat(span(1, 2), span(1, 2));
	matrix(1, 1) = 15;
	cout << "Submatrix: " << endl << sub << endl;
	cout << matrix << endl; */

	//Matrix operations
	vec b = vec(3);
	vec res = vec(4).zeros();
	b << 1 << 2 << 3;

	cout << exp(b);

	/*
	res.head(3) = matrix * b;
	arma_rng::set_seed_random();
	res.randu();
	cout << "Product" << endl << res; 
	*/



}


void testPerformance(){
	mat matrix(3000, 3000);
	matrix.randn();

	matrix(1);

	getPerformance(matrix);
}

void getPerformance(mat m){
	cout << "Test started..." << endl;

	float total = 0;
	for (int i = 0; i < ITERATIONS; i++)
	{
		clock_t begin_time = clock();
		m = m*m;
		clock_t end_time = clock();
		total += (end_time - begin_time);
	}
	total /= ITERATIONS;

	std::cout << total / CLOCKS_PER_SEC << " Average time in seconds..." << endl;

}
