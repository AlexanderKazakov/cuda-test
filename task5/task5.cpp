#include <fstream>
#include <iostream>
#include <cstdlib>

#include "solve.cpp"

// Compile: "nvcc task5.cpp -lcusparse"

using namespace std;

const int N = 400;
const float leftCorner = 0.0;
const float rightCorner = 2.0;
const float T = 10.0;
const float omega = 5.0;
const float h = (rightCorner - leftCorner) / N;
const float mu = 100.0;

float c(const float &x) {
	if (x > 1) return 1.0;
	else       return 0.1 + 3.6 * (x - 0.5)*(x - 0.5);
}

float sigma(const float &x) {
	if (x <= 1) return 0;
	else        return mu * (x-1)*(x-1);
}

complex<float> hammaM (const float &index) {
	float x = leftCorner + index * h;
	return ( omega - complex<float>(0, 1) * sigma(x) ) / ( omega * c(x)*c(x) );
}

complex<float> hammaK(const float &index) {
	float x = leftCorner + index * h;
	return ( omega ) / ( omega - complex<float>(0, 1) * sigma(x) );
}

complex<float> M(const int i, const int j) {
	if (i == j) {
		if (i == N) return h / 6 * 2 * hammaM(N - 0.5);
		else        return h / 6 * 2 * (hammaM(i - 0.5) + hammaM(i + 0.5));
	} else {
		return h / 6 * hammaM((i > j ? i : j) - 0.5);
	}
}

complex<float> K(const int i, const int j) {
	if (i == j) {
		if (i == N) 
			return 1.0f / h * hammaK(N - 0.5);
		else
			return 1.0f / h * (hammaK(i - 0.5) + hammaK(i + 0.5));
	} else {
		return - 1.0f / h * hammaK((i > j ? i : j) - 0.5);
	}
}

/**
 * Represents the right-hand side of the system of linear equations L*u = f
 * @param i string number from 0 to N-1
 * @return f(i)
 */
complex<float> f(const int i) {
	if (i == 0) return omega*omega * h / 6 * hammaM(0.5) + 1.0f / h;
	else        return 0;
}

/**
 * Represents the matrix of the system of linear equations L*u = f
 * @param i string number from 0 to N-1
 * @param j column number from 0 to N-1
 * @return L(i, j)
 */
complex<float> L(const int i, const int j) {
	return - omega*omega * M(i+1, j+1) + K (i+1, j+1);
}

/**
 * The function for solving a system of linear equations A*x = d.
 * Numbers are complex.
 * The tridiagonal (!) matrix of SLE must have diagonal dominance.
 * @param _A function gives elements of matrix by indexes from 0 to N-1
 * @param _d function gives right-hand side elements by index from 0 to N-1
 * @param N size of SLE
 * @param solution pointer to array of N complex<float>'s 
 * to place the solution in
 */
void solveSLE(complex<float> (&_A)(const int, const int), 
                     complex<float> (&_d)(const int),
                     const int N, complex<float> *solution) {
	// Coefficients of SLE
	complex<float> *a = new complex<float>[N]; // bot-diagonal
	complex<float> *b = new complex<float>[N]; // mid-diagonal
	complex<float> *c = new complex<float>[N]; // top-diagonal
	complex<float> *d = new complex<float>[N]; // right-hand side
	a[0] = 0.0;     	b[0] = _A(0, 0);
	c[0] = _A(0, 1);	d[0] = _d(0);
	for (int i = 1; i < N-1; i++) {
		a[i] = _A(i, i-1);	b[i] = _A(i, i);
		c[i] = _A(i, i+1);	d[i] = _d(i);
	}
	a[N-1] = _A(N-1, N-2);	b[N-1] = _A(N-1, N-1);
	c[N-1] = 0.0;         	d[N-1] = _d(N-1);
	
	// Calling solving function
	if ( solve(N, c, b, a, d, solution) ) {
		fprintf(stderr, "Failed to solve the SLE by GPU\n");
		exit(-1);
	}
	
	delete [] a;	delete [] b;
	delete [] c;	delete [] d;
}


int main() {
	
	complex<float> *u = new complex<float>[N];
	solveSLE(L, f, N, u);
	
	ofstream results;
	results.open("results_of_task5");
	results << 0.0 << "\t" << 1.0 << "\t" << 0.0 << "\t"
	        << imag(exp(complex<float>(0, 1) * omega * T)) << "\n";
	for(int j = 0; j < N; j++) 
		results << (j+1)*h << "\t" << real(u[j]) << "\t" << imag(u[j]) << "\t"
		        << imag(u[j] * exp(complex<float>(0, 1) * omega * T)) << "\n";
	results.close();
	
	delete [] u;
	return 0;
}
