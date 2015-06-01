#include <fstream>
#include <iostream>

#include <complex>

// Compile: "g++ task2.cpp"

using namespace std;

const int N = 200;
const double leftCorner = 0.0;
const double rightCorner = 1.0;
const double T = 10.0;
const double omega = 5.0;
const double h = (rightCorner - leftCorner) / N;

double c(const double &x) {
	return 0.1 + 3.6 * (x - 0.5)*(x - 0.5);
}

double hamma (const double &index) {
	double x = leftCorner + index * h;
	return 1.0 / ( c(x)*c(x) );
}

double M(const int i, const int j) {
	if (i == j) {
		if (i == N) return h / 6 * 2 * hamma(N - 0.5);
		else        return h / 6 * 2 * (hamma(i - 0.5) + hamma(i + 0.5));
	} else {
		return h / 6 * hamma((i > j ? i : j) - 0.5);
	}
}

double D(const int i, const int j) {
	if ( (i == j) && (i == N) ) return 1.0 / c(rightCorner);
	else                        return 0;
}

double K(const int i, const int j) {
	if (i == j) {
		if (i == N) 
			return 1.0 / h;
		else
			return 2.0 / h;
	} else {
		return - 1.0 / h;
	}
}

/**
 * Represents the right-hand side of the system of linear equations L*u = f
 * @param i string number from 0 to N-1
 * @return f(i)
 */
complex<double> f(const int i) {
	if (i == 0) return omega*omega * h / 6 * hamma(0.5) + 1.0 / h;
	else        return 0;
}

/**
 * Represents the matrix of the system of linear equations L*u = f
 * @param i string number from 0 to N-1
 * @param j column number from 0 to N-1
 * @return L(i, j)
 */
complex<double> L(const int i, const int j) {
	return - omega*omega * M(i+1, j+1) 
	       + complex<double>(0, 1) * omega * D(i+1, j+1) 
	       + K (i+1, j+1);
}

/**
 * The function for solving a system of linear equations A*x = d
 * by reduction method. Numbers are complex.
 * The tridiagonal (!) matrix of SLE must have diagonal dominance.
 * @param _A function gives elements of matrix by indexes from 0 to N-1
 * @param _d function gives right-hand side elements by index from 0 to N-1
 * @param N size of SLE
 * @param solution pointer to array of N complex<double>'s 
 * to place the solution in
 */
void reductionMethod(complex<double> (&_A)(const int, const int), 
                     complex<double> (&_d)(const int),
                     const int N, complex<double> *solution) {
	// The size of the SLE should be K + 1 = 2^k + 1
	int K = pow(2, ((int) log2(N-1)) + 1);
	// Solution of SLE
	complex<double> *x = new complex<double>[K+1];
	// Coefficients of SLE
	complex<double> *a = new complex<double>[K+1]; // sub-diagonal
	complex<double> *b = new complex<double>[K+1]; // diagonal
	complex<double> *c = new complex<double>[K+1]; // under-diagonal
	complex<double> *d = new complex<double>[K+1]; // right-hand side
	a[0] = 0.0;     	b[0] = _A(0, 0);
	c[0] = _A(0, 1);	d[0] = _d(0);
	for (int i = 1; i < N-1; i++) {
		a[i] = _A(i, i-1);	b[i] = _A(i, i);
		c[i] = _A(i, i+1);	d[i] = _d(i);
	}
	a[N-1] = _A(N-1, N-2);	b[N-1] = _A(N-1, N-1);
	c[N-1] = 0.0;         	d[N-1] = _d(N-1);
	
	// Fill in the SLE by helping elements to make its size K+1
	for (int i = N; i <= K; i++) {
		a[i] = 0;	b[i] = 1;
		c[i] = 0;	d[i] = 0;
	}
	// Reduction - at every step current odd elements are replaced from the SLE
	for(int s = 1; s <= K/2; s *= 2) {
		// Special for 0'th element
		complex<double> C = - c[0] / b[s];
		b[0] = b[0] + C * a[s];
		d[0] = d[0] + C * d[s];
		c[0] = C * c[s];
		// Special for K'th element
		complex<double> A = - a[K] / b[K-s];
		b[K] = b[K] + A * c[K-s];
		d[K] = d[K] + A * d[K-s];
		a[K] = A * a[K-s];
		// For inner elements
		for(int j = 2*s; j <= K - 2*s; j += 2*s) {
			complex<double> A = - a[j] / b[j-s];
			complex<double> C = - c[j] / b[j+s];
			b[j] = b[j] + A * c[j-s] + C * a[j+s];
			d[j] = d[j] + A * d[j-s] + C * d[j+s];
			a[j] = A * a[j-s];
			c[j] = C * c[j+s];
		}
	}
	// Bottom of reduction - SLE 2x2
	x[0] = (d[0] * b[K] - c[0] * d[K]) / (b[0] * b[K] - a[K] * c[0]);
	x[K] = (b[0] * d[K] - a[K] * d[0]) / (b[0] * b[K] - a[K] * c[0]);
	// Reverse of reduction - odd elements from neighbor even elements
	for(int s = K/2; s >= 1; s /= 2)
		for(int j = s; j <= K-s; j += 2*s)
			x[j] = (d[j] - a[j] * x[j-s] - c[j] * x[j+s]) / b[j];
	
	for(int i = 0; i < N; i++)
		solution[i] = x[i];
	
	delete [] a;	delete [] b;
	delete [] c;	delete [] d;
	delete [] x;
}


int main() {
	
	complex<double> *u = new complex<double>[N];
	reductionMethod(L, f, N, u);
	
	ofstream results;
	results.open("results_of_task2");
	results << 0.0 << "\t" << 1.0 << "\t" << 0.0 << "\t"
	        << imag(exp(complex<double>(0, 1) * omega * T)) << "\n";
	for(int j = 0; j < N; j++) 
		results << (j+1)*h << "\t" << real(u[j]) << "\t" << imag(u[j]) << "\t"
		        << imag(u[j] * exp(complex<double>(0, 1) * omega * T)) << "\n";
	results.close();
	
	delete [] u;
	return 0;
}