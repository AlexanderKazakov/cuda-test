#include <fstream>

#include <cmath>

// Compile: "g++ task1.cpp"

const int N = 200;
const double leftCorner = 0.0;
const double rightCorner = 1.0;
const double tMax = 10.0;
const double omega = 5.0;
const double h = (rightCorner - leftCorner) / N;
const double tau = h / 2;

double c(const double &x) {
	return 0.1 + 3.6 * (x - 0.5)*(x - 0.5);
}

double hamma (const double &index) {
	double x = leftCorner + index * h;
	return 1.0 / ( c(x)*c(x) );
}

double M(const int i) {
	if (i == N) return h / 2 * hamma(N - 0.5);
	else        return h / 2 * (hamma(i - 0.5) + hamma(i + 0.5));
}

double D(const int i) {
	if (i == N) return 1.0 / c(rightCorner);
	else        return 0;
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

double f(const int i, const double &t) {
	if (i == 1) return sin(omega * t) / h;
	else        return 0;
}


int main() {
	
	double p0[N + 1]; // p^(n-1)
	double p1[N + 1]; // p^n
	double p2[N + 1]; // p^(n+1)
	for (int i = 0; i <= N; i++) {
		p0[i] = p1[i] = 0;
	}
	
	double currentT = 0; // time at n-th layer
	while (currentT < tMax) {
		p2[0] = sin(omega * (currentT + tau));
		for (int i = 1; i <= N; i++) {
			// A * p^(n+1) + B + C = d
			// B from p^(n-1), C from p^n, d from f^n
			double A = M(i) + tau / 2 * D(i);
			double B = (M(i) - tau / 2 * D(i)) * p0[i];
			double C;
			if (i == 1)
				C = tau*tau * ( K(1, 1) * p1[1] + K(1, 2) * p1[2] ) 
				    - 2 * M(1) * p1[1];
			else if (i == N)
				C = tau*tau * ( K(N, N-1) * p1[N-1] + K(N, N) * p1[N] ) 
				    - 2 * M(N) * p1[N];
			else
				C = tau*tau * ( K(i, i-1) * p1[i-1] 
				              + K(i, i)   * p1[i] 
				              + K(i, i+1) * p1[i+1] )
				    - 2 * M(i) * p1[i];
			double d = tau*tau * f(i, currentT);
			p2[i] = (d - B - C) / A;
		}
		for (int i = 0; i <= N; i++) {
			p0[i] = p1[i]; p1[i] = p2[i];
		}
		currentT += tau;
	}
	
	std::ofstream results;
	results.open("results_of_task1");
	for (int i = 0; i <= N; i++)
		results << i*h << "\t" << p2[i] << "\n";	
	results.close();
	
	return 0;
}