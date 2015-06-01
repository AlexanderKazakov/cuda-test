#include <iostream>

const int N = 1000;
const int M = 4000;
const double h_x = 0.01;
const double h_y = 0.005;
const double tau = h_y / 2.0;
const double T = 5.0;
double curTime = 0.0;

double *prev;
double *curr;
double *next;

using namespace std;

/**
 * @param i
 * @param j
 * @return acoustical velocity in node (i, j)
 */
double c(const int i, const int j) {
//	double x = i * h_x;
//	double y = j * h_y;
//	if ((y >= 1.0) && (y <= 1.1)) return 0.5;
	return 1.0;
}

/**
 * @param i
 * @return value of sought function in the upper part of the area 
 * at the current moment in node (i, 0)
 */
double v(const int i) {
//	double x = i * h_x;
	return 1.0;
}

/**
 * @param i x-2d-index
 * @param j y-2d-index
 * @return 1d-index
 */
int ind(const int i, const int j) {
	return j*(N+1) + i;
}

/**
 * Calculate value in node (i, j) on the next time layer
 * @param i
 * @param j
 */
void calcNode(const int i, const int j) {
	if (j == M) 
		next[ind(i, j)] = v(i);
	else if (j == 0) 
		next[ind(i, j)] = curr[ind(i, j)] 
		                + c(i, j) * tau / h_y * (curr[ind(i, j+1)] - curr[ind(i, j)]);
	else if (i == 0) 
		next[ind(i, j)] = curr[ind(i, j)] 
		                + c(i, j) * tau / h_x * (curr[ind(i+1, j)] - curr[ind(i, j)]);
	else if (i == N) 
		next[ind(i, j)] = curr[ind(i, j)] 
		                - c(i, j) * tau / h_x * (curr[ind(i, j)] - curr[ind(i-1, j)]);
	else 
		next[ind(i, j)] = 2 * curr[ind(i, j)] - prev[ind(i, j)]
	                    + c(i, j)*c(i, j) * tau*tau / (h_x*h_x) * 
		                    (curr[ind(i+1, j)] - 2*curr[ind(i, j)] + curr[ind(i-1, j)])
		                + c(i, j)*c(i, j) * tau*tau / (h_y*h_y) * 
		                    (curr[ind(i, j+1)] - 2*curr[ind(i, j)] + curr[ind(i, j-1)]);
}

/**
 * Calculate values on the next time layer in all nodes.
 * Result is written to array curr 
 */
void doNextTimeStep() {
	for (int i = 0; i <= N; i++)
		for (int j = 0; j <= M; j++)
			calcNode(i, j);
	double *tmp = prev;
	prev = curr;
	curr = next;
	next = tmp;
}
