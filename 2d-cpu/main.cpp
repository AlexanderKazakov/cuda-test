#include <fstream>
#include <cmath>
#include <algorithm>
#include <sys/time.h>

#include "solve.cpp"


void calculate();

double mtime()
{
  struct timeval t;

  gettimeofday(&t, NULL);
  double mt = (double) (t.tv_sec) * 1000 + (double) t.tv_usec / 1000;
  return mt;
}

int main(int argc, char **argv) {
	prev = new double[(N+1)*(M+1)];
	curr = new double[(N+1)*(M+1)];
	next = new double[(N+1)*(M+1)];
	for (int i = 0; i < (N+1)*(M+1); i++) 
		prev[i] = curr[i] = next[i] = 0.0;
	
	double t1 = mtime();
	calculate();
	double t2 = mtime();
	std::cout << "Time of calculate() execution = " << t2 - t1 << std::endl;
	
	delete [] prev;
	delete [] curr;
	delete [] next;
}

template<typename T>
static void put(std::fstream &f, const T value) {
	union {
		char buf[sizeof(T)];
		T val;
	} helper;
	helper.val = value;
	std::reverse(helper.buf, helper.buf + sizeof(T));
	f.write(helper.buf, sizeof(T));
}

void save(const char *prefix, int step, double *a){
	char buffer[50];
	sprintf(buffer, "%s.%05d.vtk", prefix, step);
	std::fstream f(buffer, std::ios::out);
	if (!f) {
		std::cerr << "Unable to open file " << buffer << std::endl;
		return;
	}
	f << "# vtk DataFile Version 3.0" << std::endl;
	f << "U data" << std::endl;
	f << "BINARY" << std::endl;
	f << "DATASET STRUCTURED_POINTS" << std::endl;
	f << "DIMENSIONS " << N+1 << " " << M+1 << " 1" << std::endl;
	f << "SPACING " << h_x << " " << h_y << " 1" << std::endl;
	f << "ORIGIN 0 0 0" << std::endl;
	f << "POINT_DATA " << (N+1) * (M+1) << std::endl;
	f << "SCALARS u double" << std::endl;
	f << "LOOKUP_TABLE default" << std::endl;
	for (int j = 0 ; j < M+1; j++){
		for (int i = 0; i < N+1; i++)
			put(f, a[ind(i,j)]);
	}
	f.close();
}

/**
 * Calculate the process, save results.
 */
void calculate() {
	int counter = 0;
	double visualisationStep = T / 100;
	while (curTime <= T) {
		doNextTimeStep();
		if ( fabs(curTime - counter * visualisationStep) < 1e-5 ) {
			save("output", counter, prev);
			counter++;
		}
		curTime += tau;
 	}
}