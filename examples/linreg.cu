/**
 * @file   : linreg.cu
 * @brief  : Linear Regression with CUDA C++14, CUBLAS, CUDA Unified Memory Management
 * @details :  class CSVRow
 * 
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171014  
 * @ref    : Coursera, Andrew Ng, Intro. to Machine Learning, ex1, Exercises 1 of Week 2
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on 
 * physics, math, and engineering have helped students with their studies, 
 * and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material 
 * open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.  
 *  Just don't be an asshole and not give credit where credit is due.  
 * Peace out, never give up! -EY
 * 
 * */
/* 
 * COMPILATION TIP
 * nvcc -std=c++14 -lcublas ../src/FileIO/FileIO.cpp ../src/Axon/Axon.cu linreg.cu -o linreg.exe
 * 
 * */
 
#include "../src/FileIO/FileIO.h"			// csv2fvec
#include "../src/Axon/Axon.h"				// Axon_sh



int main(int argc, char* argv[]) { 
	/* =============== ex1data1.txt =============== */ 
	std::string filename_ex1data1 = "../data/ex1data1.txt";
	
	auto Xydata_ex1data1 = csv2fvec(filename_ex1data1); 
	
	std::vector<std::vector<float>> X_ex1data1; // choose this type, than std::vector<float> so to generalize to multi-dim. case
	std::vector<std::vector<float>> y_ex1data1;
	for (auto row : Xydata_ex1data1) { 
		std::vector<float> X_i = { row[0] }; // first "column" is input data X \in \mathbb{R}^m, d=1 features
		std::vector<float> y_i = { row[1] }; // second "column" is output data y \in \mathbb{R}^m, K=1 dim.

		X_ex1data1.push_back(X_i);	
		y_ex1data1.push_back(y_i);	
	}
	
	// notice that both X_ex1data1, y_ex1data1 are row-major ordered.  We want column-major ordering for CUBLAS
	
	
		
	int d = X_ex1data1[0].size(); // number of features
	int K = y_ex1data1[0].size(); // dim. of output
	int m = X_ex1data1.size(); 	// m = number of training examples

	// preprocess X_ex1data1
	// flatten X_ex1data1 into column-major ordering 
	auto X_ex1data1_colmaj = h_flatten_colmaj( X_ex1data1 );
	
	Axon Thetab(d,K);
	// Initialize fitting parameters with 0
	std::vector<float> h_Theta(d*K, 0.f);
	std::vector<float> h_b(K,0.f );

	Thetab.load_from_hvec(h_Theta,h_b);	// load the initial values of Theta, b
	Thetab.load_from_hXvec(X_ex1data1_colmaj,m); // load the X data
	Thetab.init_al(m);	// initialize the output layer


	/* ========== Compute cost functional J and residuals ========== */
	Thetab.rightMul();


	
}
