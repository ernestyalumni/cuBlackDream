/**
 * @file   : logreg.cu
 * @brief  : Logistic Regression with CUDA C++14, CUBLAS, CUDA Unified Memory Management
 * @details :  class CSVRow
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171021  
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
 * nvcc -arch='sm_52' -std=c++14 -lcublas ../src/Feedfwd/Axon.o ../src/Feedfwd/activationf.o ../src/Feedfwd/Feedfwd.o ../src/FileIO/FileIO.cpp logreg.cu -o logreg.exe
 * 
 * */
 
#include <string> 	// std::string

#include "../src/FileIO/FileIO.h"			// csv2fvec, h_flatten_colmaj
#include "../src/Feedfwd/Feedfwd.h"

#include "../src/Axon/activationf.h"

//#include "cublas_v2.h" 

		// custom deleter as a STRUCT for cublasHandle 
		struct del_cublasHandle_struct {
			void operator()(cublasHandle_t* ptr) { cublasDestroy(*ptr); }
		};

 
int main(int argc, char* argv[]) { 

	/* =============== ex2data1.txt =============== */ 
	std::string filename_ex2data1 = "../data/ex2data1.txt";
	auto fvec_ex2data1 = csv2fvec( filename_ex2data1) ;
	std::vector<std::vector<float>> X_ex1data2; // choose this type, than std::vector<float> so to generalize to multi-dim. case
	std::vector<std::vector<float>> y_ex1data2;
	for (auto row : fvec_ex2data1) { 
		std::vector<float> X_i = { row[0], row[1] }; // first 2 "columns" is input data X \in \mathbb{R}^m, d=2 features
		std::vector<float> y_i = { row[2] }; // second "column" is output data y \in \mathbb{R}^m, K=1 dim.
		X_ex1data2.push_back(X_i);	
		y_ex1data2.push_back(y_i);	
	}

	int d = X_ex1data2[0].size(); // number of features
	int K = y_ex1data2[0].size(); // dim. of output
	int m = X_ex1data2.size(); 	// m = number of training examples
	
	int Mx = 128;
	
	std::cout << std::endl << " For ex2data1.txt : " << std::endl; 
	std::cout << " d = " << d << ", K = " << K << ", m = " << m << std::endl;

	auto X_ex2data1_colmaj = h_flatten_colmaj(X_ex1data2); 
	auto y_ex2data1_colmaj = h_flatten_colmaj(y_ex1data2); 

	/* sanity check 
	for (auto ele : X_ex2data1_colmaj) { 
	std::cout << ele << " " ; }
	*/
	
	std::vector<int> FFsizeDims = { d,K }; 
	// pick sigmoid function
	std::vector<int> FFactfs = { 1 };

	LogisticReg logreg( FFsizeDims, FFactfs );

	// Initialize fitting parameters with 0
	std::vector<float> h_Theta(d*K, 0.f);
	std::vector<float> h_b(K,0.f );
	std::vector<std::vector<float>> h_Thetab;
	h_Thetab.push_back(h_Theta);
	h_Thetab.push_back(h_b);

	logreg.load_from_hThetaBs( h_Thetab);
	
	// sanity check
	// it WORKS
//	for (auto ele : y_ex2data1_colmaj) { std::cout << ele << " " ; } std::cout << std::endl;

	logreg.load_y_from_hvec(y_ex2data1_colmaj);

	// sanity check
	// it WORKS
/*	auto ycheck = logreg.gety();
	std::vector<float> hycheck(m*K,0.f);
	cudaMemcpy(hycheck.data(), ycheck.get(), m*K *sizeof(float),cudaMemcpyDeviceToHost);
	for (auto ele : hycheck) { 
		std::cout << ele << " " ; } std::cout << std::endl;
*/

	logreg.load_X_from_hvec(X_ex2data1_colmaj, m);
	// sanity check
/*	auto alm1check = logreg.getalm1(1);
	std::vector<float> halm1check(m*d,0.f);
	cudaMemcpy(halm1check.data(), alm1check.get(), m*d *sizeof(float),cudaMemcpyDeviceToHost);
	for (auto ele : X_ex2data1_colmaj) { std::cout << ele << " " ; } std::cout << std::endl;
	for (auto ele : halm1check) { 
		std::cout << ele << " " ; } std::cout << std::endl;
*/
	// sanity check
/*	auto alcheck = logreg.getal(1);
	sigmoid_kernel<<<(m*K+128-1)/128,128>>>(m*K, alcheck.get()) ;
	std::vector<float> halcheck(m*K,0.f);
	cudaMemcpy(halcheck.data(), alcheck.get(), m*K *sizeof(float),cudaMemcpyDeviceToHost);
	for (auto ele : halcheck) { 
		std::cout << ele << " " ; } std::cout << std::endl;
*/

	logreg.feedfwd(128);
	// sanity check
	// it WORKS
/*	auto alcheck = logreg.getal(1);
	std::vector<float> halcheck(m*K,0.f);
	cudaMemcpy(halcheck.data(), alcheck.get(), m*K *sizeof(float),cudaMemcpyDeviceToHost);
	for (auto ele : halcheck) { 
		std::cout << ele << " " ; } std::cout << std::endl;
*/

	// sanity check
	// this WORKS
/*	float result_logregcost = 0.f; 
	result_logregcost = logreg.compute_costJ_xent(128);
	std::cout << " costJ for cross-entropy function : " << result_logregcost << std::endl;
*/


	logreg.grad_desc_step(1.0,128); 

	// this WORKS
/*	auto alcheck = logreg.getal(1);
	std::vector<float> halcheck(m*K,0.f);
	cudaMemcpy(halcheck.data(), alcheck.get(), m*K *sizeof(float),cudaMemcpyDeviceToHost);
	for (auto ele : halcheck) { 
		std::cout << ele << " " ; } std::cout << std::endl;

	auto zlcheck = logreg.getzl(1);
	std::vector<float> hzlcheck(m*K,0.f);
	cudaMemcpy(hzlcheck.data(), zlcheck.get(), m*K *sizeof(float),cudaMemcpyDeviceToHost);
	for (auto ele : hzlcheck) { 
		std::cout << ele << " " ; } std::cout << std::endl;
*/

	// sanity check for grad_desc_step
	const int SIZE_Y= K * m; 
	int Nx = (SIZE_Y + Mx -1)/Mx;
/*	std::unique_ptr<float[], deleterRR_struct> Delta(new float[SIZE_Y], deleterRR_struct());
	cudaMallocManaged((void **) &Delta, SIZE_Y*sizeof(float));
	// this WORKS
	Deltaxent_kernel<<<Nx,Mx>>>(SIZE_Y, logreg.gety().get(), logreg.getal(1).get(), Delta.get() );
	std::vector<float> h_Delta(m*K, 0.f);
//	cudaMemcpy(h_Delta.data(), Delta.get(), m*K *sizeof(float),cudaMemcpyDeviceToHost);
	// this works 
/*	auto Dpsilcheck = std::move( logreg.getDpsil(1) ); 
	cudaMemcpy(h_Delta.data(), Dpsilcheck.get(), m*K *sizeof(float),cudaMemcpyDeviceToHost);
*/
	// this WORKS
/*	HadamardMultiply_kernel<<<Nx,Mx>>>(SIZE_Y, 
		logreg.getDpsil(1).get(), Delta.get());
*/
	// sanity check
/*	cudaMemcpy(h_Delta.data(), Delta.get(), m*K *sizeof(float),cudaMemcpyDeviceToHost);

	for (auto ele : h_Delta) { 
		std::cout << ele << " " ; } std::cout << std::endl;
*/
	float a1 = 1.0f/ ((float) m);
	float bet = 0.f;
	
	const int SIZE_dTHETA = d*K;
/*	std::unique_ptr<float[], deleterRR_struct> dTheta(new float[SIZE_dTHETA], deleterRR_struct());
	cudaMallocManaged((void **) &dTheta, SIZE_dTHETA*sizeof(float));

	std::unique_ptr<cublasHandle_t,del_cublasHandle_struct> handle_u(
		new cublasHandle_t);
	cublasCreate(handle_u.get());	
	
//	auto alm1check = logreg.getalm1(1);

/* this WORKS 	
	cublasSgemm(*handle_u.get(),
		CUBLAS_OP_T, CUBLAS_OP_N, d, K, m, &a1, 
		logreg.getalm1(1).get(), m, 
		Delta.get(), m , 
			&bet, dTheta.get(), d);

	cudaMemcpy(h_Theta.data(), dTheta.get(), d*K *sizeof(float),cudaMemcpyDeviceToHost);
	for (auto ele : h_Theta) { 
		std::cout << ele << " " ; } std::cout << std::endl;
*/


	// sanity check
	// this WORKS
/*	auto ptr_Theta_check = logreg.getTheta(1);
	auto ptr_b_check = logreg.getb(1); 
	cudaMemcpy(h_Theta.data(), ptr_Theta_check.get(), d*K *sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b.data(), ptr_b_check.get(), K *sizeof(float),cudaMemcpyDeviceToHost);
	std::cout << std::endl << " after 1 grad. desc., Theta : " << std::endl; 
	for (auto ele : h_Theta) { 
		std::cout << ele << " " ; } std::cout << std::endl;
	std::cout << std::endl << " after 1 grad. desc., b     : " << std::endl; 
	for (auto ele : h_b) { 
		std::cout << ele << " " ; } std::cout << std::endl;
*/
	/* ========== Compute and display cost and gradient with non-zero theta ========== */
	h_Theta[0] = 0.2f;
	h_Theta[1] = 0.2f;
	h_b[0] = -24.f;  
	h_Thetab[0] = h_Theta; 
	h_Thetab[1] = h_b;
	
	logreg.load_from_hThetaBs( h_Thetab);
	logreg.feedfwd(128);
	// this WORKS 
	// logreg.grad_desc_step(1.0,128); 

	logreg.grad_desc(5000,0.001f,128);

	// this WORKS
	float result_logregcost = 0.f; 
	result_logregcost = logreg.compute_costJ_xent(128);
	std::cout << " costJ for cross-entropy function : " << result_logregcost << std::endl;	
	// 0.21833 = J 


	// sanity check
	/* fprintf('Expected cost (approx): 0.203\n');
	 * Obtained 0.203706 after 4000 iterations with alpha=0.001
	 * fprintf('Expected theta (approx):\n');
	 * fprintf(' -25.161\n 0.206\n 0.201\n');
	 * Obtained 0.196945 0.192074 -24.0001
	 * */
/*	auto ptr_Theta_check = logreg.getTheta(1);
	auto ptr_b_check = logreg.getb(1); 
	cudaMemcpy(h_Theta.data(), ptr_Theta_check.get(), d*K *sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b.data(), ptr_b_check.get(), K *sizeof(float),cudaMemcpyDeviceToHost);
	std::cout << std::endl << " after 1 grad. desc., Theta : " << std::endl; 
	for (auto ele : h_Theta) { 
		std::cout << ele << " " ; } std::cout << std::endl;
	std::cout << std::endl << " after 1 grad. desc., b     : " << std::endl; 
	for (auto ele : h_b) { 
		std::cout << ele << " " ; } std::cout << std::endl;
*/
	/*
	 * %% ============== Part 4: Predict and Accuracies ==============
	 * %  After learning the parameters, you'll like to use it to predict the outcomes
	 * %  on unseen data. In this part, you will use the logistic regression model
	 * %  to predict the probability that a student with score 45 on exam 1 and 
	 * %  score 85 on exam 2 will be admitted.
	 * %
	 * %  Furthermore, you will compute the training and test set accuracies of 
	 * %  our model.
	 * %
	 * %  Your task is to complete the code in predict.m
	 * %  Predict probability for a student with score 45 on exam 1 
	 * %  and score 85 on exam 2 
	 */
	std::vector<float> prob { 45.f, 85.f }; 
	logreg.load_X_from_hvec(prob, 1);
	logreg.feedfwd(128); 


	std::cout << std::endl 
		<< " For a student with scores 45 and 85, we predict an admission " << std::endl;

	// sanity check
	// it WORKS
	/*
	 * fprintf('Expected value: 0.775 +/- 0.002\n\n');
	 * obtained 0.766516
	 * */
	
	auto alcheck = logreg.getal(1);
	std::vector<float> halcheck(1*K,0.f);
	cudaMemcpy(halcheck.data(), alcheck.get(), 1*K *sizeof(float),cudaMemcpyDeviceToHost);
	for (auto ele : halcheck) { 
		std::cout << ele << " " ; } std::cout << std::endl;
	



}

