/**
 * @file   : smartptr.cu
 * @brief  : Smart pointers content/source file in CUDA C++14, 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171007  
 * @ref    :  
 * 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
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
 * nvcc -std=c++14 -lcublas -dc Feedfwd.cu -o Feedfwd.o
 * 
 * */
#include "Feedfwd.h"
/*
float computeJ_L2norm(const int m, 
	std::shared_ptr<float> & yhat, std::unique_ptr<float[], deleterRR_struct> & y) {
	
}*/

/**	@class LinReg
 * 	@brief Linear Regression  
 * */

// Constructors
LinReg::LinReg(std::vector<int> & sizeDimsvec) : 
	sizeDimsvec(sizeDimsvec) 
{
	const int Lp1 = sizeDimsvec.size(); // L=total number of Axons and so L+1 is total number of layers
	for (int l=1; l<Lp1; l++)  // l= lth Axon
	{
		int s_lm1 = sizeDimsvec[l-1];
		int s_l = sizeDimsvec[l-1];
		Axons.push_back( Axon(s_lm1,s_l) );
	}
	
}
		
// LinReg::LinReg(std::vector<Axon> & axons) : Axons(axons) {	}

// member functions
/*void LinReg::addAxon(Axon & axon) {
	Axons.push_back(axon);
}
*/

// for loading output data y 
/**
 * 	@fn load_y_from_hvec 
 * 	@brief load from host, y output data, as a std::vector<float>, column-major ordered
 * */		
void LinReg::load_y_from_hvec(std::vector<float>& h_yvec) {
	const int SIZE_Y= h_yvec.size(); 
	
	std::unique_ptr<float[], deleterRR_struct> d_y(new float[SIZE_Y], deleterRR_struct());
	y = std::move(d_y);

	cudaMallocManaged((void **) &y, SIZE_Y*sizeof(float));
	
	
}

// for loading input data X into layer 0, a0, input layer
/**
 * 	@fn load_X_from_hvec
 * 	@brief load from host, X input data, as a std::vector<float>
 * 			Since we're then given m, number of examples (in dataset), 
 * 			load_X_from_hvec will then and go ahead and point (using std::shared_ptr)
 * 			output layer l-1 of Axon l-1 to input layer l-1 of Axon l
 *  @param const int m - number of examples
 * */		
void LinReg::load_X_from_hvec(std::vector<float>& h_Xvec, const int m) 
{
	const int SIZE_X = h_Xvec.size();
	const int d = sizeDimsvec[0];
	assert( SIZE_X == m*d); // check the total size dimensions is correct for "input layer" 
	
	// first Axon
	Axons[0].load_from_hXvec( h_Xvec, m);
	Axons[0].init_al(m);

	const int Lp1 = sizeDimsvec.size(); // L=total number of Axons and so L+1 is total number of layers
	
	for (int l=2;l<Lp1; l++) {
		int idx_axon = l-1; // l=2,3...L, idx_axon=1,2,...L-1

		auto tempshptr = std::move( Axons[idx_axon-1].getal() ); // temporary shared pointer, move ownership to it temporarily
		Axons[idx_axon].load_alm1_from_ptr( tempshptr );
		Axons[idx_axon].init_al(m);
		Axons[idx_axon-1].move2al_from_ptr( tempshptr); // move ownership back to al from temporary shared ptr
	}

	this->m=m; // store the number of training examples

}

void LinReg::feedfwd(int M_x) {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers
	for (int l=1; l < Lp1;l++) {
		int idx_axon = l-1; // l=1,2,...L axons, idx_axon = 0,1,...L-1 (0-based counting for C/C++/Python
		Axons[idx_axon].rightMul();	// a^{l-1} \Theta = (a^{l-1}_i)^{j_{l-1}} \Theta_{j_{l-1}}^{j_l} =: z^l 
		Axons[idx_axon].addb(M_x);	// z^l +b = (z^l_i)^{j_l} + (b^{(l)})^{j_l} =: z^l
	}
}

float LinReg::compute_costJ_L2norm() {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers
	const int K = sizeDimsvec[Lp1-1]; // size dim. of the a^L output layer for axon L, i.e. \widehat{h}, the prediction
	const int SIZE_Y= K * m; 

//	auto y_data = std::move( y.get() ); // y data, output data
	auto yhat = Axons[Lp1-2].getal(); // L+1 - 2 = L-1 which is the last axon, when counting from 0, 0,1,...L-1
	
	// custom deleter as a STRUCT for cublasHandle 
	struct del_cublasHandle_struct {
		void operator()(cublasHandle_t* ptr) { cublasDestroy(*ptr); }
	};
	
	std::unique_ptr<cublasHandle_t,del_cublasHandle_struct> handle_u(
		new cublasHandle_t);
	cublasCreate(handle_u.get());	

	// in this scope, make res to store results from taking the difference
	std::unique_ptr<float[], deleterRR_struct> res(new float[SIZE_Y], deleterRR_struct());
	cudaMallocManaged((void **) &res, SIZE_Y*sizeof(float));


	/**
	 * @details C = \alpha op(A) + \beta op(B)
	 * lda - input - leading dim. of 2-dim. array used to store matrix A, lda = m 
	 * ldb - input - leading dim. of 2-dim. array used to store matrix B, ldb = m 
	 * ldc - input - leading dim. of 2-dim. array used to store matrix C, ldc = m 
	 * @note Why lda-ldb=ldc = m is because, for linear regression case, 
	 * 			yhat or \widehat{y} \in \text{Mat}_{\mathbb{R}}(m,K), it's a matrix of 
	 * m rows and K columns.  Since we're assuming COLUMN-major ordering, m is the "leading dim." 
	 * */
	float a1 = 1.0f;
	float bet = -1.0f; 
	cublasSgeam(*handle_u.get(), 
		CUBLAS_OP_N, CUBLAS_OP_N, m, K, &a1, 
		yhat.get(), 
//		m, &bet, y_data.get(), m, 
		m, &bet, y.get(), m, 
		res.get(), m );
					
	float costJ = 0.f;
	// do the L2 Euclidean norm element-wise
	cublasSnrm2(*handle_u.get(), SIZE_Y, res.get(), 1, &costJ);
	costJ = 0.5f*costJ*costJ/((float) m) ;
	
	// return unique_ptr for y data ownership back
//	y = std::move( y_data);
	
	return costJ;
	
}



// destructor
LinReg::~LinReg() {}
