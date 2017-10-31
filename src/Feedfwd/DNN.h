/**
 * @file   : DNN.h
 * @brief  : DNN header file in CUDA C++14, 
 * @details : the Deep Neural Network (DNN) class and related functions 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171031
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
 * nvcc -std=c++14 -arch='sm_52' -lcublas -dc ../Axon/Axon.cu ../Axon/activationf.cu Feedfwd.cu
 * */
#ifndef __DNN_H__
#define __DNN_H__ 

#include <cassert> 			// assert

#include "../Axon/Axon.h"	// deleterRR_struct, setconstval_kernel

/* =============== CUDA functions =============== */


/* =============== CUDA kernel functions =============== */
__global__ void costJ_xent_kernel(const int, const float*, const float*, float*);

/**
 * @fn Deltaxent_kernel, __global__ void Deltaxent_kernel
 * @brief compute Delta for the so-called cross-entropy loss function
 * @details Compute
 * ( \widehat{y}^k_{(i)} - y_{(i)}^k )/ (\widehat{y}^k_{(i)} (1 - \widehat{y}_{(i)}^k ) ) 
*/
__global__ void Deltaxent_kernel(const int, const float*, const float*, float* ) ;

/**
 * 	@fn HadamardMultiply
 * 	@brief element-wise multiply  
 * 	@details B:= A \odot B
 * */
__global__ void HadamardMultiply_kernel(const int, const float*, float*) ;


/* ==================== Deep Neural Network (DNN) class ==================== */

/**	@class DNN
 * 	@brief Deep Neural Network (DNN; i.e. Artificial Neural Network (ANN), 
 * 		i.e. so-called "Fully Connected layers")
 * */
class DNN
{
	protected:
		std::vector<Axon_act> Axons;
		std::unique_ptr<float[], deleterRR_struct> y;
		std::vector<int> sizeDimsvec;
		std::vector<int> actfs_intvec;	// \in \mathbb{Z}^{L}, for each layer l=1,2,...L, there's an activation function associated

		int m; // number of examples in the dataset

		// this is used to calculate if have enough threads
		int MAX_SIZE_1DARR; // maximum device grid size in x-dimension

		// custom deleter as a STRUCT for cublasHandle 
		struct del_cublasHandle_struct {
			void operator()(cublasHandle_t* ptr) { cublasDestroy(*ptr); }
		};
	

	public:
		// Constructor
		/** 
		 * 	@fn DNN::DNN
		 * 	@brief Constructor for DNN class 
		 * 	@param sizeDimsvec - std::vector<int> &
		 * 	@param actfs_intvec - std::vector<int> &
		 * 	@param idx_device - const int 
		 * */
		DNN(std::vector<int> &, std::vector<int> &, const int idx_device=0);

		// member functions

		// for loading (Theta,B) values from host
		void load_from_hThetaBs(std::vector<std::vector<float>> & ) ; 		 

		// for loading output data y 
		/**
		 * 	@fn load_y_from_hvec 
		 * 	@brief load from host, y output data, as a std::vector<float>, column-major ordered
		 * */		
		void load_y_from_hvec(std::vector<float>&);

		// for loading input data X into layer 0, a0, input layer
		/**
		 * 	@fn load_X_from_hvec
		 * 	@brief load from host, X input data, as a std::vector<float>
		 *  @param const int m - number of examples
		 * */		
		void load_X_from_hvec(std::vector<float>& , const int);

		/* =============== "getting" functions =============== */
		// for getting Theta,b, and lth layer al, zl (after activation function applied), lth Axon, l=1,2,...L
		std::unique_ptr<float[],deleterRR_struct> getTheta(const int);  
		
		std::unique_ptr<float[],deleterRR_struct> getb(const int);	

		std::shared_ptr<float> getalm1(const int);	// l=1,2,...L

		std::shared_ptr<float> getal(const int);	// l=1,2,...L	

		std::unique_ptr<float[],deleterRR_struct> gety();


		/* ========== Feedforward ========== */
		/**
		 *  @fn feedfwd
		 * 	@brief Feedforward
		 * 	@param Mx, int Mx=128, default to 128 threads in a single thread block
		 * 		when adding the bias to the output layer of an axon, choose the number of threads in a single 
		 * */
		void feedfwd(int Mx=256);

		/* ========== Cost functional J ========== */
		float compute_costJ_L2norm();

		float compute_costJ_xent(const int Mx=256);


		/* ========== Gradient Descent ========== */
		/**	@fn grad_desc_step
		 *	@param Mx - number of threads in a (single) thread block in x-direction
		 * 				this is needed for setconstval_kernel, to create a vector of 1's as 
		 * 				a numerical trick for the usual (mathematical) Kronecker delta function	 
		 * */
		void grad_desc_step_logreg(const float alpha_rate=0.01f, int Mx=256);
		
		/**	@fn grad_desc
		 *	@param Mx - number of threads in a (single) thread block in x-direction
		 * 				this is needed in the following:
		 * 				in feedfwd, for addb, because we're doing "row-wise" addition of a row vector
		 * 					across a matrix, 
		 * 				and 
		 * 				in grad_desc_step, for setconstval_kernel, to create a vector of 1's as 
		 * 				a numerical trick for the usual (mathematical) Kronecker delta function	 
		 * */
		void grad_desc_logreg(const int iterations=1500, const float alpha_rate=0.01f, int M_x=256);
		
		
		// destructor
		~DNN();
		
};





/* ==================== END of Deep Neural Network class ==================== */

#endif // __DNN_H__

