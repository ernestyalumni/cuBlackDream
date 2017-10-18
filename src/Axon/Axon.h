/**
 * @file   : Axon.h
 * @brief  : Axon header file in CUDA C++14, 
 * @details : the Axon class should contain the weights (Theta or denoted W), and bias b
 * 				between each layers
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171015
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
 * nvcc -std=c++14 -lcublas -dc Axon.cu -o Axon.o
 * 
 * */
#ifndef __AXON_H__
#define __AXON_H__ 

#include <memory>  // std::shared_ptr, std::unique_ptr 
#include <vector>  // std::vector

#include "cublas_v2.h" 



/* =============== custom deleters =============== */

// field K = float; RR = real numbers, float  
auto deleterRR_lambda=[&](float* ptr){ cudaFree(ptr); };

/* custom deleter as a struct */ 
struct deleterRR_struct
{
	void operator()(float* ptr) const 
	{
		cudaFree(ptr);
	}
};

/* =============== END of custom deleters =============== */


/* =============== CUDA kernel functions =============== */
/** @fn addb
 * 	@brief add bias 
 * 	@note if this function was declared inside a class, as a class member, 
 * 			I obtained:
 * 			error: illegal combination of memory qualifiers 
 * 	@details Given (a_l)_i^{\  \  j} \in \text{Mat}_{\mathbb{R}}(m, s_l), 
 * 				we want to add a bias b, but along the "columns", b=b^j
 * 				assume (a_l) is COLUMN-major ordered.  
 *  			it is reasonable to assume m > s_l 
 * 				(i.e. number of rows, m, also representing the number of input examples, 
 * 				s_l = size dims. of "layer" l, a_l, or number of "nodes" of a_l
 * */
__global__ void addb_kernel(const int, const int, float*,const float*);


/* =============== Axon classes =============== */

/**
 *  @class Axon_u
 *  @brief Axon, using unique pointers, contains weights matrix (Theta), and bias b between 2 layers
 */
class Axon
{
	private:
		// size dims. of Theta,b - "weights" and bias
		int l; // lth layer, l=1,2,...L for L total number of layers
		int s_lm1; // size dim. of l-1th layer
		int s_l;	// size dim. of lth layer

		int m; // number of examples, m

		// custom deleter as a STRUCT for cublasHandle 
		struct del_cublasHandle_struct {
			void operator()(cublasHandle_t* ptr) { cublasDestroy(*ptr); }
		};

		// members 
		/* Theta,b - "weights" and bias
		 * treat Theta, b as unique_ptr because they belong directly to Axon */
		std::unique_ptr<float[], deleterRR_struct> Theta;
		std::unique_ptr<float[], deleterRR_struct> b;

		/* "layers" a^{l-1}, a^l, i.e. alm1, al
		 * treat as shared_ptr because other Axons would want to point to them */

		// "layers" a^{l-1}, a^l, i.e. alm1, al
		std::shared_ptr<float> alm1;
		std::shared_ptr<float> al;


	public:
		// Constructor
		Axon(const int s_lm1, const int s_l);

		// Copy Constructor
		/**
		  *  @fn Axon(const Axon& old_axon)
		  *  @brief copy constructor for Axon class
		  *  @note C++11 && token used to mean "rvalue reference", rvalue reference and move constructors
		  * @ref http://www.geeksforgeeks.org/copy-constructor-in-cpp/
		  * https://stackoverflow.com/questions/16030081/copy-constructor-for-a-class-with-unique-ptr
		  * https://en.wikipedia.org/wiki/C%2B%2B11#Rvalue_references_and_move_constructors
		  * */
		Axon( Axon &&); 
		
		// operator overload assignment = 
		Axon &operator=(Axon &&);
		
		// member functions
		// for loading given values onto Theta, b
		void load_from_hvec(std::vector<float>&,std::vector<float>& );
		
		/**
		 * 	@fn load_from_d 
		 * 	@brief (Theta,b) on device GPU -> std::vector on host 
		 * */
		void load_from_d(std::vector<float>&, std::vector<float>& );

		// for loading input data X into layer l-1, alm1
		/**
		 * 	@fn load_from_hXvec 
		 * 	@brief load from host, X input data, as a std::vector<float>
		 *  @param const int m - number of examples
		 * */
		void load_from_hXvec(std::vector<float>&, const int );

		/** We're not transferring ownership, so we don't use std::move
		 * @ref https://stackoverflow.com/questions/41871115/why-would-i-stdmove-an-stdshared-ptr
		 * */
		void load_alm1_from_ptr(std::shared_ptr<float> &);

		/** We're transferring ownership, so we  use std::move
		 * @ref https://stackoverflow.com/questions/41871115/why-would-i-stdmove-an-stdshared-ptr
		 * */
		void move2al_from_ptr(std::shared_ptr<float> & ptr_sh_output_layer) ;

		
		// initialize layer l
		/**
		 * 	@fn init_al 
		 * 	@brief initialize layer l
		 *  @param const int m - number of examples
		 * */
		void init_al(const int);

		/**
		 *  @fn getSizeDims
		 *  @brief for getting size dimensions, vector of 3 ints
		 *  @details ( s_lm1, s_l, m )
		 * */
		std::vector<int> getSizeDims();
		
		// for getting Theta,b, and lth layer al, zl (after activation function applied)
		std::unique_ptr<float[],deleterRR_struct> getTheta();
		
		std::unique_ptr<float[],deleterRR_struct> getb();

		std::shared_ptr<float> getalm1();

		std::shared_ptr<float> getal();


		/**
		 *  @fn rightMul
		 *  @class Axon_sh
		 * 	@brief right multiplication
		 * */
		void rightMul(); 

		/* ========== Add bias ========== */
		void addb(const int);


		// destructor
		~Axon();			

};






#endif 
