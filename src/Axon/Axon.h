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

#include "activationf.h" 

#include "cublas_v2.h" 

/* =============== custom deleters =============== */

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

/** @fn setconstval_kernel
 * 	@brief set a float array of length Lx all to values of const_val 
 * 	@details cudaMemset only sets an array to 0 value; we want value of 1
 * */
__global__ void setconstval_kernel(const int, const float, float*);

/* =============== note on activation function =============== */
/*
nvlink error   : Multiple definition of 'd_activat_fs' in 'RModule.o', first defined in 'Axon.o'
nvlink error   : Multiple definition of 'D_activat_fs' in 'RModule.o', first defined in 'Axon.o'
nvlink fatal   : Internal error: duplicate relocations at same address
// function pointer type for __device__ activation functions
// pf = processing function
// the following DOES NOT work in CUDA
using activat_pf = std::add_pointer<float(float)>::type;

// array of function pointers pointing to activation functions  
* DOES NOT work in CUDA (i.e. array of function pointers) when in separate code to compile 
/* =============== END of activation functions =============== */


/* ==================== Axon classes ==================== */

/* =============== Axon class; no activation =============== */

/**
 *  @class Axon
 *  @brief Axon, using unique and shared pointers, contains weights matrix (Theta), and bias b between 2 layers
 */
class Axon
{
	// protected chosen instead of private becuase members of derived class can access protected members, but not private
	protected:
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

		// Move Constructor
		/**
		  *  @fn Axon(const Axon& old_axon)
		  *  @brief move constructor for Axon class
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

		void move2alm1_from_ptr(std::shared_ptr<float> & ptr_sh_input_layer) ;

		
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
		
		// for getting (and moving back) Theta,b, and lth layer al, zl (after activation function applied)
		std::unique_ptr<float[],deleterRR_struct> getTheta();
		
		std::unique_ptr<float[],deleterRR_struct> getb();

		void move2Theta_from_ptr(std::unique_ptr<float[], deleterRR_struct> & ) ;

		void move2b_from_ptr(std::unique_ptr<float[], deleterRR_struct> & ) ;


		/**
		 * @fn Axon::getalm1
		 * @details we don't use std::move because we don't want to change (move) 
		 * 	ownership of the pointer (and the memory it points to) because we're 
		 *  dealing with a shared_ptr (you can move it, but then we'd want to use a 
		 * 	unique_ptr; we want to share it)
		 * */
		std::shared_ptr<float> getalm1();

		/**
		 * @fn Axon::getal
		 * @details we don't use std::move because we don't want to change (move) 
		 * 	ownership of the pointer (and the memory it points to) because we're 
		 *  dealing with a shared_ptr (you can move it, but then we'd want to use a 
		 * 	unique_ptr; we want to share it)
		 * */
		std::shared_ptr<float> getal();

		/* =============== "connect" the Axon =============== */
		/* Once Axon has been setup, by the above, do the following to 
		/* "connect through" the Axon */
		/**
		 *  @fn rightMul
		 *  @class Axon_sh
		 * 	@brief right multiplication
		 * 	@ref https://stackoverflow.com/questions/25797291/should-a-virtual-c-method-implementation-in-cpp-file-be-marked-virtual
		 * 		http://en.cppreference.com/w/cpp/language/virtual
		 * 		https://www.ibm.com/support/knowledgecenter/en/SSQ2R2_8.5.1/com.ibm.tpf.toolkit.compilers.doc/ref/langref_os390/cbclr21020.htm#HDRCPLR139
		 * 	@details C++ Standard n3337 § 7.1.2/5 says: 
		 * 				The virtual specifier shall be used only in the initial declaration of a non-static class member function; 
		 * 				Virtual functions are member functions whose behavior can be overridden in derived classes. 
		 * 				As opposed to non-virtual functions, the overridden behavior is preserved even 
		 * 				if there is no compile-time information about the actual type of the class.
		 * 			Bottom line: Use virtual functions when you expect a class to be used as a base class in a derivation  
		 * 						and the derived class may override the function implementation.
		 * */
		virtual void rightMul(); 

		/* ========== Add bias ========== */
		virtual void addb(const int M_x, const int N_x=0);

		// destructor
		~Axon();			

};

/* =============== Axon class; with activation =============== */
/**
 *  @class Axon_act
 *  @brief Axon_act, using unique and shared pointers, contains weights matrix (Theta), and bias b between 2 layers
 *  @ref https://en.wikibooks.org/wiki/C%2B%2B_Programming/Classes/Inheritance
 * 			6.1.1 of Discovering Moder C++, Peter Gottschling
 * 	@details 3 types of class inheritance: public, private, and protected; 
 * 		Use keyword public to implement public inheritance; classes who inherit with keyword public from base class,
 * 		inherit all pbulic members as public members, protected data inherited as protected data, and 
 * 		private data inherited but can't be accessed directly by class
 */

class Axon_act : public Axon  
{
	protected:  
		// idx_actf, index for choice of activation function, idx_actf= 0,1,...5, see activationf.h
		int idx_actf;

		// intermediate "layer" zl 
		std::unique_ptr<float[], deleterRR_struct> zl;

		// partial derivatives of the so-called "activation-layer" with respecet to zl
		std::unique_ptr<float[], deleterRR_struct> Dpsil; 

	public:
		// Constructor
		/** 
		 *	@fn Axon_act::Axon_act
		 * 	@brief class constructor for Axon_act
		 * 	@param idx_actf, const int idx_actf, index for choice of activation function, 
		 * 		idx_actf= 0,1,...5, see activationf.h
		 *  @ref 6.1.2 Iheriting Constructors, Peter Gottschling, Discovering Modern C++
		 * https://msdn.microsoft.com/en-us/library/s16xw1a8.aspx
		 *  @details 
		 * */
		Axon_act(const int s_lm1, const int s_l, const int idx_actf);

		// Move Constructor
		/**
		  *  @fn Axon(const Axon_act& old_axon)
		  *  @brief copy constructor for Axon_act class
		  *  @note C++11 && token used to mean "rvalue reference", rvalue reference and move constructors
		  * @ref http://www.geeksforgeeks.org/copy-constructor-in-cpp/
		  * https://stackoverflow.com/questions/16030081/copy-constructor-for-a-class-with-unique-ptr
		  * https://en.wikipedia.org/wiki/C%2B%2B11#Rvalue_references_and_move_constructors
		  * */
		Axon_act( Axon_act &&); 
		
		// operator overload assignment = 
		Axon_act &operator=(Axon_act &&);

		// initialize layer l
		/**
		 * 	@fn init_al 
		 * 	@brief initialize layer l
		 *  @param const int m - number of examples
		 * */
		void init_zlal(const int);

		// for getting (and moving back) Theta,b, and lth layer al, zl (after activation function applied)
		std::unique_ptr<float[],deleterRR_struct> getzl();

		// for getting lth layer Dzl (after activation function applied)
		std::unique_ptr<float[],deleterRR_struct> getDpsil();


		/* =============== "connect" the Axon =============== */
		/* Once Axon has been setup, by the above, do the following to 
		/* "connect through" the Axon */
		/**
		 *  @fn rightMul
		 *  @class Axon_sh
		 * 	@brief right multiplication
		 * */
		void rightMul(); 

		/* ========== Add bias ========== */
		void addb(const int M_x, const int N_x=0);

		/* ========== activate with activation function ========== */
		void actf( const int M_x, const int N_x=0); 

		/* ========== partial derivatives with respect to z^l of psi^l(z^l) ========== */
		void do_Dpsi( const int M_x, const int N_x=0); 

		
};


#endif 	// Axon classes end


