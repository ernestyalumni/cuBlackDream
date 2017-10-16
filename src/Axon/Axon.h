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
		Axon(const int, const int);

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
		
		// initialize layer l
		void init_al(const int);

		// for getting size dimensions
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
		

		// destructor
		~Axon();			

};



/**
 *  @class Axon_sh
 *  @brief Axon, using shared pointers, contains weights matrix (Theta), and bias b between 2 layers
 */
class Axon_sh
{
	private:
		// size dims. of Theta,b - "weights" and bias
		int l; // lth layer, l=1,2,...L for L total number of layers
		int s_lm1; // size dim. of l-1th layer
		int s_l;	// size dim. of lth layer

		int m; // number of examples, m

		// member custom deleter as struct; auto lambda not allowed here
		struct deleterRR {
			void operator()(float* ptr) const
			{
				cudaFree(ptr);
			}
		};
		
		// members 
		// Theta,b - "weights" and bias
		std::shared_ptr<float> Theta;
		std::shared_ptr<float> b;

		// "layers" a^{l-1}, a^l, i.e. alm1, al
		std::shared_ptr<float> alm1;
		std::shared_ptr<float> al;


	public:
		// Constructor
		Axon_sh(const int, const int);
		
		// member functions
		// for loading given values onto Theta, b
		void load_from_hvec(std::vector<float>&,std::vector<float>& );
		
		/**
		 * 	@fn load_from_d 
		 * 	@brief (Theta,b) on device GPU -> std::vector on host 
		 * */
		void load_from_d(std::vector<float>&, std::vector<float>& );
		
		void load_from_uniq(std::unique_ptr<float[],deleterRR> &,std::unique_ptr<float[],deleterRR> &);
		
		// for loading input data X into layer l-1, alm1
		/**
		 * 	@fn load_from_hXvec 
		 * 	@brief load from host, X input data, as a std::vector<float>
		 *  @param const int m - number of examples
		 * */
		void load_from_hXvec(std::vector<float>&, const int );
		
		// initialize layer l
		void init_al(const int);

		// for getting size dimensions
		std::vector<int> getSizeDims();

		
		// for getting Theta,b, and lth layer al, zl (after activation function applied)
		std::shared_ptr<float> getTheta();
		
		std::shared_ptr<float> getb();

		std::shared_ptr<float> getalm1();

		std::shared_ptr<float> getal();


		/**
		 *  @fn rightMul
		 *  @class Axon_sh
		 * 	@brief right multiplication
		 * */
		void rightMul(); 
		

		// destructor
		~Axon_sh();			
};


/**
 *  @class Axon_u
 *  @brief Axon, using unique pointers, contains weights matrix (Theta), and bias b between 2 layers
 */
class Axon_u
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
		// Theta,b - "weights" and bias
		std::unique_ptr<float[], deleterRR_struct> Theta;
		std::unique_ptr<float[], deleterRR_struct> b;

		// "layers" a^{l-1}, a^l, i.e. alm1, al
		std::unique_ptr<float[], deleterRR_struct> alm1;
		std::unique_ptr<float[], deleterRR_struct> al;

	public:
		// Constructor
		Axon_u(const int, const int);

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
		
		// initialize layer l
		void init_al(const int);

		// for getting size dimensions
		std::vector<int> getSizeDims();
		
		// for getting Theta,b, and lth layer al, zl (after activation function applied)
		std::unique_ptr<float[],deleterRR_struct> getTheta();
		
		std::unique_ptr<float[],deleterRR_struct> getb();

		std::unique_ptr<float[],deleterRR_struct> getalm1();

		std::unique_ptr<float[],deleterRR_struct> getal();


		/**
		 *  @fn rightMul
		 *  @class Axon_sh
		 * 	@brief right multiplication
		 * */
		void rightMul(); 
		

		// destructor
		~Axon_u();			


};



#endif 
