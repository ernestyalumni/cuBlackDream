/**
 * @file   : Feedfwd.h
 * @brief  : Feedforward header file in CUDA C++14, 
 * @details : the Feedforward class and related functions 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171017
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
 * nvcc -std=c++14 -lcublas ../Axon/Axon.o -dc Feedfwd.cu -o Feedfwd.o
 * 
 * */
#ifndef __FEEDFWD_H__
#define __FEEDFWD_H__ 

#include <memory> 			// std::shared_ptr, std::unique_ptr
#include <cassert> 			// assert

#include "../Axon/Axon.h"	// deleterRR_struct



/** @fn : computeJ_L2norm
 * 	@brief Compute cost functional J, using the L2-norm form
 * */ 
/*float computeJ_L2norm(const int,
	std::shared_ptr<float> &,std::unique_ptr<float[],deleterRR_struct> & );
*/
/**	@class LinReg
 * 	@brief Linear Regression  
 * */
class LinReg
{
	private:
		std::vector<Axon> Axons;
		std::unique_ptr<float[], deleterRR_struct> y;
		std::vector<int> sizeDimsvec;

		int m; // number of examples in the dataset

	public:
		// Constructor
		LinReg(std::vector<int> &);
		
//		LinReg(std::vector<Axon> &);
		
		// member functions
//		void addAxon(Axon &);
		
	

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
		void load_X_from_hvec(std::vector<float>&, const int);


		/* ========== Feedforward ========== */
		/**
		 *  @fn feedfwd
		 * 	@brief Feedforward
		 * 	@param Mx, int Mx=128, default to 128 threads in a single thread block
		 * 		when adding the bias to the output layer of an axon, choose the number of threads in a single 
		 * */
		void feedfwd(int Mx=128);

		
		/* ========== Cost functional J ========== */
		float compute_costJ_L2norm();
		
		// destructor
		~LinReg();
		
};

#endif // __FEEDFWD_H__
