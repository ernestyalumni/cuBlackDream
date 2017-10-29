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
 * nvcc -std=c++14 -arch='sm_52' -dc ../Axon/Axon.cu ../Axon/activationf.cu Feedfwd.cu
 * */
#include "Feedfwd_deep.h"  

/* =============== CUDA functions =============== */
int get_max_device_array_size1d(const int idx_device) {
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount( &count )

	if (count>0) {
		cudaGetDeviceProperties( &prop, idx_device );
		int MAX_SIZE = prop.maxGridSize[0];		
		return MAX_SIZE;
	} else {
		return EXIT_FAILURE;
}

/* =============== CUDA kernel functions =============== */
/** @fn setconstval_kernel
 * 	@brief set a float array of length Lx all to values of const_val 
 * 	@details cudaMemset only sets an array to 0 value; we want value of 1
 * */
__global__ void setconstval_kernel(const int Lx, const float const_val, float* A) {
	int kx = threadIdx.x + blockDim.x * blockIdx.x; 
	if (kx >= Lx) { 
		return ; 
	} 
	for (int tid =kx; tid < Lx; tid += gridDim.x*blockDim.x) {
		A[tid] = const_val ; 	
	}
}

__global__ void costJ_xent_kernel(const int Lx, const float* y, const float* yhat, float* s) {
	int kx = threadIdx.x + blockDim.x*blockIdx.x;
	if (kx >= Lx) { return; } // more than enough threads were launched to calculate the Lx elements
	
	/* this for loop ensure that, if in case SIZE > gridDimx.x*blockDim.x (or even if  
	 * 	SIZE >> gridDimx.x*blockDim.x so we have more values to compute than threads on a GPU!)  
	 * that everything gets computed) */
	for (int tid=kx; tid < Lx; tid += gridDim.x*blockDim.x ) { 
		float y_val = y[tid];
		float yhat_val = yhat[tid]; 
		float s_val = - y_val * logf( yhat_val) - (1.f - y_val) * logf( 1.f - yhat_val);
		s[tid] = s_val;
	}
}

/**
 * @fn Deltaxent_kernel, __global__ void Deltaxent_kernel
 * @brief compute Delta for the so-called cross-entropy loss function
 * @details Compute
 * ( \widehat{y}^k_{(i)} - y_{(i)}^k )/ (\widehat{y}^k_{(i)} (1 - \widehat{y}_{(i)}^k ) ) 
*/
__global__ void Deltaxent_kernel(const int Lx, const float* y, const float* yhat, float* Delta) {
	int kx = threadIdx.x + blockDim.x*blockIdx.x;
	if (kx >= Lx) { return; } // more than enough threads were launched to calculate the Lx elements
	
	/* this for loop ensure that, if in case SIZE > gridDimx.x*blockDim.x (or even if  
	 * 	SIZE >> gridDimx.x*blockDim.x so we have more values to compute than threads on a GPU!)  
	 * that everything gets computed) */
	for (int tid=kx; tid < Lx; tid += gridDim.x*blockDim.x ) { 
		float y_val = y[tid];
		float yhat_val = yhat[tid]; 
		float Delta_ik = (yhat_val - y_val)/(yhat_val*(1.0f - yhat_val)); 
		Delta[tid] = Delta_ik;
	}
}

/**
 * 	@fn HadamardMultiply
 * 	@brief element-wise multiply  
 * */
__global__ void HadamardMultiply_kernel(const int SIZE, const float* A, float* B) {
	int kx = threadIdx.x + blockDim.x*blockIdx.x;
	if (kx >= SIZE) { return; } // more than enough threads were launched to calculate the Lx elements
	
	/* this for loop ensure that, if in case SIZE > gridDimx.x*blockDim.x (or even if  
	 * 	SIZE >> gridDimx.x*blockDim.x so we have more values to compute than threads on a GPU!)  
	 * that everything gets computed) */
	for (int tid=kx; tid < SIZE; tid += gridDim.x*blockDim.x ) { 
		float A_val = A[tid];
		float B_val = B[tid];
		float C_val = A_val * B_val; 
		B[tid] = C_val;
	}
}

/* ==================== Deep Neural Network (DNN) class ==================== */

/**	@class DNN
 * 	@brief Deep Neural Network (DNN; i.e. Artificial Neural Network (ANN), 
 * 		i.e. so-called "Fully Connected layers")
 * */

 // Constructors
/** 
 * 	@fn DNN::DNN
 * 	@brief Constructor for DNN class 
 * 	@param sizeDimsvec - std::vector<int> &
 * 	@param actfs_intvec - std::vector<int> &
 * 	@param idx_device - const int 
 * */
DNN::DNN(std::vector<int> & sizeDimsvec, std::vector<int> & actfs_intvec, 
			const int idx_device) : 
	sizeDimsvec(sizeDimsvec), actfs_intvec(actfs_intvec) 
{
	const int Lp1 = sizeDimsvec.size(); // L=total number of Axons and so L+1 is total number of layers

	// add Axons sequentially, l=1,2,...L
	for (int l=1; l<Lp1; l++)  // l= lth Axon
	{
		int s_lm1 = sizeDimsvec[l-1];
		int s_l = sizeDimsvec[l];
		int idx_actf = actfs_intvec[l-1];
		Axons.push_back( Axon_act(s_lm1,s_l,idx_actf) );
	}
	
	// get maximum grid dimension on the device, numbered idx_device (usually 0th device GPU)
	MAX_SIZE_1DARR = get_max_device_array_size1d(idx_device);
}

// member functions

// for loading (Theta,B) values from host
void DNN::load_from_hThetaBs(std::vector<std::vector<float>> & hThetaBs) {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers
	assert (hThetaBs.size()/2 == (Lp1-1) );	// sanity check with input is correct
	for (int l=1; l<Lp1; l++)  // l= lth Axon, l=1,2...L
	{	
		int idx_axon = l-1; // idx_axon=0,1,...L-1
		int idx_Theta = idx_axon*2;
		int idx_b 	  = idx_axon*2 + 1;
		
		Axons[idx_axon].load_from_hvec( hThetaBs[idx_Theta], hThetaBs[idx_b] );
	}
}

// for loading output data y 
/**
 * 	@fn load_y_from_hvec 
 * 	@brief load from host, y output data, as a std::vector<float>, column-major ordered
 * */		
void DNN::load_y_from_hvec(std::vector<float>& h_yvec) {
	const int SIZE_Y= h_yvec.size(); 
	
	std::unique_ptr<float[], deleterRR_struct> d_y(new float[SIZE_Y], deleterRR_struct());
	cudaMallocManaged((void **) &d_y, SIZE_Y*sizeof(float));
	y = std::move(d_y);

	cudaMemcpy(y.get(), h_yvec.data(), SIZE_Y*sizeof(float),cudaMemcpyHostToDevice);	
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
void DNN::load_X_from_hvec(std::vector<float>& h_Xvec, const int m) 
{
	const int SIZE_X = h_Xvec.size();
	const int d = sizeDimsvec[0];
	
	// first Axon
	Axons[0].load_from_hXvec( h_Xvec, m);
	Axons[0].init_zlal(m);

	const int Lp1 = sizeDimsvec.size(); // L=total number of Axons and so L+1 is total number of layers
	
	for (int l=2;l<Lp1; l++) {
		int idx_axon = l-1; // l=2,3...L, idx_axon=1,2,...L-1

		// must do move for 1 command immediately below, because otherwise, error: initial value of reference 
		// to non-const must be an lvalue
		auto tempshptr = std::move( Axons[idx_axon-1].getal() ); // temporary shared pointer, move ownership to it temporarily
		Axons[idx_axon].load_alm1_from_ptr( tempshptr );
		Axons[idx_axon].init_zlal(m);
		Axons[idx_axon-1].move2al_from_ptr( tempshptr); // move ownership back to al from temporary shared ptr

	}
	this->m=m; // store the number of training examples
}


/* =============== "getting" functions =============== */

// for getting Theta,b, and lth layer of lth Axon al, zl (after activation function applied)

std::unique_ptr<float[], deleterRR_struct> DNN::getTheta(const int l) {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers
	assert (l < Lp1);	// sanity check that l=1,2,...L
	int idx_axon = l-1; // ind_axon=0,1,...L-1, 0-based counting
	auto ptr = std::move( Axons[idx_axon].getTheta() );

	return ptr;
}

std::unique_ptr<float[],deleterRR_struct> DNN::getb(const int l) {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers
	assert (l < Lp1);	// sanity check that l=1,2,...L
	int idx_axon = l-1; // ind_axon=0,1,...L-1, 0-based counting
	auto ptr = std::move( Axons[idx_axon].getb() );

	return ptr;
}



std::shared_ptr<float> DNN::getal(const int l) {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers
	assert (l < Lp1);	// sanity check that l=1,2,...L
	int idx_axon = l-1; // ind_axon=0,1,...L-1, 0-based counting
	auto ptr = std::move( Axons[idx_axon].getal() );

	return ptr;
}

std::unique_ptr<float[],deleterRR_struct> DNN::gety() {
	auto ptr = std::move(y);
	return ptr;
}

/* ========== Feedforward ========== */
/**
 *  @fn feedfwd
 * 	@brief Feedforward
 * 	@param Mx, int Mx=128, default to 128 threads in a single thread block
 * 		when adding the bias to the output layer of an axon, choose the number of threads in a single 
 * */
void DNN::feedfwd(int M_x) {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers

	Nx_MAX = (MAX_SIZE_1DARR + M_x - 1)/M_x;  

	for (int l=1; l < Lp1;l++) {
		int idx_axon = l-1; // l=1,2,...L axons, idx_axon = 0,1,...L-1 (0-based counting for C/C++/Python

		s_l = sizeDimsvec[l];		
		int SIZE_A_L = s_l*m; 
		int N_x = (SIZE_A_L + M_x - 1)/M_x;
		N_x = max(N_x,0);
		N_x = min(N_x,Nx_MAX);

		Axons[idx_axon].rightMul();	// a^{l-1} \Theta = (a^{l-1}_i)^{j_{l-1}} \Theta_{j_{l-1}}^{j_l} =: z^l 
		Axons[idx_axon].addb(M_x,N_x);	// z^l +b = (z^l_i)^{j_l} + (b^{(l)})^{j_l} =: z^l

		/**
		 * @note EY : 20171023 remember to fix the calculation of (thread) blocks on a grid to allow for 
		 * arrays of size >> total number of threads allowed on the grid
		 * */
		Axons[idx_axon].actf(M_x,N_x); 
		Axons[idx_axon].do_Dpsi(M_x,N_x); 

	}
}






