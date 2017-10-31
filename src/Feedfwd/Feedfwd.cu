/**
 * @file   : Feedfwd.cu
 * @brief  : Feedforward content/source file in CUDA C++14, 
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
#include "Feedfwd.h"

/* =============== CUDA functions =============== */

/* =============== CUDA kernel functions =============== */

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

/* ==================== Linear Regression class ==================== */

/**	@class LinReg
 * 	@brief Linear Regression  
 * */

// Constructors
LinReg::LinReg(std::vector<int> & sizeDimsvec, 
					const int idx_device) : 
	sizeDimsvec(sizeDimsvec) 
{
	const int Lp1 = sizeDimsvec.size(); // L=total number of Axons and so L+1 is total number of layers
	for (int l=1; l<Lp1; l++)  // l= lth Axon
	{
		int s_lm1 = sizeDimsvec[l-1];
		int s_l = sizeDimsvec[l];
		Axons.push_back( Axon(s_lm1,s_l,idx_device) );
	}	

	// get maximum grid dimension on the device, numbered idx_device (usually 0th device GPU)
	MAX_SIZE_1DARR = get_max_device_array_size1d(idx_device);
	
}
		
// member functions

// for loading (Theta,B) values from host
void LinReg::load_from_hThetaBs(std::vector<std::vector<float>> & hThetaBs) {
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
void LinReg::load_y_from_hvec(std::vector<float>& h_yvec) {
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

		auto tempshptr = Axons[idx_axon-1].getal() ; // temporary shared pointer, don't move, but share ownership to it temporarily
		

		// must do move for 1 command immediately below, because otherwise, error: initial value of reference 
		// to non-const must be an lvalue
		Axons[idx_axon].load_alm1_from_ptr( tempshptr );
		Axons[idx_axon].init_al(m);
		tempshptr.reset();
	}

	this->m=m; // store the number of training examples
}


/* =============== "getting" functions =============== */

// for getting Theta,b, and lth layer of lth Axon al, zl (after activation function applied)

std::unique_ptr<float[], deleterRR_struct> LinReg::getTheta(const int l) {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers
	assert (l < Lp1);	// sanity check that l=1,2,...L
	int idx_axon = l-1; // ind_axon=0,1,...L-1, 0-based counting
	auto ptr = std::move( Axons[idx_axon].getTheta() );

	return ptr;
}

std::unique_ptr<float[],deleterRR_struct> LinReg::getb(const int l) {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers
	assert (l < Lp1);	// sanity check that l=1,2,...L
	int idx_axon = l-1; // ind_axon=0,1,...L-1, 0-based counting
	auto ptr = std::move( Axons[idx_axon].getb() );

	return ptr;
}

std::shared_ptr<float> LinReg::getalm1(const int l) {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers
	assert (l < Lp1);	// sanity check that l=1,2,...L
	int idx_axon = l-1; // ind_axon=0,1,...L-1, 0-based counting
	auto ptr = Axons[idx_axon].getalm1() ;

	return ptr;
}


std::shared_ptr<float> LinReg::getal(const int l) {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers
	assert (l < Lp1);	// sanity check that l=1,2,...L
	int idx_axon = l-1; // ind_axon=0,1,...L-1, 0-based counting
	auto ptr = Axons[idx_axon].getal() ;

	return ptr;
}

std::unique_ptr<float[],deleterRR_struct> LinReg::gety() {
	auto ptr = std::move(y);
	return ptr;
}

/* ========== Feedforward ========== */

void LinReg::feedfwd(int M_x) {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers

	for (int l=1; l < Lp1;l++) {
		int idx_axon = l-1; // l=1,2,...L axons, idx_axon = 0,1,...L-1 (0-based counting for C/C++/Python

		/* ===== grid, thread block size dimensions ===== */
		// each Axon will take care of determining number of (thread) blocks on a grid, N_x
	
		Axons[idx_axon].rightMul();	// a^{l-1} \Theta = (a^{l-1}_i)^{j_{l-1}} \Theta_{j_{l-1}}^{j_l} =: z^l 
		Axons[idx_axon].addb(M_x);	// z^l +b = (z^l_i)^{j_l} + (b^{(l)})^{j_l} =: z^l
	}
}

float LinReg::compute_costJ_L2norm() {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers
	const int K = sizeDimsvec[Lp1-1]; // size dim. of the a^L output layer for axon L, i.e. \widehat{h}, the prediction
	const int SIZE_Y= K * m; 

	auto yhat = Axons[Lp1-2].getal(); // L+1 - 2 = L-1 which is the last axon, when counting from 0, 0,1,...L-1
	

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
		m, &bet, y.get(), m, 
		res.get(), m );
					
	float costJ = 0.f;
	// do the L2 Euclidean norm element-wise
	cublasSnrm2(*handle_u.get(), SIZE_Y, res.get(), 1, &costJ);
	costJ = 0.5f*costJ*costJ/((float) m) ;
	
	/** 
	 * @ref https://stackoverflow.com/questions/21589595/does-using-reset-on-a-stdshared-ptr-delete-all-instances
	 * */
	yhat.reset();
	return costJ;
	
}

/**	@fn grad_desc_step
 *	@param Mx - number of threads in a (single) thread block in x-direction
 * 				this is needed for setconstval_kernel, to create a vector of 1's as 
 * 				a numerical trick for the usual (mathematical) Kronecker delta function	 
 * */
void LinReg::grad_desc_step(  const float alpha_rate, int Mx)
{
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers
	const int K = sizeDimsvec[Lp1-1]; // size dim. of the a^L output layer for axon L, i.e. \widehat{h}, the prediction
	const int SIZE_Y= K * m; 

	const int d = sizeDimsvec[Lp1-2];

	std::unique_ptr<cublasHandle_t,del_cublasHandle_struct> handle_u(
		new cublasHandle_t);
	cublasCreate(handle_u.get());	

	// in this scope, make res to store results from taking the difference
	std::unique_ptr<float[], deleterRR_struct> res(new float[SIZE_Y], deleterRR_struct());
	cudaMallocManaged((void **) &res, SIZE_Y*sizeof(float));
	
	auto yhat = Axons[Lp1-2].getal(); // L+1 - 2 = L-1 which is the last axon, when counting from 0, 0,1,...L-1
	auto ThetaL = std::move( Axons[Lp1-2].getTheta() );
	auto bL = std::move( Axons[Lp1-2].getb() );

	
	// residual \equiv Delta = (yhat - y) \in \text{Mat}_{\mathbb{R}}(m,K)
	float a1 = 1.0f;
	float bet = -1.0f; 
	cublasSgeam(*handle_u.get(), 
		CUBLAS_OP_N, CUBLAS_OP_N, m, K, &a1, 
		yhat.get(), m, &bet, y.get(), m, 
		res.get(), m );

	auto a0 = Axons[Lp1-2].getalm1();
	
	a1 = 1.0f/ ((float) m);
	bet = 0.f;
	// \sum_{i=1}^m (a_i^{(0)})^j \Delta_i^p = \frac{ \partial J }{ \partial \Theta_j^p }
	

	const int SIZE_dTHETA = d*K;
	std::unique_ptr<float[], deleterRR_struct> dTheta(new float[SIZE_dTHETA], deleterRR_struct());
	cudaMallocManaged((void **) &dTheta, SIZE_dTHETA*sizeof(float));
	
	// dTheta = (1./m)*dTheta; dTheta \in \text{Mat}_{\mathbb{R}}(d,K)
	cublasSgemm(*handle_u.get(),
		CUBLAS_OP_T, CUBLAS_OP_N, d, K, m, &a1, a0.get(), m, res.get(), m , 
			&bet, dTheta.get(), d);
	
	// dB = (1./m)*dB ; dB \in \mathbb{R}^K
	const int SIZE_dB = K;
	std::unique_ptr<float[], deleterRR_struct> dB(new float[SIZE_dB], deleterRR_struct());
	cudaMallocManaged((void **) &dB, SIZE_dB*sizeof(float));

	// create 1s array, array of 1s
	const int SIZE_ONES = m;
	std::unique_ptr<float[], deleterRR_struct> ones(new float[SIZE_ONES], deleterRR_struct());
	cudaMallocManaged((void **) &ones, SIZE_ONES*sizeof(float));

	/* ===== grid, thread block size dimensions ===== */
	// M_x = number of threads in a (single) block in x-direction
	int Nx = (SIZE_ONES + Mx - 1)/Mx; 
	if ( MAX_SIZE_1DARR < SIZE_ONES ) {
		Nx = (MAX_SIZE_1DARR + Mx - 1) / Mx ; }

	setconstval_kernel<<<Nx,Mx>>>(m,1.0f, ones.get() );
 
	// this is a clever way to do summation
	cublasSgemm(*handle_u.get(), CUBLAS_OP_N, CUBLAS_OP_N, 1,K,m, 
		&a1, ones.get(), 1, res.get(), m, 
		&bet, dB.get(), 1); 

		
	bet = 1.0f; 
	a1 = -1.0f * alpha_rate; 

	// actual gradient descent iteration step
	cublasSaxpy( *handle_u.get(), SIZE_dTHETA, &a1, dTheta.get(), 1, ThetaL.get(), 1);
	cublasSaxpy( *handle_u.get(), SIZE_dB, &a1, dB.get(), 1, bL.get(), 1);



	// return ownership of yhat,a0 back to the Feed-forward "network"
	yhat.reset(); 
	a0.reset(); 
	Axons[Lp1-2].move2Theta_from_ptr(ThetaL);
	Axons[Lp1-2].move2b_from_ptr(bL);

}

/**	@fn grad_desc
 *	@param Mx - number of threads in a (single) thread block in x-direction
 * 				this is needed in the following:
 * 				in feedfwd, for addb, because we're doing "row-wise" addition of a row vector
 * 					across a matrix, 
 * 				and 
 * 				in grad_desc_step, for setconstval_kernel, to create a vector of 1's as 
 * 				a numerical trick for the usual (mathematical) Kronecker delta function	 
 * */
void LinReg::grad_desc(  const int iterations, const float alpha_rate, int Mx)
{
	for (int iter=0; iter < iterations; iter++) 
	{
		feedfwd(Mx);
		grad_desc_step( alpha_rate, Mx);
		
	}
}

// destructor
LinReg::~LinReg() {}

/* ==================== END of Linear Regression class ==================== */

/* ==================== Logistic Regression class ==================== */

/**	@class LogisticReg
 * 	@brief Logistic Regression  
 * */
 
 // Constructors
LogisticReg::LogisticReg(std::vector<int> & sizeDimsvec, std::vector<int> & actfs_intvec, 
	const int idx_device) : 
	sizeDimsvec(sizeDimsvec), actfs_intvec(actfs_intvec)
{
	const int Lp1 = sizeDimsvec.size(); // L=total number of Axons and so L+1 is total number of layers
	for (int l=1; l<Lp1; l++)  // l= lth Axon
	{
		int s_lm1 = sizeDimsvec[l-1];
		int s_l = sizeDimsvec[l];
		int idx_actf = actfs_intvec[l-1];
		Axons.push_back( Axon_act(s_lm1,s_l,idx_actf, idx_device) );
	}	
	
	// get maximum grid dimension on the device, numbered idx_device (usually 0th device GPU)
	MAX_SIZE_1DARR = get_max_device_array_size1d(idx_device);
	
}
		
// member functions

// for loading (Theta,B) values from host
void LogisticReg::load_from_hThetaBs(std::vector<std::vector<float>> & hThetaBs) {
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
void LogisticReg::load_y_from_hvec(std::vector<float>& h_yvec) {
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
void LogisticReg::load_X_from_hvec(std::vector<float>& h_Xvec, const int m) 
{
	const int SIZE_X = h_Xvec.size();
	const int d = sizeDimsvec[0];
	assert( SIZE_X == m*d); // check the total size dimensions is correct for "input layer" 
	
	// first Axon
	Axons[0].load_from_hXvec( h_Xvec, m);
	Axons[0].init_zlal(m);

	const int Lp1 = sizeDimsvec.size(); // L=total number of Axons and so L+1 is total number of layers
	
	for (int l=2;l<Lp1; l++) {
		int idx_axon = l-1; // l=2,3...L, idx_axon=1,2,...L-1

		// if we didn't assign the shared_ptr and so "share" it, 
		// must do move for 1 command immediately below, because otherwise, error: initial value of reference 
		// to non-const must be an lvalue
		auto tempshptr = Axons[idx_axon-1].getal(); // temporary shared pointer, share, NOT move, ownership to it temporarily
		Axons[idx_axon].load_alm1_from_ptr( tempshptr );
		Axons[idx_axon].init_zlal(m);
//		Axons[idx_axon-1].move2al_from_ptr( tempshptr); // move ownership back to al from temporary shared ptr
		tempshptr.reset();
	}
	this->m=m; // store the number of training examples
}


/* =============== "getting" functions =============== */

// for getting Theta,b, and lth layer of lth Axon al, zl (after activation function applied)

std::unique_ptr<float[], deleterRR_struct> LogisticReg::getTheta(const int l) {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers
	assert (l < Lp1);	// sanity check that l=1,2,...L
	int idx_axon = l-1; // ind_axon=0,1,...L-1, 0-based counting
	auto ptr = std::move( Axons[idx_axon].getTheta() );

	return ptr;
}

std::unique_ptr<float[],deleterRR_struct> LogisticReg::getb(const int l) {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers
	assert (l < Lp1);	// sanity check that l=1,2,...L
	int idx_axon = l-1; // ind_axon=0,1,...L-1, 0-based counting
	auto ptr = std::move( Axons[idx_axon].getb() );

	return ptr;
}

std::shared_ptr<float> LogisticReg::getalm1(const int l) {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers
	assert (l < Lp1);	// sanity check that l=1,2,...L
	int idx_axon = l-1; // ind_axon=0,1,...L-1, 0-based counting
	auto ptr = Axons[idx_axon].getalm1();

	return ptr;
}


std::shared_ptr<float> LogisticReg::getal(const int l) {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers
	assert (l < Lp1);	// sanity check that l=1,2,...L
	int idx_axon = l-1; // ind_axon=0,1,...L-1, 0-based counting
	auto ptr = Axons[idx_axon].getal();

	return ptr;
}

std::unique_ptr<float[],deleterRR_struct> LogisticReg::gety() {
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

void LogisticReg::feedfwd(int M_x) {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers

	for (int l=1; l < Lp1;l++) {
		int idx_axon = l-1; // l=1,2,...L axons, idx_axon = 0,1,...L-1 (0-based counting for C/C++/Python
		Axons[idx_axon].rightMul();	// a^{l-1} \Theta = (a^{l-1}_i)^{j_{l-1}} \Theta_{j_{l-1}}^{j_l} =: z^l 
		Axons[idx_axon].addb(M_x);	// z^l +b = (z^l_i)^{j_l} + (b^{(l)})^{j_l} =: z^l

		/**
		 * @note EY : 20171023 remember to fix the calculation of (thread) blocks on a grid to allow for 
		 * arrays of size >> total number of threads allowed on the grid
		 * */
		Axons[idx_axon].actf(M_x); 
		Axons[idx_axon].do_Dpsi(M_x);
	}
}

/* ========== Cost functional J ========== */

float LogisticReg::compute_costJ_xent(const int Mx) {
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers
	const int K = sizeDimsvec[Lp1-1]; // size dim. of the a^L output layer for axon L, i.e. \widehat{h}, the prediction
	const int SIZE_Y= K * m; 

	auto yhat = Axons[Lp1-2].getal(); // L+1 - 2 = L-1 which is the last axon, when counting from 0, 0,1,...L-1


	// in this scope, make res to store results from taking the so-called cross-entropy function, element-wise
	std::unique_ptr<float[], deleterRR_struct> entropys(new float[SIZE_Y], deleterRR_struct());
	cudaMallocManaged((void **) &entropys, SIZE_Y*sizeof(float));

	/* ===== grid, thread block size dimensions ===== */
	// M_x = number of threads in a (single) block in x-direction
	/** 
	 * @note EY : 20171023 I will need to change this calculation of 
	 * N_x = number of (thread) blocks on a grid in x-direction
	 * so to allow for SIZE_Y >> max. allowed gridDimx.x*blockDim.x i.e. maximum allowed threads to launch on a grid
	 * */
	const int Nx = (SIZE_Y + Mx -1)/Mx;
	costJ_xent_kernel<<<Nx,Mx>>>( SIZE_Y, y.get(), yhat.get(), entropys.get() );

	// ========== now do the summation ========== 

	// create 1s array, array of 1s
	const int SIZE_ONES = SIZE_Y;
	std::unique_ptr<float[], deleterRR_struct> ones(new float[SIZE_ONES], deleterRR_struct());
	cudaMallocManaged((void **) &ones, SIZE_ONES*sizeof(float));

	/* ===== grid, thread block size dimensions ===== */
		// M_x = number of threads in a (single) block in x-direction
	setconstval_kernel<<<Nx,Mx>>>(m,1.0f, ones.get() );
 
	// this is a clever way to do summation
	std::unique_ptr<cublasHandle_t,del_cublasHandle_struct> handle_u(
		new cublasHandle_t);
	cublasCreate(handle_u.get());	

	float costJ = 0.f;
	cublasSdot( *handle_u.get(), SIZE_Y, entropys.get(), 1, ones.get(), 1, &costJ);
	costJ = costJ/((float) m);

//	Axons[Lp1-2].move2al_from_ptr(yhat);
	yhat.reset();
	return costJ;
	
} 

/**	@fn grad_desc_step
 *	@param Mx - number of threads in a (single) thread block in x-direction
 * 				this is needed for setconstval_kernel, to create a vector of 1's as 
 * 				a numerical trick for the usual (mathematical) Kronecker delta function	 
 * */
void LogisticReg::grad_desc_step(  const float alpha_rate, int M_x)
{
	const int Lp1 = sizeDimsvec.size(); // L = total number of Axons and so L+1 is total number of layers
	const int K = sizeDimsvec[Lp1-1]; // size dim. of the a^L output layer for axon L, i.e. \widehat{h}, the prediction
	const int SIZE_Y=  m * K ; 

	const int d = sizeDimsvec[Lp1-2];

	std::unique_ptr<cublasHandle_t,del_cublasHandle_struct> handle_u(
		new cublasHandle_t);
	cublasCreate(handle_u.get());	

	/* ===== grid, thread block size dimensions ===== */
	// M_x = number of threads in a (single) block in x-direction
	int Nx = (SIZE_Y + M_x - 1)/M_x; 
	if ( MAX_SIZE_1DARR < SIZE_Y ) {
		Nx = (MAX_SIZE_1DARR + M_x - 1) / M_x ; }

	// in this scope, make Delta to store results from take the partial derivative of the cross entropy function
	std::unique_ptr<float[], deleterRR_struct> Delta(new float[SIZE_Y], deleterRR_struct());
	cudaMallocManaged((void **) &Delta, SIZE_Y*sizeof(float));
	
	auto yhat = Axons[Lp1-2].getal(); // L+1 - 2 = L-1 which is the last axon, when counting from 0, 0,1,...L-1
	auto ThetaL = std::move( Axons[Lp1-2].getTheta() );
	auto bL = std::move( Axons[Lp1-2].getb() );

	Deltaxent_kernel<<<Nx,M_x>>>(SIZE_Y, y.get(), yhat.get(), Delta.get()) ;

	// then do the Hadamard product with dPsi^(L)/dz^(L)
	auto dPsiLdzL = Axons[Lp1-2].getDpsil(); // L+1 - 2 = L-1 which is the last axon, when counting from 0, 0,1,...L-1
	
	HadamardMultiply_kernel<<<Nx,M_x>>>(SIZE_Y, dPsiLdzL.get(), Delta.get());


	auto a0 = Axons[Lp1-2].getalm1();
	
	float a1 = 1.0f/ ((float) m);
	float bet = 0.f;
	// \sum_{i=1}^m (a_i^{(0)})^j \Delta_i^p = \frac{ \partial J }{ \partial \Theta_j^p }

	const int SIZE_dTHETA = d*K;
	std::unique_ptr<float[], deleterRR_struct> dTheta(new float[SIZE_dTHETA], deleterRR_struct());
	cudaMallocManaged((void **) &dTheta, SIZE_dTHETA*sizeof(float));
	
	// dTheta = (1./m)*dTheta; dTheta \in \text{Mat}_{\mathbb{R}}(d,K)
	cublasSgemm(*handle_u.get(),
		CUBLAS_OP_T, CUBLAS_OP_N, d, K, m, &a1, a0.get(), m, Delta.get(), m , 
			&bet, dTheta.get(), d);

	// dB = (1./m)*dB ; dB \in \mathbb{R}^K
	const int SIZE_dB = K;
	std::unique_ptr<float[], deleterRR_struct> dB(new float[SIZE_dB], deleterRR_struct());
	cudaMallocManaged((void **) &dB, SIZE_dB*sizeof(float));

	// create 1s array, array of 1s
	const int SIZE_ONES = m;
	std::unique_ptr<float[], deleterRR_struct> ones(new float[SIZE_ONES], deleterRR_struct());
	cudaMallocManaged((void **) &ones, SIZE_ONES*sizeof(float));

	/* ===== grid, thread block size dimensions ===== */
	// M_x = number of threads in a (single) block in x-direction
	Nx = (SIZE_ONES + M_x - 1)/M_x; 
	if ( MAX_SIZE_1DARR < SIZE_ONES ) {
		Nx = (MAX_SIZE_1DARR + M_x - 1) / M_x ; }

	setconstval_kernel<<<Nx,M_x>>>(m,1.0f, ones.get() );
 
	// this is a clever way to do summation
	cublasSgemm(*handle_u.get(), CUBLAS_OP_N, CUBLAS_OP_N, 1,K,m, 
		&a1, ones.get(), 1, Delta.get(), m, 
		&bet, dB.get(), 1); 


	bet = 1.0f; 
	a1 = -1.0f * alpha_rate; 

	// actual gradient descent iteration step
	cublasSaxpy( *handle_u.get(), SIZE_dTHETA, &a1, dTheta.get(), 1, ThetaL.get(), 1);
	cublasSaxpy( *handle_u.get(), SIZE_dB, &a1, dB.get(), 1, bL.get(), 1);


	// return ownership of yhat,a0 back to the Feed-forward "network"
	yhat.reset(); 
	a0.reset(); 
	Axons[Lp1-2].move2Theta_from_ptr(ThetaL);
	Axons[Lp1-2].move2b_from_ptr(bL);
	Axons[Lp1-2].move2Dpsil_from_ptr(dPsiLdzL);
	
}


/**	@fn grad_desc
 *	@param Mx - number of threads in a (single) thread block in x-direction
 * 				this is needed in the following:
 * 				in feedfwd, for addb, because we're doing "row-wise" addition of a row vector
 * 					across a matrix, 
 * 				and 
 * 				in grad_desc_step, for setconstval_kernel, to create a vector of 1's as 
 * 				a numerical trick for the usual (mathematical) Kronecker delta function	 
 * */
void LogisticReg::grad_desc(  const int iterations, const float alpha_rate, int M_x)
{
	for (int iter=0; iter < iterations; iter++) 
	{
		feedfwd(M_x);
		grad_desc_step( alpha_rate, M_x);
		
	}
}



// destructor
LogisticReg::~LogisticReg() {}
