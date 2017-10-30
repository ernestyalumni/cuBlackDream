/**
 * @file   : Axon.cu
 * @brief  : Axon content/source file in CUDA C++14, 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171007  
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
#include "Axon.h"
#include "activationf.h"

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

	for (int tid=kx; tid < Lx; tid += gridDim.x * blockDim.x ) {
		A[tid] = const_val ; 	
	}
}


/* ==================== Axon classes ==================== */

/* =============== Axon class; no activation =============== */


// constructor 
Axon::Axon(const int s_lm1,const int s_l) : s_lm1(s_lm1), s_l(s_l)  {
	const int SIZE_THETA = s_l*s_lm1;

	std::unique_ptr<float[], deleterRR_struct> d_Theta(new float[SIZE_THETA]);
	cudaMallocManaged((void **) &d_Theta,SIZE_THETA*sizeof(float));
	Theta = std::move(d_Theta);

	std::unique_ptr<float[], deleterRR_struct> d_b(new float[s_l]);
	cudaMallocManaged((void **) &d_b,s_l*sizeof(float));
	b = std::move(d_b);
}

// Move Constructor
/**
 *  @fn Axon(const Axon& old_axon)
 *  @brief copy constructor for Axon class
 * 	@ref http://www.geeksforgeeks.org/copy-constructor-in-cpp/
 * https://stackoverflow.com/questions/16030081/copy-constructor-for-a-class-with-unique-ptr
 * https://en.wikipedia.org/wiki/C%2B%2B11#Rvalue_references_and_move_constructors
 * */
Axon::Axon(Axon&& old_axon) : Theta(std::move(old_axon.Theta)), b(std::move(old_axon.b))
{
	s_lm1 = old_axon.s_lm1;
	s_l = old_axon.s_l;
	m = old_axon.m;
	
	l = old_axon.l; // lth layer
	
	alm1 = std::move( old_axon.alm1 );
	al = std::move( old_axon.al );	
}

// operator overload assignment = 
Axon & Axon::operator=(Axon && old_axon) {
	s_lm1 = old_axon.s_lm1;
	s_l = old_axon.s_l;
	m = old_axon.m;
	
	l = old_axon.l; // lth layer

	// shared_ptrs moved
	alm1 = std::move( old_axon.alm1 );
	al = std::move( old_axon.al );	

	// unique_ptrs moved
	Theta = std::move(old_axon.Theta);
	b = std::move( old_axon.b );

	return *this;
}

// member functions
void Axon::load_from_hvec(std::vector<float>& h_Theta,std::vector<float>& h_b) {
	const int SIZE_THETA = s_l*s_lm1;

	cudaMemcpy(Theta.get(), h_Theta.data(), SIZE_THETA*sizeof(float),cudaMemcpyHostToDevice);	
	cudaMemcpy(b.get(), h_b.data(), s_l*sizeof(float),cudaMemcpyHostToDevice);	
}	

/**
 * 	@fn load_from_d 
 * 	@brief (Theta,b) on device GPU -> std::vector on host 
 * */
void Axon::load_from_d(std::vector<float>& h_Theta, std::vector<float>& h_b) {
	const int SIZE_THETA = s_l*s_lm1;

	cudaMemcpy(h_Theta.data(), Theta.get(), SIZE_THETA*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b.data(), b.get(), s_l*sizeof(float),cudaMemcpyDeviceToHost);

}		

// for loading input data X into layer l-1, alm1
/**
 * 	@fn load_from_hXvec 
 * 	@brief load from host, X input data, as a std::vector<float>
 *  @param const int m - number of examples
 * */
void Axon::load_from_hXvec(std::vector<float>& h_X, const int m) {
	const int SIZE_S_LM1 = m * s_lm1;
	
	if (!alm1.get()) {
		std::shared_ptr<float> d_alm1(new float[SIZE_S_LM1], deleterRR_struct() ); 	// d_alm1; alm1 on device GPU
		cudaMallocManaged((void **) &d_alm1,SIZE_S_LM1*sizeof(float));
		alm1 = std::move(d_alm1);
		cudaMemcpy(alm1.get(), h_X.data(), SIZE_S_LM1 *sizeof(float),cudaMemcpyHostToDevice);
		
	} else {
		cudaMemcpy(alm1.get(), h_X.data(), SIZE_S_LM1 *sizeof(float),cudaMemcpyHostToDevice);
	}
	this->m = m;
}

	/** We're not transferring ownership, so we don't use std::move
	 * @ref https://stackoverflow.com/questions/41871115/why-would-i-stdmove-an-stdshared-ptr
	 * */
void Axon::load_alm1_from_ptr(std::shared_ptr<float> & ptr_sh_input_layer) 
{
	alm1 = ptr_sh_input_layer;
}

/** We're transferring ownership, so we  use std::move
  * @ref https://stackoverflow.com/questions/41871115/why-would-i-stdmove-an-stdshared-ptr
  * */
void Axon::move2al_from_ptr(std::shared_ptr<float> & ptr_sh_output_layer) 
{
	al = std::move( ptr_sh_output_layer );
}

void Axon::move2alm1_from_ptr(std::shared_ptr<float> & ptr_sh_input_layer) 
{
	alm1 = std::move( ptr_sh_input_layer );
}


// initialize layer l
/**
 * 	@fn init_al 
 * 	@brief initialize layer l
 *  @param const int m - number of examples
 * */
void Axon::init_al(const int m) { 
	const int SIZE_S_L = m * s_l;

	std::shared_ptr<float> d_al(new float[SIZE_S_L], deleterRR_struct() ); 	// d_al; al on device GPU
	cudaMallocManaged((void **) &d_al,SIZE_S_L*sizeof(float));
	al = std::move(d_al);
	cudaMemset(al.get(), 0.f, SIZE_S_L*sizeof(float));

	this->m = m;
}



// for getting size dimensions
std::vector<int> Axon::getSizeDims() {
	std::vector<int> sizedimsvec = { s_lm1, s_l, m };
	return sizedimsvec;
}


// for getting Theta,b, and lth layer al, zl (after activation function applied)

std::unique_ptr<float[], deleterRR_struct> Axon::getTheta() {
	auto ptr = std::move(Theta);
	return ptr;
}

std::unique_ptr<float[],deleterRR_struct> Axon::getb() {
	auto ptr = std::move(b);
	return ptr;
}

void Axon::move2Theta_from_ptr(std::unique_ptr<float[], deleterRR_struct> & ptr_Theta) 
{
	Theta = std::move( ptr_Theta );
}

void Axon::move2b_from_ptr(std::unique_ptr<float[], deleterRR_struct> & ptr_b) 
{
	b = std::move( ptr_b );
}

/**
 * @fn Axon::getalm1
 * @details we don't use std::move because we don't want to change (move) 
 * 	ownership of the pointer (and the memory it points to) because we're 
 *  dealing with a shared_ptr (you can move it, but then we'd want to use a 
 * 	unique_ptr; we want to share it)
 * */
std::shared_ptr<float> Axon::getalm1() {
	auto ptr = alm1;
	return ptr;
}

/**
 * @fn Axon::getal
 * @details we don't use std::move because we don't want to change (move) 
 * 	ownership of the pointer (and the memory it points to) because we're 
 *  dealing with a shared_ptr (you can move it, but then we'd want to use a 
 * 	unique_ptr; we want to share it)
 * */
std::shared_ptr<float> Axon::getal() {
	auto ptr = al;
	return ptr;
}


/* =============== "connect" the Axon =============== */
/* Once Axon has been setup, by the above, do the following to 
/* "connect through" the Axon */

/**
 *  @fn rightMul
 *  @class Axon_
 * 	@brief right multiplication
 * */
void Axon::rightMul() {
	float a1 = 1.0f;
	float bet = 0.f;
	
	std::unique_ptr<cublasHandle_t,del_cublasHandle_struct> handle_u(
		new cublasHandle_t);
	cublasCreate(handle_u.get());
		
	cublasSgemm(*handle_u.get(),CUBLAS_OP_N,CUBLAS_OP_N,m,s_l,s_lm1,
		&a1, alm1.get(),m, Theta.get(),s_lm1,&bet,al.get(),m);

}

	/* ========== Add bias ========== */
/** 
 * 	@fn Axon::addb
 * 	@param const int N_x = number of (thread) blocks on grid in x-direction
 *  @param const int M_x = number of threads in a (single, thread) block in x-direction
 * 	@details N_x, M_x determined before by feedfwd class
 * */
void Axon::addb(const int M_x,const int N_x ) {

	/* ===== grid, thread block size dimensions ===== */
	const int SIZE_A_L = m * s_l; // m * s_l = (number of examples)*(size dim. or no. of nodes of lth layer)
	
	// M_x = number of threads in a (single) block in x-direction
	int Nx = 0;
	if (N_x == 0) { 
		const int Nx_calc = (SIZE_A_L + M_x -1)/M_x;
		Nx = max( Nx_calc, N_x);
	} else {
		Nx = N_x;
	}

	// create 1s array, array of 1s
	std::unique_ptr<float[], deleterRR_struct> ones(new float[SIZE_A_L], deleterRR_struct());
	cudaMallocManaged((void **) &ones, SIZE_A_L*sizeof(float));

	setconstval_kernel<<<Nx,M_x>>>(SIZE_A_L,1.0f, ones.get() );

	// create "broadcasted" array for bias b
	std::unique_ptr<float[], deleterRR_struct> broadcast_b(new float[SIZE_A_L], deleterRR_struct());
	cudaMallocManaged((void **) &broadcast_b, SIZE_A_L*sizeof(float));

	std::unique_ptr<cublasHandle_t,del_cublasHandle_struct> handle_u(
		new cublasHandle_t);
	cublasCreate(handle_u.get());	

	// C = A x diag(X)
	cublasSdgmm( *handle_u.get(), CUBLAS_SIDE_RIGHT, 
		m, s_l, ones.get(), m, b.get(), 1, broadcast_b.get(), m);

	// C = \alpha op(A) + \beta op(B)
	float a1 = 1.0f; 
	float bet = 1.0f;
	cublasSgeam(*handle_u.get(), CUBLAS_OP_N, CUBLAS_OP_N, 
		m, s_l, 
		&a1, broadcast_b.get(), m, &bet, al.get(), m, 
		al.get(), m);


}
	


// destructor
Axon::~Axon() {}


/* =============== Axon class; with activation =============== */
// Constructor
Axon_act::Axon_act(const int s_lm1,const int s_l, const int idx_actf) : 
		Axon(s_lm1, s_l), idx_actf(idx_actf) { }

// Move Constructor
/**
 *  @fn Axon_act(const Axon& old_axon)
 *  @brief copy constructor for Axon class
 * 	@ref http://www.geeksforgeeks.org/copy-constructor-in-cpp/
 * https://stackoverflow.com/questions/16030081/copy-constructor-for-a-class-with-unique-ptr
 * https://en.wikipedia.org/wiki/C%2B%2B11#Rvalue_references_and_move_constructors
 * https://msdn.microsoft.com/en-us/library/s16xw1a8.aspx
 * */
Axon_act::Axon_act(Axon_act&& old_axon) 
	: 	Axon(std::move(old_axon)), // error: function "Axon::Axon(const Axon &)" (declared implicitly) cannot be referenced -- it is a deleted function

	 zl(std::move(old_axon.zl)),
	 Dpsil(std::move(old_axon.Dpsil))
{
	idx_actf = old_axon.idx_actf;
}


// operator overload assignment = 
Axon_act & Axon_act::operator=(Axon_act && old_axon) 
{

	idx_actf = old_axon.idx_actf;

	zl = std::move( old_axon.zl );
	Dpsil = std::move( old_axon.Dpsil);

	return *this;
}

// initialize layer l
/**
 * 	@fn init_zlal 
 * 	@brief initialize layer l
 *  @param const int m - number of examples
 * */
void Axon_act::init_zlal(const int m) { 
	const int SIZE_S_L = m * s_l;

	std::shared_ptr<float> d_al(new float[SIZE_S_L], deleterRR_struct() ); 	// d_al; al on device GPU
	cudaMallocManaged((void **) &d_al,SIZE_S_L*sizeof(float));
	al = std::move(d_al);
	cudaMemset(al.get(), 0.f, SIZE_S_L*sizeof(float));

	std::unique_ptr<float[], deleterRR_struct> d_zl(new float[SIZE_S_L], deleterRR_struct());
	cudaMallocManaged((void **) &d_zl,SIZE_S_L*sizeof(float));
	zl = std::move(d_zl);

	this->m = m;
}

// for getting Theta,b, and lth layer al, zl (after activation function applied)

std::unique_ptr<float[],deleterRR_struct> Axon_act::getzl() {
	auto ptr = std::move(zl);
	return ptr;
}

std::unique_ptr<float[],deleterRR_struct> Axon_act::getDpsil() {
	auto ptr = std::move(Dpsil);
	return ptr;
}


/* =============== "connect" the Axon =============== */
/* Once Axon has been setup, by the above, do the following to 
/* "connect through" the Axon */
/**
 *  @fn rightMul
 *  @class Axon_act
 * 	@brief right multiplication
 * */
void Axon_act::rightMul() {
	float a1 = 1.0f;
	float bet = 0.f;
	
	std::unique_ptr<cublasHandle_t,del_cublasHandle_struct> handle_u(
		new cublasHandle_t);
	cublasCreate(handle_u.get());
	
	cublasSgemm(*handle_u.get(),CUBLAS_OP_N,CUBLAS_OP_N,m,s_l,s_lm1,
		&a1,alm1.get(),m,Theta.get(),s_lm1,&bet,zl.get(),m);


}



	/* ========== Add bias ========== */
/** 
 * 	@fn Axon_act::addb
 * 	@param const int N_x = number of (thread) blocks on grid in x-direction
 *  @param const int M_x = number of threads in a (single, thread) block in x-direction
 * 	@details N_x, M_x determined before by feedfwd class
 * */
void Axon_act::addb(const int M_x, const int N_x) {

	/* ===== grid, thread block size dimensions ===== */
	const int SIZE_Z_L = m * s_l; // m * s_l = (number of examples)*(size dim. or no. of nodes of lth layer)
	
	// M_x = number of threads in a (single) block in x-direction
	int Nx = 0;
	if (N_x == 0) { 
		const int Nx_calc = (SIZE_Z_L + M_x -1)/M_x;
		Nx = max( Nx_calc, N_x);
	} else {
		Nx = N_x;
	}

	// create 1s array, array of 1s
	std::unique_ptr<float[], deleterRR_struct> ones(new float[SIZE_Z_L], deleterRR_struct());
	cudaMallocManaged((void **) &ones, SIZE_Z_L*sizeof(float));

	setconstval_kernel<<<Nx,M_x>>>(SIZE_Z_L,1.0f, ones.get() );

	// create "broadcasted" array for bias b
	std::unique_ptr<float[], deleterRR_struct> broadcast_b(new float[SIZE_Z_L], deleterRR_struct());
	cudaMallocManaged((void **) &broadcast_b, SIZE_Z_L*sizeof(float));

	std::unique_ptr<cublasHandle_t,del_cublasHandle_struct> handle_u(
		new cublasHandle_t);
	cublasCreate(handle_u.get());	

	// C = A x diag(X)
	cublasSdgmm( *handle_u.get(), CUBLAS_SIDE_RIGHT, 
		m, s_l, ones.get(), m, b.get(), 1, broadcast_b.get(), m);

	// C = \alpha op(A) + \beta op(B)
	float a1 = 1.0f; 
	float bet = 1.0f;
	cublasSgeam(*handle_u.get(), CUBLAS_OP_N, CUBLAS_OP_N, 
		m, s_l, 
		&a1, broadcast_b.get(), m, &bet, zl.get(), m, 
		zl.get(), m);

}

/* ========== activate with activation function ========== */
void Axon_act::actf( const int M_x, const int N_x) {

	/* ===== grid, thread block size dimensions ===== */
	const int SIZE_Z_L = m * s_l; // m * s_l = (number of examples)*(size dim. or no. of nodes of lth layer)

	cudaMemcpy(al.get(), zl.get(), sizeof(float) * SIZE_Z_L, cudaMemcpyDeviceToDevice) ; 
	
	// M_x = number of threads in a (single) block in x-direction
	int Nx = 0;
	if (N_x == 0) { 
		const int Nx_calc = (SIZE_Z_L + M_x -1)/M_x;
		Nx = max( Nx_calc, N_x);
	} else {
		Nx = N_x;
	}

	/** using array of function ptr doesn't work because it has to be located to device code and, refer here: 
	 * @ref http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-pointers
	 * https://devtalk.nvidia.com/default/topic/457094/cuda-programming-and-performance/how-can-i-use-__device__-function-pointer-in-cuda-/3
	 * https://stackoverflow.com/questions/15644261/cuda-function-pointers/15646771#15646771
	general_activation_function_kernel<<<Nx,M_x>>>( SIZEDIM_Z_L, ptr_zl.get(), idx_actf );
	*/
	if (idx_actf==0) {
		identity_kernel<<<Nx,M_x>>>(SIZE_Z_L, al.get()); 
	} else if (idx_actf==1) {
		sigmoid_kernel<<<Nx,M_x>>>(SIZE_Z_L, al.get() );
	} else if (idx_actf==2) {
		tanh_kernel<<<Nx,M_x>>>(SIZE_Z_L, al.get() );
	} else if (idx_actf==3) {
		tanh_kernel<<<Nx,M_x>>>(SIZE_Z_L, al.get() );
	} else if (idx_actf==4) {
		arctan_kernel<<<Nx,M_x>>>(SIZE_Z_L, al.get() );
	} else if (idx_actf==5) {
		ReLU_kernel<<<Nx,M_x>>>(SIZE_Z_L, al.get() );
	}	


} 


/* ========== partial derivatives with respect to z^l of psi^l(z^l) ========== */
void Axon_act::do_Dpsi( const int M_x, const int N_x) {
	// initialize (i.e. instantiate, construct) 
	const int SIZE_Z_L = m * s_l;

	std::unique_ptr<float[], deleterRR_struct> d_Dpsi(new float[SIZE_Z_L], deleterRR_struct());
	cudaMallocManaged((void **) &d_Dpsi,SIZE_Z_L*sizeof(float));


	/* ===== grid, thread block size dimensions ===== */
	cudaMemcpy(d_Dpsi.get(), zl.get(), sizeof(float) * SIZE_Z_L, cudaMemcpyDeviceToDevice) ; 
	
	// M_x = number of threads in a (single) block in x-direction
	int Nx = 0;
	if (N_x == 0) { 
		const int Nx_calc = (SIZE_Z_L + M_x -1)/M_x;
		Nx = max( Nx_calc, N_x);
	} else {
		Nx = N_x;
	}

	/** using array of function ptr doesn't work because it has to be located to device code and, refer here: 
	 * @ref http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-pointers
	 * https://devtalk.nvidia.com/default/topic/457094/cuda-programming-and-performance/how-can-i-use-__device__-function-pointer-in-cuda-/3
	 * https://stackoverflow.com/questions/15644261/cuda-function-pointers/15646771#15646771
	general_activation_function_kernel<<<Nx,M_x>>>( SIZEDIM_Z_L, ptr_zl.get(), idx_actf );
	*/
	if (idx_actf==0) {
		D_identity_kernel<<<Nx,M_x>>>(SIZE_Z_L, zl.get(), d_Dpsi.get() );
	} else if (idx_actf==1) {
		D_sigmoid_kernel<<<Nx,M_x>>>(SIZE_Z_L, zl.get(), d_Dpsi.get() );
	} else if (idx_actf==2) {
		D_tanh_kernel<<<Nx,M_x>>>(SIZE_Z_L, zl.get(), d_Dpsi.get() );
	} else if (idx_actf==3) {
		D_tanh_kernel<<<Nx,M_x>>>(SIZE_Z_L, zl.get(), d_Dpsi.get() );
	} else if (idx_actf==4) {
		D_arctan_kernel<<<Nx,M_x>>>(SIZE_Z_L, zl.get(), d_Dpsi.get() );
	} else if (idx_actf==5) {
		D_ReLU_kernel<<<Nx,M_x>>>(SIZE_Z_L, zl.get(), d_Dpsi.get() );
	}	

	// Remember to move ptr_zl back to zl
	Dpsil = std::move(d_Dpsi);

}


