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
__global__ void addb_kernel(const int m, const int s_l, float* a_l,const float* b) {
	int k = threadIdx.x + blockDim.x * blockIdx.x ; // k is global thread index 

	// assume COLUMN-major ordering 
	int j = k/m; 	// this is the jth index in matrix a_l(i,j)

	__shared__ float sh_bj[1]; // shared bj, jth component of b
	
	if ( j >= s_l) // check if j is access element outside of b\in \mathbb{R}^{s_l}
	{ return; } else { 
		sh_bj[0] = b[j]; 
	}
	int SIZEDIM_A_L = m*s_l;
	/* check if we've launched too many threads than needed to compute
	 * over all of s_l \in \text{Mat}_{\mathbb{R}}(m,s_l) - this could be the case 
	 * given arbitrary thread blocks launched in <<<>>>
	 * */
	if (k >= SIZEDIM_A_L)  
	{
		return ; 
	}
	__syncthreads();
	
	a_l[k] = a_l[k] + sh_bj[0];
}


/* =============== activation function =============== */

// general activation functions to plug in these function pointers  

__global__ void general_activation_function_kernel(const int SIZE,float* z,const int idx_actf) {
	int k_x = threadIdx.x + blockDim.x * blockIdx.x;  
	if (k_x >= SIZE) { return; }
	
	/* this for loop ensure that, if in case SIZE > gridDimx.x*blockDim.x (or even if  
	 * 	SIZE >> gridDimx.x*blockDim.x so we have more values to compute than threads on a GPU!)  
	 * that everything gets computed) */
	for (int tid=k_x; k_x < SIZE; k_x += gridDim.x*blockDim.x ) { 
		float a_val = 0.f;
		a_val = z[tid];
	
//		a_val = (d_activat_fs[idx_actf])(a_val);
		z[tid] = a_val;
	}
}

// the derivative of an activation function, denoted with D
__global__ void general_Dactivation_function_kernel(const int SIZE,const float* z,float* d_a,const int idx_actf)  {
	int k_x = threadIdx.x + blockDim.x * blockIdx.x;  
	if (k_x >= SIZE) { return; }
	
	/* this for loop ensure that, if in case SIZE > gridDimx.x*blockDim.x (or even if  
	 * 	SIZE >> gridDimx.x*blockDim.x so we have more values to compute than threads on a GPU!)  
	 * that everything gets computed) */
	for (int tid=k_x; k_x < SIZE; k_x += gridDim.x*blockDim.x ) { 
		float a_val = 0.f;
		a_val = z[tid];
	
//		a_val = (D_activat_fs[idx_actf])(a_val);
		d_a[tid] = a_val;
	}
}


/* =============== END of activation functions =============== */



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

std::shared_ptr<float> Axon::getalm1() {
	auto ptr = std::move(alm1);
	return ptr;
}


std::shared_ptr<float> Axon::getal() {
	auto ptr = std::move(al);
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
	
	auto A_u = std::move( alm1 );
	auto B_u = std::move( Theta );
	auto C_u = std::move( al );
	
	cublasSgemm(*handle_u.get(),CUBLAS_OP_N,CUBLAS_OP_N,m,s_l,s_lm1,&a1,A_u.get(),m,B_u.get(),s_lm1,&bet,C_u.get(),m);

	cudaDeviceSynchronize();

	alm1 = std::move(A_u);
	Theta = std::move(B_u);
	al = std::move(C_u);
}

	/* ========== Add bias ========== */
/** 
 * 	@fn Axon::addb
 * 	@param const int N_x = number of (thread) blocks on grid in x-direction
 *  @param const int M_x = number of threads in a (single, thread) block in x-direction
 * 	@details N_x, M_x determined before by feedfwd class
 * */
void Axon::addb(const int M_x,const int N_x ) {
	auto ptr_al = std::move( al );
	auto ptr_b  = std::move(b);

	/* ===== grid, thread block size dimensions ===== */
	const int SIZEDIM_A_L = m * s_l; // m * s_l = (number of examples)*(size dim. or no. of nodes of lth layer)
	
	// M_x = number of threads in a (single) block in x-direction
	const int Nx_calc = (SIZEDIM_A_L + M_x -1)/M_x;

	int Nx = max( Nx_calc, N_x);

	addb_kernel<<<Nx,M_x>>>(m,s_l,ptr_al.get(),ptr_b.get() );

	al = std::move(ptr_al);
	b  = std::move(ptr_b);
}
	


// destructor
Axon::~Axon() {}


/* =============== Axon class; with activation =============== */
// Constructor
Axon_act::Axon_act(const int s_lm1,const int s_l, const int idx_actf) : 
		Axon(s_lm1, s_l), idx_actf(idx_actf)   {
}

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
//: Axon(old_axon) 
	{

	idx_actf = old_axon.idx_actf;

	zl = std::move( old_axon.zl );
	Dpsil = std::move( old_axon.Dpsil );

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

// for getting Theta,b, and lth layer zl, Dpsil (after activation function applied)

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
	
	auto A_u = std::move( alm1 );
	auto B_u = std::move( Theta );
	auto C_u = std::move( zl );
	
	cublasSgemm(*handle_u.get(),CUBLAS_OP_N,CUBLAS_OP_N,m,s_l,s_lm1,&a1,A_u.get(),m,B_u.get(),s_lm1,&bet,C_u.get(),m);

	cudaDeviceSynchronize();

	alm1 = std::move(A_u);
	Theta = std::move(B_u);
	zl = std::move(C_u);
}


	/* ========== Add bias ========== */
/** 
 * 	@fn Axon_act::addb
 * 	@param const int N_x = number of (thread) blocks on grid in x-direction
 *  @param const int M_x = number of threads in a (single, thread) block in x-direction
 * 	@details N_x, M_x determined before by feedfwd class
 * */
void Axon_act::addb(const int M_x, const int N_x) {
	auto ptr_zl = std::move( zl );
	auto ptr_b  = std::move(b);

	/* ===== grid, thread block size dimensions ===== */
	const int SIZEDIM_Z_L = m * s_l; // m * s_l = (number of examples)*(size dim. or no. of nodes of lth layer)
	
	// M_x = number of threads in a (single) block in x-direction
	const int Nx_calc = (SIZEDIM_Z_L + M_x -1)/M_x;

	if (N_x ==0) { 
		int Nx = max(N_x, Nx_calc);
	} else { int Nx = N_x; }  

//	int Nx = max(N_x, Nx_calc);
	
//	addb_kernel<<<Nx,M_x>>>(m,s_l,ptr_zl.get(),ptr_b.get() );
	addb_kernel<<<N_x,M_x>>>(m,s_l,ptr_zl.get(),ptr_b.get() );


	zl = std::move(ptr_zl);
	b  = std::move(ptr_b);
}

/* ========== activate with activation function ========== */
void Axon_act::actf( const int M_x, const int N_x) {
/*	auto ptr_zl = std::move( zl );
	auto ptr_al = std::move( al );
*/
	/* ===== grid, thread block size dimensions ===== */
	const int SIZEDIM_Z_L = m * s_l; // m * s_l = (number of examples)*(size dim. or no. of nodes of lth layer)

//	cudaMemcpy(ptr_al.get(), ptr_zl.get(), sizeof(float) * SIZEDIM_Z_L, cudaMemcpyDeviceToDevice) ; 
	cudaMemcpy(al.get(), zl.get(), sizeof(float) * SIZEDIM_Z_L, cudaMemcpyDeviceToDevice) ; 
		
	// M_x = number of threads in a (single) block in x-direction
	const int Nx_calc = (SIZEDIM_Z_L + M_x -1)/M_x;

	if (N_x ==0) { 
		int Nx = max(N_x, Nx_calc);
	} else { int Nx = N_x; }  

	/** using array of function ptr doesn't work because it has to be located to device code and, refer here: 
	 * @ref http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-pointers
	 * https://devtalk.nvidia.com/default/topic/457094/cuda-programming-and-performance/how-can-i-use-__device__-function-pointer-in-cuda-/3
	 * https://stackoverflow.com/questions/15644261/cuda-function-pointers/15646771#15646771
	general_activation_function_kernel<<<Nx,M_x>>>( SIZEDIM_Z_L, ptr_zl.get(), idx_actf );
	*/
/*	if (idx_actf==1) {
		sigmoid_kernel<<<Nx,M_x>>>(SIZEDIM_Z_L, ptr_al.get() );
	} else if (idx_actf==2) {
		tanh_kernel<<<Nx,M_x>>>(SIZEDIM_Z_L, ptr_al.get() );
	} else if (idx_actf==3) {
		tanh_kernel<<<Nx,M_x>>>(SIZEDIM_Z_L, ptr_al.get() );
	} else if (idx_actf==4) {
		arctan_kernel<<<Nx,M_x>>>(SIZEDIM_Z_L, ptr_al.get() );
	} else if (idx_actf==5) {
		ReLU_kernel<<<Nx,M_x>>>(SIZEDIM_Z_L, ptr_al.get() );
	}	

*/
	if (idx_actf==0) {
		identity_kernel<<<Nx,M_x>>>(SIZEDIM_Z_L, al.get() );
	}
	else if (idx_actf==1) {
		sigmoid_kernel<<<Nx,M_x>>>(SIZEDIM_Z_L, al.get() );
	} else if (idx_actf==2) {
		tanh_kernel<<<Nx,M_x>>>(SIZEDIM_Z_L, al.get() );
	} else if (idx_actf==3) {
		tanh_kernel<<<Nx,M_x>>>(SIZEDIM_Z_L, al.get() );
	} else if (idx_actf==4) {
		arctan_kernel<<<Nx,M_x>>>(SIZEDIM_Z_L, al.get() );
	} else if (idx_actf==5) {
		ReLU_kernel<<<Nx,M_x>>>(SIZEDIM_Z_L, al.get() );
	}	


//	zl = std::move(ptr_zl);	
//	al = std::move(ptr_al);	
} 

/* ========== partial derivatives with respect to z^l of psi^l(z^l) ========== */
void Axon_act::do_Dpsi( const int M_x, const int N_x) {
	// initialize (i.e. instantiate, construct) 
	const int SIZEDIM_Z_L = m * s_l;

	std::unique_ptr<float[], deleterRR_struct> d_Dpsi(new float[SIZEDIM_Z_L], deleterRR_struct());
	cudaMallocManaged((void **) &d_Dpsi,SIZEDIM_Z_L*sizeof(float));

	auto ptr_zl = std::move( zl );

	/* ===== grid, thread block size dimensions ===== */
	cudaMemcpy(d_Dpsi.get(), ptr_zl.get(), sizeof(float) * SIZEDIM_Z_L, cudaMemcpyDeviceToDevice) ; 
	
	// M_x = number of threads in a (single) block in x-direction
	const int Nx_calc = (SIZEDIM_Z_L + M_x -1)/M_x;

	if (N_x ==0) { 
		int Nx = max(N_x, Nx_calc);
	} else { int Nx = N_x; }  

	/** using array of function ptr doesn't work because it has to be located to device code and, refer here: 
	 * @ref http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-pointers
	 * https://devtalk.nvidia.com/default/topic/457094/cuda-programming-and-performance/how-can-i-use-__device__-function-pointer-in-cuda-/3
	 * https://stackoverflow.com/questions/15644261/cuda-function-pointers/15646771#15646771
	general_activation_function_kernel<<<Nx,M_x>>>( SIZEDIM_Z_L, ptr_zl.get(), idx_actf );
	*/
	if (idx_actf==0) {
		D_identity_kernel<<<Nx,M_x>>>(SIZEDIM_Z_L, ptr_zl.get(), d_Dpsi.get() );
	} else if (idx_actf==1) {
		D_sigmoid_kernel<<<Nx,M_x>>>(SIZEDIM_Z_L, ptr_zl.get(), d_Dpsi.get() );
	} else if (idx_actf==2) {
		D_tanh_kernel<<<Nx,M_x>>>(SIZEDIM_Z_L, ptr_zl.get(), d_Dpsi.get() );
	} else if (idx_actf==3) {
		D_tanh_kernel<<<Nx,M_x>>>(SIZEDIM_Z_L, ptr_zl.get(), d_Dpsi.get() );
	} else if (idx_actf==4) {
		D_arctan_kernel<<<Nx,M_x>>>(SIZEDIM_Z_L, ptr_zl.get(), d_Dpsi.get() );
	} else if (idx_actf==5) {
		D_ReLU_kernel<<<Nx,M_x>>>(SIZEDIM_Z_L, ptr_zl.get(), d_Dpsi.get() );
	}	

	// Remember to move ptr_zl back to zl
	zl = std::move(ptr_zl);	
	Dpsil = std::move(d_Dpsi);

}
