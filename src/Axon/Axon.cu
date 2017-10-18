/**
 * @file   : Axon.cu
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
 * nvcc -std=c++14 -lcublas -dc Axon.cu -o Axon.o
 * 
 * */
#include "Axon.h"

/**
 *  @class Axon_u
 *  @brief Axon, using unique pointers, contains weights matrix (Theta), and bias b between 2 layers
 */

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
//	int i = k % m; 	// this the ith index in a matrix a_l(i,j)
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

// Copy Constructor
/**
 *  @fn Axon(const Axon& old_axon)
 *  @brief copy constructor for Axon class
 * 	@ref http://www.geeksforgeeks.org/copy-constructor-in-cpp/
 * https://stackoverflow.com/questions/16030081/copy-constructor-for-a-class-with-unique-ptr
 * https://en.wikipedia.org/wiki/C%2B%2B11#Rvalue_references_and_move_constructors
 * */
Axon::Axon(Axon&& old_axon) : Theta(std::move(old_axon.Theta)), b(std::move(old_axon.b))
{

//	alm1 = std::move( old_axon.alm1 );
//	al = std::move( old_axon.al );

/*	std::vector<int> old_sizeDims = old_axon.getSizeDims();
	s_lm1 = old_sizeDims[0]; 
	s_l = old_sizeDims[1];
	m = old_sizeDims[2];
*/
	s_lm1 = old_axon.s_lm1;
	s_l = old_axon.s_l;
	m = old_axon.m;
	
	l = old_axon.l; // lth layer
	//del_cublasHandle_struct = old_axon.del_cublasHandle_struct;
	
/*	Theta = std::move( old_axon.getTheta() );
	b = std::move( old_axon.getb( ));
	alm1 = std::move( old_axon.getalm1() );
	al = std::move( old_axon.getal() );
*/
//	Theta = std::move(old_axon.Theta);
//	b = std::move( old_axon.b );
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

std::shared_ptr<float> Axon::getalm1() {
	auto ptr = std::move(alm1);
	return ptr;
}


std::shared_ptr<float> Axon::getal() {
	auto ptr = std::move(al);
	return ptr;
}


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
void Axon::addb(const int M_x) {
	auto ptr_al = std::move( al );
	auto ptr_b  = std::move(b);

	/* ===== grid, thread block size dimensions ===== */
	const int SIZEDIM_A_L = m * s_l; // m * s_l = (number of examples)*(size dim. or no. of nodes of lth layer)
	
	// M_x = number of threads in a (single) block in x-direction
	const int Nx = (SIZEDIM_A_L + M_x -1)/M_x;

	addb_kernel<<<Nx,M_x>>>(m,s_l,ptr_al.get(),ptr_b.get() );

	al = std::move(ptr_al);
	b  = std::move(ptr_b);
}
	


// destructor
Axon::~Axon() {}







