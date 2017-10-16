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
 * nvcc -std=c++14 -dc Axon.cu -o Axon.o
 * 
 * */
#include "Axon.h"

/**
 *  @class Axon_u
 *  @brief Axon, using unique pointers, contains weights matrix (Theta), and bias b between 2 layers
 */

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


// initialize layer l
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

// destructor
Axon::~Axon() {}




/**
 *  @class Axon_sh
 *  @brief Axon, using shared pointers, contains weights matrix (Theta), and bias b between 2 layers
 */

// constructor 
Axon_sh::Axon_sh(const int s_lm1,const int s_l) : s_lm1(s_lm1), s_l(s_l)  {
	const int SIZE_THETA = s_l*s_lm1;
	std::shared_ptr<float> d_Theta(new float[SIZE_THETA], deleterRR()); 	// d_Theta; Theta on device GPU
	cudaMallocManaged((void **) &d_Theta,SIZE_THETA*sizeof(float));
	Theta = std::move(d_Theta);

	std::shared_ptr<float> d_b(new float[s_l], deleterRR()); 	// d_Theta; Theta on device GPU
	cudaMallocManaged((void **) &d_b,s_l*sizeof(float));
	b = std::move(d_b);

}

// member functions
void Axon_sh::load_from_hvec(std::vector<float>& h_Theta,std::vector<float>& h_b) {
	const int SIZE_THETA = s_l*s_lm1;

	cudaMemcpy(Theta.get(), h_Theta.data(), SIZE_THETA*sizeof(float),cudaMemcpyHostToDevice);	
	cudaMemcpy(b.get(), h_b.data(), s_l*sizeof(float),cudaMemcpyHostToDevice);	
}	

/**
 * 	@fn load_from_d 
 * 	@brief (Theta,b) on device GPU -> std::vector on host 
 * */
void Axon_sh::load_from_d(std::vector<float>& h_Theta, std::vector<float>& h_b) {
	const int SIZE_THETA = s_l*s_lm1;

	cudaMemcpy(h_Theta.data(), Theta.get(), SIZE_THETA*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_b.data(), b.get(), s_l*sizeof(float),cudaMemcpyDeviceToHost);

}		

void Axon_sh::load_from_uniq(std::unique_ptr<float[],deleterRR> & ptr_unique_Theta, 
								std::unique_ptr<float[],deleterRR> & ptr_unique_b) {
	Theta = std::move(ptr_unique_Theta) ;
	b = std::move(ptr_unique_b) ;
}

// for loading input data X into layer l-1, alm1
/**
 * 	@fn load_from_hXvec 
 * 	@brief load from host, X input data, as a std::vector<float>
 *  @param const int m - number of examples
 * */
void Axon_sh::load_from_hXvec(std::vector<float>& h_X, const int m) {
	const int SIZE_S_LM1 = m * s_lm1;

	if (!alm1.get()) {
		std::shared_ptr<float> d_alm1(new float[SIZE_S_LM1], deleterRR()); 	// d_alm1; alm1 on device GPU
		cudaMallocManaged((void **) &d_alm1,SIZE_S_LM1*sizeof(float));
		alm1 = std::move(d_alm1);
		cudaMemcpy(alm1.get(), h_X.data(), SIZE_S_LM1 *sizeof(float),cudaMemcpyHostToDevice);
		
	} else {
		cudaMemcpy(alm1.get(), h_X.data(), SIZE_S_LM1 *sizeof(float),cudaMemcpyHostToDevice);
	}

	this->m = m;
}

// initialize layer l
void Axon_sh::init_al(const int m) { 
	const int SIZE_S_L = m * s_l;

	std::shared_ptr<float> d_al(new float[SIZE_S_L], deleterRR()); 	// d_al; al on device GPU
	cudaMallocManaged((void **) &d_al,SIZE_S_L*sizeof(float));
	al = std::move(d_al);
	cudaMemset(al.get(), 0.f, SIZE_S_L*sizeof(float));

	this->m = m;
}


// for getting size dimensions
std::vector<int> Axon_sh::getSizeDims() {
	std::vector<int> sizedimsvec = { s_lm1, s_l, m };
	return sizedimsvec;
}


// for getting Theta,b, and lth layer al, zl (after activation function applied)

std::shared_ptr<float> Axon_sh::getTheta() {
	auto ptr = std::move(Theta);
	return ptr;
}

std::shared_ptr<float> Axon_sh::getb() {
	auto ptr = std::move(b);
	return ptr;
}

std::shared_ptr<float> Axon_sh::getalm1() {
	auto ptr = std::move(alm1);
	return ptr;
}


std::shared_ptr<float> Axon_sh::getal() {
	auto ptr = std::move(al);
	return ptr;
}

/**
 *  @fn rightMul
 *  @class Axon_sh
 * 	@brief right multiplication
 * */
void Axon_sh::rightMul() {
	auto A_sh = std::move( alm1 );
	auto B_sh = std::move( Theta );
	auto C_sh = std::move( al );

//	Prod_sh( m, s_l, s_lm1, 1.0f, A_sh, B_sh, 0.f, C_sh);

	auto del_cublasHandle=[&](cublasHandle_t* ptr) { cublasDestroy(*ptr); };

	// moved the cublasHandle_t environment into the product itself
	std::shared_ptr<cublasHandle_t> handle_sh(
		new cublasHandle_t, del_cublasHandle);
	cublasCreate(handle_sh.get());

	float a1 = 1.0f;
	float bet = 0.f;

	cublasSgemm(*handle_sh.get(),CUBLAS_OP_N,CUBLAS_OP_N,m,s_l,s_lm1,&a1,A_sh.get(),m,B_sh.get(),s_lm1,&bet,C_sh.get(),m);

	alm1 = std::move( A_sh);
	Theta = std::move( B_sh );
	al = std::move( C_sh );
} 

// destructor
Axon_sh::~Axon_sh() {}


/**
 *  @class Axon_u
 *  @brief Axon, using unique pointers, contains weights matrix (Theta), and bias b between 2 layers
 */

// constructor 
Axon_u::Axon_u(const int s_lm1,const int s_l) : s_lm1(s_lm1), s_l(s_l)  {
	const int SIZE_THETA = s_l*s_lm1;

	std::unique_ptr<float[], deleterRR_struct> d_Theta(new float[SIZE_THETA]);
	cudaMallocManaged((void **) &d_Theta,SIZE_THETA*sizeof(float));
	Theta = std::move(d_Theta);

	std::unique_ptr<float[], deleterRR_struct> d_b(new float[s_l]);
	cudaMallocManaged((void **) &d_b,s_l*sizeof(float));
	b = std::move(d_b);
}

// member functions
void Axon_u::load_from_hvec(std::vector<float>& h_Theta,std::vector<float>& h_b) {
	const int SIZE_THETA = s_l*s_lm1;

	cudaMemcpy(Theta.get(), h_Theta.data(), SIZE_THETA*sizeof(float),cudaMemcpyHostToDevice);	
	cudaMemcpy(b.get(), h_b.data(), s_l*sizeof(float),cudaMemcpyHostToDevice);	
}	

/**
 * 	@fn load_from_d 
 * 	@brief (Theta,b) on device GPU -> std::vector on host 
 * */
void Axon_u::load_from_d(std::vector<float>& h_Theta, std::vector<float>& h_b) {
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
void Axon_u::load_from_hXvec(std::vector<float>& h_X, const int m) {
	const int SIZE_S_LM1 = m * s_lm1;
	
	std::unique_ptr<float[], deleterRR_struct> d_alm1(new float[SIZE_S_LM1]);
	cudaMallocManaged((void **) &d_alm1, SIZE_S_LM1*sizeof(float));
	alm1 = std::move(d_alm1);
	cudaMemcpy(alm1.get(), h_X.data(), SIZE_S_LM1 *sizeof(float),cudaMemcpyHostToDevice);

	this->m = m;
			
}


// initialize layer l
void Axon_u::init_al(const int m) { 
	const int SIZE_S_L = m * s_l;

	std::unique_ptr<float[], deleterRR_struct> d_al(new float[SIZE_S_L]);
	cudaMallocManaged((void **) &d_al, SIZE_S_L*sizeof(float));
	al = std::move(d_al);

	cudaMemset(al.get(), 0.f, SIZE_S_L*sizeof(float));

	this->m = m;
}

// for getting size dimensions
std::vector<int> Axon_u::getSizeDims() {
	std::vector<int> sizedimsvec = { s_lm1, s_l, m };
	return sizedimsvec;
}


// for getting Theta,b, and lth layer al, zl (after activation function applied)

std::unique_ptr<float[], deleterRR_struct> Axon_u::getTheta() {
	auto ptr = std::move(Theta);
	return ptr;
}

std::unique_ptr<float[],deleterRR_struct> Axon_u::getb() {
	auto ptr = std::move(b);
	return ptr;
}

std::unique_ptr<float[],deleterRR_struct> Axon_u::getalm1() {
	auto ptr = std::move(alm1);
	return ptr;
}


std::unique_ptr<float[],deleterRR_struct> Axon_u::getal() {
	auto ptr = std::move(al);
	return ptr;
}

/**
 *  @fn rightMul
 *  @class Axon_sh
 * 	@brief right multiplication
 * */
void Axon_u::rightMul() {
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

// destructor
Axon_u::~Axon_u() {}


