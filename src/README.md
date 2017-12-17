# [`./Axon`](https://github.com/ernestyalumni/cuBlackDream/tree/master/src/Axon)    

## on the *move constructor*  

In [`Axon.h`](https://github.com/ernestyalumni/cuBlackDream/blob/master/src/Axon/Axon.h)    

```  
class Axon
{
	// protected chosen instead of private because members of derived class can access protected members, but not private
	protected:
		// size dims. of Theta,b - "weights" and bias
		int l; // lth layer, l=1,2,...L for L total number of layers
		int s_lm1; // size dim. of l-1th layer
		int s_l;	// size dim. of lth layer

		int m; // number of examples, m

		// this is used to calculate if have enough threads
		int MAX_SIZE_1DARR; // maximum device grid size in x-dimension
		int MAX_THREADBLOCK; // maximum number of threads in a (single thread) block

		// custom deleter as a STRUCT for cublasHandle 
		struct del_cublasHandle_struct {
			void operator()(cublasHandle_t* ptr) { cublasDestroy(*ptr); }
		};

		// members 
		std::unique_ptr<float[], deleterRR_struct> Theta;
		std::unique_ptr<float[], deleterRR_struct> b;

		std::shared_ptr<float> alm1;
		std::shared_ptr<float> al;

	public:
		// Constructor
		Axon(const int s_lm1, const int s_l, const int idx_device=0);

		// Move Constructor
		Axon( Axon &&); 
		
		// operator overload assignment = 
		Axon &operator=(Axon &&);
		
		// member functions
		void load_from_hvec(std::vector<float>&,std::vector<float>& );
		void load_from_d(std::vector<float>&, std::vector<float>& );
		void load_from_hXvec(std::vector<float>&, const int );
		void load_alm1_from_ptr(std::shared_ptr<float> &);

		void move2al_from_ptr(std::shared_ptr<float> & ptr_sh_output_layer) ;
		void move2alm1_from_ptr(std::shared_ptr<float> & ptr_sh_input_layer) ;

		void init_al(const int);

		std::vector<int> getSizeDims();
		
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
		std::shared_ptr<float> getal();

		virtual void rightMul(); 
		virtual void addb(const int M_x=128);

		// destructor
		~Axon();			
};
```  

inherited class `Axon_act` from base class `Axon`, in [`Axon.h`](https://github.com/ernestyalumni/cuBlackDream/blob/master/src/Axon/Axon.h)  

```  
class Axon_act : public Axon  
{
	protected:  
		int idx_actf;

		std::unique_ptr<float[], deleterRR_struct> zl;
		std::unique_ptr<float[], deleterRR_struct> Dpsil; 

	public:
		// Constructor
		Axon_act(const int s_lm1, const int s_l, const int idx_actf, const int idx_device=0);

		// Move Constructor
		Axon_act( Axon_act &&); 
		
		// operator overload assignment = 
		Axon_act &operator=(Axon_act &&);

		void init_zlal(const int);

		void move2Dpsil_from_ptr(std::unique_ptr<float[], deleterRR_struct> & ) ;

		std::unique_ptr<float[],deleterRR_struct> getzl();
		std::unique_ptr<float[],deleterRR_struct> getDpsil();

		void rightMul(); 

		void addb(const int M_x);
		void actf( const int M_x, const int N_x=0); 

		/* ========== partial derivatives with respect to z^l of psi^l(z^l) ========== */
		void do_Dpsi( const int M_x, const int N_x=0); 		
};
```  

*abridged version*

```  
class Axon
{
	protected:
		// size dims. of Theta,b - "weights" and bias
		int l; // lth layer, l=1,2,...L for L total number of layers
		int s_lm1; // size dim. of l-1th layer
		int s_l;	// size dim. of lth layer

		int m; // number of examples, m

		// this is used to calculate if have enough threads
		int MAX_SIZE_1DARR; // maximum device grid size in x-dimension
		int MAX_THREADBLOCK; // maximum number of threads in a (single thread) block

		// members 
		std::unique_ptr<float[], deleterRR_struct> Theta;
		std::unique_ptr<float[], deleterRR_struct> b;

		std::shared_ptr<float> alm1;
		std::shared_ptr<float> al;

	public:
		// Constructor
		Axon(const int s_lm1, const int s_l, const int idx_device=0);

		// Move Constructor
		Axon( Axon &&); 
		
		// operator overload assignment = 
		Axon &operator=(Axon &&);
		
		// member functions
		void move2Theta_from_ptr(std::unique_ptr<float[], deleterRR_struct> & ) ;
		void move2b_from_ptr(std::unique_ptr<float[], deleterRR_struct> & ) ;

		/**
		 * @fn Axon::getalm1
		 * @details we don't use std::move because we don't want to change (move) 
		 * 	ownership of the pointer (and the memory it points to) because we're 
		 *  dealing with a shared_ptr 
		 * */
		std::shared_ptr<float> getalm1();
		std::shared_ptr<float> getal();

		virtual void rightMul(); 
		virtual void addb(const int M_x=128);

		// destructor
		~Axon();			
};
```  


inherited class `Axon_act` from base class `Axon`, in [`Axon.h`](https://github.com/ernestyalumni/cuBlackDream/blob/master/src/Axon/Axon.h)  

```  
class Axon_act : public Axon  
{
	protected:  
		int idx_actf;

		std::unique_ptr<float[], deleterRR_struct> zl;
		std::unique_ptr<float[], deleterRR_struct> Dpsil; 

	public:
		// Constructor
		Axon_act(const int s_lm1, const int s_l, const int idx_actf, const int idx_device=0);

		// Move Constructor
		Axon_act( Axon_act &&); 
		
		// operator overload assignment = 
		Axon_act &operator=(Axon_act &&);

		void init_zlal(const int);

		void move2Dpsil_from_ptr(std::unique_ptr<float[], deleterRR_struct> & ) ;

		std::unique_ptr<float[],deleterRR_struct> getzl();
		std::unique_ptr<float[],deleterRR_struct> getDpsil();

		void rightMul(); 

		void addb(const int M_x);
		void actf( const int M_x, const int N_x=0); 

		/* ========== partial derivatives with respect to z^l of psi^l(z^l) ========== */
		void do_Dpsi( const int M_x, const int N_x=0); 		
};
```  


*abridged version*
```  
class Axon_act : public Axon  
{
	protected:  
		std::unique_ptr<float[], deleterRR_struct> zl;
		std::unique_ptr<float[], deleterRR_struct> Dpsil; 

	public:
		// Constructor
		Axon_act(const int s_lm1, const int s_l, const int idx_actf, const int idx_device=0);

		// Move Constructor
		Axon_act( Axon_act &&); 
		
		// operator overload assignment = 
		Axon_act &operator=(Axon_act &&);

		void move2Dpsil_from_ptr(std::unique_ptr<float[], deleterRR_struct> & ) ;

		std::unique_ptr<float[],deleterRR_struct> getzl();
		std::unique_ptr<float[],deleterRR_struct> getDpsil();

		void rightMul(); 

		void addb(const int M_x);
		void actf( const int M_x, const int N_x=0); 

		/* ========== partial derivatives with respect to z^l of psi^l(z^l) ========== */
		void do_Dpsi( const int M_x, const int N_x=0); 		
};
```  

and then in [`Axon.cu`](https://github.com/ernestyalumni/cuBlackDream/blob/master/src/Axon/Axon.cu):  

```  
// Move Constructor
Axon::Axon(Axon&& old_axon) : Theta(std::move(old_axon.Theta)), b(std::move(old_axon.b))
{
	s_lm1 = old_axon.s_lm1;
	s_l = old_axon.s_l;
	m = old_axon.m;

	MAX_SIZE_1DARR = old_axon.MAX_SIZE_1DARR ;  
	MAX_THREADBLOCK = old_axon.MAX_THREADBLOCK;
	
	l = old_axon.l; // lth layer
	
	alm1 = std::move( old_axon.alm1 );
	al = std::move( old_axon.al );	
}

// operator overload assignment = 
Axon & Axon::operator=(Axon && old_axon) {
	s_lm1 = old_axon.s_lm1;
	s_l = old_axon.s_l;
	m = old_axon.m;

	MAX_SIZE_1DARR = old_axon.MAX_SIZE_1DARR ;  
	MAX_THREADBLOCK = old_axon.MAX_THREADBLOCK;

	l = old_axon.l; // lth layer

	// shared_ptrs moved
	alm1 = std::move( old_axon.alm1 );
	al = std::move( old_axon.al );	

	// unique_ptrs moved
	Theta = std::move(old_axon.Theta);
	b = std::move( old_axon.b );

	return *this;
}
```  

and then, for the inherited classes, in [`Axon.cu`](https://github.com/ernestyalumni/cuBlackDream/blob/master/src/Axon/Axon.cu):  

```  
// Move Constructor
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

```  

*abridged versions*
```  
// Move Constructor
Axon::Axon(Axon&& old_axon) : Theta(std::move(old_axon.Theta)), 
								b(std::move(old_axon.b))
{
	s_lm1 = old_axon.s_lm1;
	s_l = old_axon.s_l;
	m = old_axon.m;
	l = old_axon.l; // lth layer

	MAX_SIZE_1DARR = old_axon.MAX_SIZE_1DARR ;  
	MAX_THREADBLOCK = old_axon.MAX_THREADBLOCK;
	
	alm1 = std::move( old_axon.alm1 );
	al = std::move( old_axon.al );	
}

// operator overload assignment = 
Axon & Axon::operator=(Axon && old_axon) {
	s_lm1 = old_axon.s_lm1;
	s_l = old_axon.s_l;
	m = old_axon.m;
	l = old_axon.l; // lth layer

	MAX_SIZE_1DARR = old_axon.MAX_SIZE_1DARR ;  
	MAX_THREADBLOCK = old_axon.MAX_THREADBLOCK;

	// shared_ptrs moved
	alm1 = std::move( old_axon.alm1 );
	al = std::move( old_axon.al );	

	// unique_ptrs moved
	Theta = std::move(old_axon.Theta);
	b = std::move( old_axon.b );

	return *this;
}
```  

```  
// Move Constructor
Axon_act::Axon_act(Axon_act&& old_axon) 
	: 	Axon(std::move(old_axon)), 

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
```  



and it's actually used here, with [`Feedfwd.cu`](https://github.com/ernestyalumni/cuBlackDream/blob/master/src/Feedfwd/Feedfwd.cu) (look at when `.push_back` is used):  

```  
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
		Axons.push_back( std::move( Axon(s_lm1,s_l,idx_device) ));
	}	

	// get maximum grid dimension on the device, numbered idx_device (usually 0th device GPU)
	MAX_SIZE_1DARR = get_max_device_array_size1d(idx_device);
	
}  

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
		Axons.push_back( std::move( Axon_act(s_lm1,s_l,idx_actf, idx_device) ) );
	}	
	
	// get maximum grid dimension on the device, numbered idx_device (usually 0th device GPU)
	MAX_SIZE_1DARR = get_max_device_array_size1d(idx_device);
	
}
```  

Interesting to note that the `nvcc` GPU compiler had to see the `std::unique_ptr` custom destructor globally:  

```  
/* =============== custom deleters =============== */

/* custom deleter as a struct */ 
struct deleterRR_struct
{
	void operator()(float* ptr) const 
	{
		cudaFree(ptr);
	}
};

```

cf. [Move with vector::push_back ; stackoverflow](https://stackoverflow.com/questions/11572669/move-with-vectorpush-back)

"using `push_back(x)` would create a copy of the object, while `push_back(move(x))` would tell `push_back()` that it may "steal" the contents of `x`, leaving `x` in an unusable and undefined state.

Consider if you had a vector of lists (`std::vector<std::list<int> >`) and you wanted to push a list containing 100,000 elements. Without `move()`, the entire list structure and all 100,000 elements will be copied. With `move()`, some pointers and other small bits of data get shuffled around, and that's about it. This will be lots faster, and will require less overall memory consumption."  

