# cuBlackDream (cuBD)
Machine Learning and Deep Learning in CUDA C++20 for CUDA 12.2, CUDNN 8.9.3 and above; GPU-accelerated (parallel programming) first.

- Parallel programming, parallel algorithms *only* as the aim to move all the computation to the GPU
- Fast File I/O with C++14 *binary* format (`std::ios::binary`)
- Multi-GPU ready (CUDA Unified Memory Management used exclusively)
- C++20 smart pointers

## Building the C++ project

Make a new Build directory from the "top" directory. You can name it anything but typically I like to name it "BuildGcc", or `Build<compiler type>` where compiler type is the compiler I'm using, such as GCC.
```
cuBlackDream$ mkdir BuildGcc
```
Then
```
cd BuildGcc
cmake ../Source
make
```
Then running `./Check` will run the unit tests suite. For example,

```
cuBlackDream/BuildGcc$ ./Check
```

## Creating and starting a virtual environment for Python 3

Create a directory for a virtual environment:

```
/cuBlackDream$ python3 -m venv ./venv/
```

Activate it:
```
/cuBlackDream$ source ./venv/bin/activate
```
You should see the prompt have a prefix `(venv)`.

Deactivate it:
```
deactivate
```

## Glossary for directory/Explanation of Folder Structure here  

- `src`  : source files, the actual code
- `examples` 
- `data` : sample datasets 
- `LICENSE`  
- `README.md`  

### Why the name `cuBlackDream`?  

> With artificial intelligence we are summoning the demon.  <cite>Elon Musk</cite>  cf. [WP, Oct. 24, 2014](https://www.washingtonpost.com/news/innovations/wp/2014/10/24/elon-musk-with-artificial-intelligence-we-are-summoning-the-demon/?utm_term=.3a9b517cdddf)

If so, then maybe this is the stuff of *black dreams.*  

## TensorFlow and Docker

Reference [Download a TensorFlow Docker image](https://www.tensorflow.org/install/docker)

Go to `Scripts/TensorFlowDocker` for a shell script that has the commands to run.

## `pytest`

You will need to run the unit tests for Python code from the "root" or "top" of this repository:

```
/cuBlackDream# pytest ./unit_tests/PyTorch
```
for example, in order to import modules from `CuISOX`.