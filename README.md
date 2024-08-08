# A independent C++ inference pipeline without DataFlow for single Conv2d layer with OpenSource library (DNNL)

## C++ Script Inference using oneDNN Library  
The repo contains Independent CPP inference with oneDNN Library for Convolution layer.

## Machine Requirements:
- Processor Architecture: x86
- RAM: Minimum 8GB
- OS: Ubuntu 20.04 
- Storage: Minimum 64GB

## Prequisites:
- oneDNN Library [documentation](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneDNN)
- cmake version 3.29.3
- g++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
- Python 3.8.19 (create virtual enviroment)

## Setting Up the Environment

Before running the project, you need to activate the Python virtual environment.

### Activate the Virtual Environment

```
source myenv/bin/activate
```

### Installing Dependencies

After activating the environment, install the necessary dependencies:

```
pip install -r requirements.txt
```

# Install Prequisites

 Build the oneDNN library by referring to the [documentation](https://oneapi-src.github.io/oneDNN/dev_guide_build.html)

## Cloning the Repo 
Clone the repo using the following command  
```
git clone https://github.com/prachi-mcw/dnnl_conv_inference.git
cd dnnl_conv_inference
```  

## How  Run Python Convolution Layer Inference (For Verification Of CPP Output)


Run the python script to get convolution layer output and dump it to output file
```
python conv_inference.py
```

Note: Execute the Python Inference before running this section
## To Build and Run the CPP Convolution Layer Inference 
  
1. Compile the C++ script 
```
g++ conv2d.cpp -ldnnl -lcurl -lz -o mnist_conv -I<PATH_ON_ONEDNN> 
```
Replace <PATH_ON_ONEDNN> with the actual path of the oneDNN library folder

2. Run the executable file  created by above command 
```
./mnist_conv
```

## Comparing Outputs 
- To Compare the outputs of c++ inference pipeline and python script use compare.py file 

- output are stored in .bin files separately for both c++ and python scripts 
```
python compare.py
```










