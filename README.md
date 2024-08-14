# A independent C++ inference pipeline without DataFlow for ResNet18 Model with OpenSource library (DNNL)

## C++ Script Inference using oneDNN Library  
The repo contains Independent CPP inference with oneDNN Library for ResNet18 Model.

# DNNL Library ResNet18 Inference 
The repo contains the Independent C++ inference with ARM compute Library for [ResNet 18 onnx model](https://huggingface.co/frgfm/resnet18/blob/main/model.onnx). The Python inference is used for validation purpose.

## Machine Requirements:
- Processor Architecture: x86_64
- RAM: Minimum 8GB
- OS: Ubuntu 20.04 
- Storage: Minimum 64GB

## Prequisites
* G++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
* cmake version 3.29.3
* GNU Make 4.2.1
* [cnpy](https://github.com/rogersce/cnpy) 
* oneDNN Library [documentation](https://github.com/oneapi-src/oneAPI-samples/tree/master/Libraries/oneDNN)
* Python 3.8.10 (create a virtual environment)

## Install Prequisites

1. Build the oneDNN library by referring to the [documentation](https://oneapi-src.github.io/oneDNN/dev_guide_build.html)
2. Build the cnpy library by following the steps in [documentation](https://github.com/rogersce/cnpy?tab=readme-ov-file#installation)  

## Cloning the Repo 
Clone the repo using the following command  
```
git clone https://github.com/prachi-mcw/DNNL_Inference/tree/resnet_18_inference
cd dnnl_resnet18_inference
```  

## Setting Up the Environment

Before running the project, you need to activate the Python virtual environment.
```
source <PATH_TO_PYTHON_ENV>/bin/activate
pip install -r requirements.txt
```

## How to Run Python Inference (Used for Validation of C++ Inference Output)
1. Navigate to the project directory
```
    cd dnnl_resnet18_inference
```
2. Download input image 
```
    mkdir input 
    wget -O inputs/chainsaw.png https://stihlusa-images.imgix.net/Category/41/Teaser.png
```
3. Download the ResNet18 onnx model from hugging face repo 
```
    wget https://huggingface.co/frgfm/resnet18/resolve/main/model.onnx
```
4. Run the python script to dump weights from the onnx model 
```
    python model_dumper.py
```  
5. Run the python inference script to load image, preprocess and dump output files
```
    python inference.py <input_image_path>
```
Sample Usage
```
    python inference.py input/chainsaw.png
```

## How to Build and Run C++ Inference 
_Note: Execute the Python inference before running C++ Inference_
1. Navigate to the project directory
```
    cd dnnl_resnet18_inference
```
2. Build the cpp inference program
```
    g++ inference.cpp -ldnnl -lcnpy -lz -o inference
``` 
3. Run the program 
```
    ./inference
```

## Comparing Outputs
1. All the output files are stored in outputs/ folder, Manual comparison of files can be done using the compare.py file 
```
    python compare.py <file_1.npy> <file_2.npy>
```
Sample usage
```
    python compare.py outputs/cpp_gemm_output.npy outputs/py_output_gemm.npy
```
Sample output 
```
    $  python compare.py outputs/cpp_gemm_output.npy outputs/py_output_gemm.npy
    Files are identical upto 4 decimals
```
2. To compare the model's output use the output_validator file with last layer's output of cpp and python scripts
```
    python output_validator.py <cpp_output_file.npy> <python_output_file.npy> 
```
Sample usage
```
    python output_validator.py outputs/cpp_gemm_output.npy outputs/py_output_gemm.npy
```
Sample output
```
    $ python output_validator.py outputs/cpp_gemm_output.npy outputs/py_output_gemm.npy
    Files are identical upto 4 decimals
    Predicted Pytorch: chain saw
    Predicted C++: chain saw
```
