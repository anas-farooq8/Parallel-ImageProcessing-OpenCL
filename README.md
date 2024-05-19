
# Optimal Image Preprocessing using OpenCL
## Analysis Towards Melanoma Detection
This project aims to provide hands-on experience with parallel and distributed computing using OpenCL by converting a dataset of colored images of skin lesions into grayscale images. Specifically, it focuses on processing images from "The ISIC 2020 Challenge Dataset" to aid in early skin cancer detection and diagnosis.

## Objective
The objective of this project is to convert the ISIC 2020 Challenge Dataset, which consists of dermoscopic images of skin lesions, into grayscale images using OpenCL parallel computing. By converting these images to grayscale, the complexity is reduced, facilitating tasks such as lesion segmentation and aiding in computational efficiency.

## The ISIC 2020 Challenge Dataset
The ISIC dataset is a crucial resource for dermatology and medical image analysis, offering a diverse collection of high-quality skin lesion images. It supports the development of diagnostic systems, segmentation algorithms, and deep learning models for automated lesion detection and classification. Converting these images to grayscale simplifies image representation and aids in analysis and processing.

## Download the Dataset
Test
Set available at containing 10,982 JPEG images of different sizes:
https://isic-challenge-data.s3.amazonaws.com/2020/ISIC_2020_Test_JPEG.zip

##  Problem Description
You are provided with a dataset of colored images of skin lesions in JPEG format. The task is to convert these colored images to grayscale images using OpenCL parallel computing.

# Installation

## Steps to Follow:

* Install WSL (Windows Subsystem for Linux):
    - Open PowerShell or terminal as administrator and run:
    ```bash
    wsl --install
    ```

*  [Clone the repository](https://github.com/anas-farooq8/Parallel-ImageProcessing-OpenCL.git) to your local machine.

* Download the stb_image and stb_image_write .h files from the github link.
    [Click here.](https://github.com/nothings/stb/tree/master)

*  Navigate to the clonned directory; paste the .h files; update package list and install necessary packages:
    ```bash
    cd  Parallel-ImageProcessing-OpenCL
    sudo apt-get update && \
    sudo apt-get install -y pocl-opencl-icd ocl-icd-opencl-dev gcc clinfo
    ```

* Check available OpenCL devices:
    ```bash
    clinfo
    ```

* If your Intel integrated GPU doesn't appear, run:
    ```bash
    sudo apt install intel-opencl-icd
    ```

## Compilation & Execution
`gcc host.c -o host -lm -lOpenCL`

`./host`



### Additional Notes: My graphics card (Intel(R) UHD Graphics 620)
## Additional Notes

### 1. Change the Paths in the Code
Update the paths for the input and output images in `host.c`:

```c
// Specify the input and output image paths
const char* inputPath = "";
const char* outputPath = "";
```

### 2. Select the desired Platform and Device Id
```c
// Select the Platform Id
err = clGetPlatformIDs(1, &platform_id, NULL);

// Select the Device Id
err |= clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
```

### 3. Experiment with the local size
```c
// Select the local group size according to your gpu hardware.
    // Max work group size                       256 => (Intel(R) UHD Graphics 620)
    localSize[0] = 16;
    localSize[1] = 16;
```

### Sample Input & output
![ISIC_9498081](https://github.com/anas-farooq8/Parallel-ImageProcessing-OpenCL/assets/150327092/bb9f672e-707d-43ae-86ee-f9bab3b05dac)

![ISIC_9498081_GrayScale](https://github.com/anas-farooq8/Parallel-ImageProcessing-OpenCL/assets/150327092/93ed6567-11ce-42ba-848b-67f1a27367a6)
