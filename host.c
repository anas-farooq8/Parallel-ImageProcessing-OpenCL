#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 300
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

void print_devices() {
    // Get available platforms and devices
    cl_platform_id platforms[10];
    cl_device_id devices[10];
    cl_uint num_platforms, num_devices;

    // Get available platforms
    clGetPlatformIDs(10, platforms, &num_platforms);

    printf("Number of Platforms: %d\n", num_platforms);

    for (cl_uint i = 0; i < num_platforms; i++) {
        char platform_name[1024];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        printf("Platform %d: %s\n", i + 1, platform_name);

        // Get devices available on this platform
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 10, devices, &num_devices);

        for (cl_uint j = 0; j < num_devices; j++) {
            char device_name[1024];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
            printf("  Device %d: %s\n", j + 1, device_name);
        }
    }
}

int main() {
    // OpenCL variables
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem inputBuffer, outputBuffer;
    cl_int err;
    
    print_devices();

    // Specify the input and output image paths
    const char* inputPath = "/home/anasfarooq8/OpenCL-and-Docker/ISIC_2020_Test_Input/ISIC_0073502.jpg";
    const char* outputPath = "/home/anasfarooq8/OpenCL-and-Docker/output/ISIC_0073502_GrayScale.JPG";

    // Load the image
    int width, height, channels;
    unsigned char* image = stbi_load(inputPath, &width, &height, &channels, 0);
    if (!image) {
        printf("Failed to load image\n");
        return -1;
    }

    printf("\nImage loaded successfully. Width: %d, Height: %d, Channels: %d\n", width, height, channels);

    // Prepare a buffer for the grayscale output
    unsigned char* grayscale = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    // Get the specified platform and device
    err = clGetPlatformIDs(1, &platform_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error getting platform. %d\n", err);
        return -1;
    }
    err |= clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error getting device. %d\n", err);
        return -1;
    }

    // Determine device type and print the device name
    cl_device_type deviceType;
    char deviceName[1024];
    clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    printf("Running on device: %s\n", deviceName);

    // Create a context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating context.\n");
        return -1;
    }

    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating command queue.\n");
        return -1;
    }

    // Create the program
    const char *kernelSource = 
    "__kernel void grayscale(__global unsigned char* input, __global unsigned char* output, int width, int height, int channels) {"
    "    int x = get_global_id(0);"
    "    int y = get_global_id(1);"
    "    if (x < width && y < height) {"
    "        int index = (y * width + x) * channels;"
    "        float gray = 0.299f * input[index] + 0.587f * input[index + 1] + 0.114f * input[index + 2];"
    "        output[y * width + x] = (unsigned char)gray;"
    "    }"
    "}";

    // Create the program
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating program.\n");
        return -1;
    }

    // Build the program
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error building program.\n");
        return -1;
    }

    // Create the kernel
    kernel = clCreateKernel(program, "grayscale", &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating kernel.\n");
        return -1;
    }

    // Create memory buffers
    inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, width * height * channels * sizeof(unsigned char), image, NULL);
    outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(unsigned char), NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error creating memory buffers.\n");
        return -1;
    }

    // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &height);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &channels);
    if (err != CL_SUCCESS) {
        printf("Error setting kernel arguments: %d\n", err);
        return -1;
    }

    // Set global and local sizes
    // size_t globalSize[2] = { (size_t)width, (size_t)height };
    // Set global and local sizes
    size_t globalSize[2] = { (size_t)((width + 15) / 16) * 16, (size_t)((height + 15) / 16) * 16 };

    // Determine local size based on device type
    size_t localSize[2];
    if (deviceType == CL_DEVICE_TYPE_CPU) {
        // CPU: Use local size of 1x1
        localSize[0] = 1;
        localSize[1] = 1;
    } else {
        // GPU: Use the maximum work-group size supported
        size_t maxLocalSize, maxComputeUnits;
        maxLocalSize = maxComputeUnits = 0;
        clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxLocalSize, NULL);
        clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), &maxComputeUnits, NULL);
        // Max work group size                       256 => (Intel(R) UHD Graphics 620)
        localSize[0] = 16;
        localSize[1] = 16;
        printf("Max Work Group Size: %zu\n", maxLocalSize);
        printf("Max Compute Units: %zu\n", maxComputeUnits);
    }

    printf("Global Size X: %zu\n", globalSize[0]);
    printf("Global Size Y: %zu\n", globalSize[1]);

    printf("Local Size X: %zu\n", localSize[0]);
    printf("Local Size Y: %zu\n", localSize[1]);

    clock_t start = clock();

    // Execute the kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error executing kernel: %d\n", err);
        return -1;
    }

    clock_t end = clock();

    // Read the result
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, width * height * sizeof(unsigned char), grayscale, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error reading result: %d\n", err);
        return -1;
    }
    
    // Clean up
    if (inputBuffer)
        clReleaseMemObject(inputBuffer);
    if (outputBuffer)
        clReleaseMemObject(outputBuffer);
    if (kernel)
        clReleaseKernel(kernel);
    if (program)
        clReleaseProgram(program);
    if (queue)
        clReleaseCommandQueue(queue);
    if (context)
        clReleaseContext(context);

    // Save the Converted GrayScale Image
    stbi_write_jpg(outputPath, width, height, 1, grayscale, 100);
    free(grayscale);
    stbi_image_free(image);
    printf("Image Converted & Saved Successfully!\n");

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken for grayscale conversion: %f seconds\n", time_taken);

    return 0;
}
