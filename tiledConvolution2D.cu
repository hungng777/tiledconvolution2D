/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include "tiledConvolution2D.hpp"


#define     FILTER_RADIUS   2
#define     IN_TILE_DIM     32
#define     OUT_TILE_DIM    ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))

// Convolution filter
__constant__    float   F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

// Gaussian Blur filter values
float F_h[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1] = {
    {0.003, 0.014, 0.022, 0.014, 0.003},
    {0.014, 0.061, 0.100, 0.061, 0.014},
    {0.022, 0.100, 0.165, 0.100, 0.022},
    {0.014, 0.061, 0.100, 0.061, 0.014},
    {0.003, 0.014, 0.022, 0.014, 0.003}
};

/*
float F_h[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1] = {
    {0.04, 0.04, 0.04, 0.04, 0.04},
    {0.04, 0.04, 0.04, 0.04, 0.04},
    {0.04, 0.04, 0.04, 0.04, 0.04},
    {0.04, 0.04, 0.04, 0.04, 0.04},
    {0.04, 0.04, 0.04, 0.04, 0.04}
};
*/
/*
float F_h[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1] = {
    {1, 4, 6, 4, 1},
    {4, 16, 24, 16, 4},
    {6, 24, 36, 24, 16},
    {4, 16, 24, 16, 4},
    {1, 4, 6, 4, 1}
};
*/


/*
 * CUDA Kernel Device code
 *
 */
__device__ void convolution_tiled_2D_const_mem(float *N, float *P, int width, int height)
{
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    // Shared memory for input tile 
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];

    // Load input tile 
    if (col >= 0 && col < width && row >= 0 && row < height) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Ensure the whole input tile is loaded by all threads before starting the convolution calculations
    __syncthreads();

    // Calculating output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    // Turning off the threads at the edges of the block
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0;

            // Carry out the convolution calculations
            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                    Pvalue += F[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];
                }
            }

            // Save the convolution result into the P result array
            P[row * width + col] = Pvalue;
        }
    }
}

__global__ void convolution_tiled_2D_const_mem_kernel(float *N_r, float *N_g, float *N_b, 
                                                      float *P_r, float *P_g, float *P_b,
                                                      int width, int height)
{
    convolution_tiled_2D_const_mem(N_r, P_r, width, height);    // handle the red (R) color
    convolution_tiled_2D_const_mem(N_g, P_g, width, height);    // handle the green (G) color
    convolution_tiled_2D_const_mem(N_b, P_b, width, height);    // handle the blue (B) color
}

__host__ void allocateDeviceMemory(int rows, int columns)
{
    // Allocate device constant symbols for convolution filter
    cudaMemcpyToSymbol(F, &F_h, sizeof(F_h), 0, cudaMemcpyHostToDevice);
}

__host__ void executeKernel(float *N_r, float *N_g, float *N_b, float *P_r, float *P_g, float *P_b, int rows, int columns)
{
    cout << "Executing kernel\n";

    //Launch the convolution Kernel
    dim3 blocksPerGrid((columns + OUT_TILE_DIM - 1)/ OUT_TILE_DIM, (rows + OUT_TILE_DIM - 1)/ OUT_TILE_DIM, 1);
    dim3 threadsPerBlock(IN_TILE_DIM, IN_TILE_DIM, 1);

    convolution_tiled_2D_const_mem_kernel<<<blocksPerGrid, threadsPerBlock>>>(N_r, N_g, N_b, P_r, P_g, P_b, columns, rows);
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();
}

// Reset the device and exit
__host__ void cleanUpDevice()
{
    cout << "Cleaning CUDA device\n";
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ std::tuple<std::string, std::string, std::string, int> parseCommandLineArguments(int argc, char *argv[])
{
    cout << "Parsing CLI arguments\n";
    int threadsPerBlock = 256;
    std::string inputImage = "sloth.png";
    std::string outputImage = "grey-sloth.png";
    std::string currentPartId = "test";

    for (int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if (option.compare("-i") == 0)
        {
            inputImage = value;
        }
        else if (option.compare("-o") == 0)
        {
            outputImage = value;
        }
        else if (option.compare("-t") == 0)
        {
            threadsPerBlock = atoi(value.c_str());
        }
        else if (option.compare("-p") == 0)
        {
            currentPartId = value;
        }
    }
    cout << "inputImage: " << inputImage << " outputImage: " << outputImage << " currentPartId: " << currentPartId << " threadsPerBlock: " << threadsPerBlock << "\n";
    return {inputImage, outputImage, currentPartId, threadsPerBlock};
}

__host__ std::tuple<int, int, float *, float *, float *> readImageFromFile(std::string inputFile)
{
    cout << "Reading Image From File\n";
    Mat img = imread(inputFile, IMREAD_COLOR);
    
    const int rows = img.rows;
    const int columns = img.cols;
    size_t size = sizeof(float) * rows * columns;

    cout << "Rows: " << rows << " Columns: " << columns << "\n";

    float *r, *g, *b;
    cudaMallocManaged(&r, size);
    cudaMallocManaged(&g, size);
    cudaMallocManaged(&b, size);
    
    for(int y = 0; y < rows; ++y)
    {
        for(int x = 0; x < columns; ++x)
        {
            Vec3b rgb = img.at<Vec3b>(y, x);
            r[y*rows+x] = rgb.val[0];
            g[y*rows+x]= rgb.val[1];
            b[y*rows+x] = rgb.val[2];
        }
    }

    return {rows, columns, r, g, b};
}


int main(int argc, char *argv[])
{
    std::tuple<std::string, std::string, std::string, int> parsedCommandLineArgsTuple = parseCommandLineArguments(argc, argv);
    std::string inputImage = get<0>(parsedCommandLineArgsTuple);
    std::string outputImage = get<1>(parsedCommandLineArgsTuple);
    std::string currentPartId = get<2>(parsedCommandLineArgsTuple);
    int threadsPerBlock = get<3>(parsedCommandLineArgsTuple);
    try 
    {
        auto[rows, columns, N_r, N_g, N_b] = readImageFromFile(inputImage);

        size_t size = sizeof(float) * rows * columns;
        float *P_r, *P_g, *P_b;

        // Allocate unified memory for result RGB arrays
        cudaMallocManaged(&P_r, size);
        cudaMallocManaged(&P_g, size);
        cudaMallocManaged(&P_b, size);

        allocateDeviceMemory(rows, columns);

        executeKernel(N_r, N_g, N_b, P_r, P_g, P_b, rows, columns);

        Mat colorImage(rows, columns, CV_8UC3);
        vector<int> compression_params;
        compression_params.push_back(IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);

        for(int y = 0; y < rows; ++y)
        {
            for(int x = 0; x < columns; ++x)
            {
                colorImage.at<Vec3b>(y,x) = Vec3b(P_r[y*rows+x], P_g[y*rows+x], P_b[y*rows+x]);
            }
        }

        imwrite(outputImage, colorImage, compression_params);

        cleanUpDevice();
    }
    catch (Exception &error_)
    {
        cout << "Caught exception: " << error_.what() << endl;
        return 1;
    }
    return 0;
}