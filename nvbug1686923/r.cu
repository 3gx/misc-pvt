#include <cuda_runtime_api.h>
#include <iostream>
#include <cstdlib>

#include <time.h>

__global__ void just_launch(){ };

int main(int argc, char** argv){

        if (argc != 4){
                std::cout << "number_of_blocks number_of_threads cycles" << std::endl;
        };

        cudaError_t status;
        struct timespec start, stop;

        clock_gettime(CLOCK_MONOTONIC, &start);

        status = cudaFree(0);                   if (status != cudaSuccess){ std::cout << cudaGetErrorString(status) << std::endl; };

        for (int i = 0; i < atoi(argv[3]); ++i){
                just_launch<<<atoi(argv[1]), atoi(argv[2]), 0>>>();
        };

        status = cudaDeviceSynchronize();       if (status != cudaSuccess){ std::cout << cudaGetErrorString(status) << std::endl; };

        clock_gettime(CLOCK_MONOTONIC, &stop);
        double secs = (double)(stop.tv_sec - start.tv_sec) + (stop.tv_nsec/1000000.0 - start.tv_nsec/1000000.0)/1000.0;
        std::cout << "name,Duration" << std::endl;
        std::cout << "just_launch " <<  secs << std::endl;

        status = cudaDeviceReset();             if (status != cudaSuccess){ std::cout << cudaGetErrorString(status) << std::endl; };
};
