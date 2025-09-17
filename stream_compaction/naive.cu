#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

# define blockSize 128

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void naiveKernScan(int n, int d, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index >= n) {
                return;
            }

            // Compute offset, 2^d.
            int offset = 1 << (d - 1);

            if (index >= offset) {
                odata[index] = idata[index] + idata[index - offset];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Create double buffers.
            int* dev_bufferA;
            int* dev_bufferB;

            // Allocate buffers.
            cudaMalloc((void**)&dev_bufferA, n * sizeof(int));
            checkCUDAError("cudaMalloc failed to create buffer A.");
            cudaMalloc((void**)&dev_bufferB, n * sizeof(int));
            checkCUDAError("cudaMalloc failed to create buffer B.");

            cudaMemcpy(dev_bufferA, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // TODO
            // For each kernel until log2N kernels do:
            for (int d = 1; d <= ilog2ceil(n); d++) {
                // Call naive scan kernel.
                naiveKernScan << <fullBlocksPerGrid, blockSize >> > (n, d, dev_bufferB, dev_bufferA);
                checkCUDAError("naiveKernScan failed.");

                // Swap buffers.
                std::swap(dev_bufferA, dev_bufferB);
            }
            cudaDeviceSynchronize();
            timer().endGpuTimer();

            // Memcpy invoked kern process to odata.
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_bufferA, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

            // Free data.
            cudaFree(dev_bufferA);
            cudaFree(dev_bufferB);
        }
    }
}
