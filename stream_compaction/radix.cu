#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "radix.h"
#include "efficient.h"

# define blockSize 128

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // Compute inverse bits given dev_idata, dev_b, and dev_e.
        // dev_e is inverse of dev_b.
        __global__ void kernComputeInverseBits(int n, int bit, int* idata, int* b, int* e) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index >= n) {
                return;
            }

            // Get bit for dev_b via masking.
            int mask = 1 << bit;
            b[index] = (idata[index] & mask) ? 1 : 0;
            // e is inverse of b.
            e[index] = 1 - b[index];
        }

        // Compute each t using current index, f, and totalFalse value.
        __global__ void kernComputeT(int n, int* f, int* t, int totalFalse) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);

            if (index >= n) {
                return;
            }

            t[index] = index - f[index] + totalFalse;
        }

        // Scatter based on address d.
        __global__ void kernScatterRadix(int n, int* idata, int* odata, int* b, int* t, int* f) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n) return;

            if (b[index] == 0) {
                odata[f[index]] = idata[index];
            }
            else {
                odata[t[index]] = idata[index];
            }
        }

        /**
         * Implementation of parallel radix sort on GPU.
         */
        void radixSort(int n, int* odata, const int* idata) {

            // Create buffers.
            int* dev_idata;
            int* dev_odata;
            int* dev_b;
            int* dev_e;
            int* dev_f;
            int* dev_t;

            int totalFalse;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc failed to create buffer.");

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc failed to create buffer.");

            cudaMalloc((void**)&dev_b, n * sizeof(int));
            checkCUDAError("cudaMalloc failed to create buffer.");

            cudaMalloc((void**)&dev_e, n * sizeof(int));
            checkCUDAError("cudaMalloc failed to create buffer.");

            cudaMalloc((void**)&dev_f, n * sizeof(int));
            checkCUDAError("cudaMalloc failed to create buffer.");

            cudaMalloc((void**)&dev_t, n * sizeof(int));
            checkCUDAError("cudaMalloc failed to create buffer.");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // For loop through int size.
            for (int i = 0; i < sizeof(int) * 8; i++) {
                // Step 1: compute b and e
                kernComputeInverseBits << <fullBlocksPerGrid, blockSize >> > (n, i, dev_idata, dev_b, dev_e);
                cudaDeviceSynchronize();

                // Step 2: scan e -> f
                StreamCompaction::Efficient::scanDevice(n, dev_f, dev_e);
                cudaDeviceSynchronize();

                // Step 3: compute totalFalse
                int lastE, lastF;
                cudaMemcpy(&lastE, dev_e + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&lastF, dev_f + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
                int totalFalse = lastE + lastF;

                // Step 4: compute t
                kernComputeT << <fullBlocksPerGrid, blockSize >> > (n, dev_f, dev_t, totalFalse);
                cudaDeviceSynchronize();

                // Step 5: scatter values correctly
                kernScatterRadix << <fullBlocksPerGrid, blockSize >> > (n, dev_idata, dev_odata, dev_b, dev_t, dev_f);
                cudaDeviceSynchronize();

                // Swap for next iteration
                std::swap(dev_idata, dev_odata);
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_idata, n * sizeof(int), cudaMemcpyDeviceToHost);

            // Free buffers.
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_b);
            cudaFree(dev_e);
            cudaFree(dev_f);
            cudaFree(dev_t);
        }
    }
}
