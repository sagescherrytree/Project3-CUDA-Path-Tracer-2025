#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

# define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // Upsweep on kernel.
        __global__ void kernUpsweep(int n, int stride, int* idata) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);

            if (k >= n) {
                return;
            }

            int index = k * stride + stride - 1;
            int left = index - (stride >> 1);

            idata[index] += idata[left];
        }

        // Downsweep on kernel.
        __global__ void kernDownsweep(int n, int stride, int* idata) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);

            if (k >= n) {
                return;
            }

            int index = k * stride + stride - 1;
            int left = index - (stride >> 1);

            int temp = idata[left];
            idata[left] = idata[index];
            idata[index] += temp;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            // One buffer for in place scan.
            int* dev_buffer;

            // Set new n w/ log2ceil function.
            int round_n = 1 << ilog2ceil(n);

            // Allocate buffer.
            cudaMalloc((void**)&dev_buffer, round_n * sizeof(int));
            checkCUDAError("cudaMalloc failed to create buffer.");

            cudaMemset(dev_buffer, 0, round_n * sizeof(int));
            cudaMemcpy(dev_buffer, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO

            // Upsweep.
            for (int d = 0; d < ilog2ceil(n); d++) {
                // Optimisation to get faster runtime.
                int stride = 1 << (d + 1);
                int nodes = round_n / stride;
                if (nodes == 0) {
                    break;
                }
                // Call fullBlocksPerGrid with new node size.
                dim3 fullBlocksPerGrid((nodes + blockSize - 1) / blockSize);
                // Call upsweep kern function w/ rounded n val.
                kernUpsweep << < fullBlocksPerGrid, blockSize >> > (nodes, stride, dev_buffer);
                checkCUDAError("kernUpsweep failed.");
                cudaDeviceSynchronize();
            }

            // Downsweep. 
            // Set (round_n - 1) val in dev_buffer = 0.
            cudaMemset(dev_buffer + (round_n - 1), 0, sizeof(int));
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                // Optimisation to get faster runtime.
                int stride = 1 << (d + 1);
                int nodes = round_n / stride;
                if (nodes == 0) {
                    break;
                }
                // Call fullBlocksPerGrid with new node size.
                dim3 fullBlocksPerGrid((nodes + blockSize - 1) / blockSize);
                // Call downsweep kern func.
                kernDownsweep << < fullBlocksPerGrid, blockSize >> > (nodes, stride, dev_buffer);
                checkCUDAError("kernDownsweep failed.");
                cudaDeviceSynchronize();
            }
            timer().endGpuTimer();

            // Copy data back to host from dev_buffer.
            cudaMemcpy(odata, dev_buffer, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_buffer);
        }

        // Device-only scan to mitigate timer issues, for radix sort.
        void scanDevice(int n, int* dev_odata, const int* dev_idata) {
            // dev_idata: input device array
            // dev_odata: output device array (pre-allocated)

            int* dev_buffer;
            int round_n = 1 << ilog2ceil(n);

            cudaMalloc((void**)&dev_buffer, round_n * sizeof(int));
            cudaMemset(dev_buffer, 0, round_n * sizeof(int));
            cudaMemcpy(dev_buffer, dev_idata, n * sizeof(int), cudaMemcpyDeviceToDevice);

            for (int d = 0; d < ilog2ceil(n); d++) {
                // Optimisation to get faster runtime.
                int stride = 1 << (d + 1);
                int nodes = round_n / stride;

                // Call fullBlocksPerGrid with new node size.
                dim3 fullBlocksPerGrid((nodes + blockSize - 1) / blockSize);
                // Call upsweep kern function w/ rounded n val.
                kernUpsweep << < fullBlocksPerGrid, blockSize >> > (nodes, stride, dev_buffer);
                checkCUDAError("kernUpsweep failed.");
                cudaDeviceSynchronize();
            }

            // Downsweep. 
            // Set (round_n - 1) val in dev_buffer = 0.
            cudaMemset(dev_buffer + (round_n - 1), 0, sizeof(int));
            for (int d = ilog2ceil(n) - 1; d >= 0; d--) {
                // Optimisation to get faster runtime.
                int stride = 1 << (d + 1);
                int nodes = round_n / stride;

                // Call fullBlocksPerGrid with new node size.
                dim3 fullBlocksPerGrid((nodes + blockSize - 1) / blockSize);
                // Call downsweep kern func.
                kernDownsweep << < fullBlocksPerGrid, blockSize >> > (nodes, stride, dev_buffer);
                checkCUDAError("kernDownsweep failed.");
                cudaDeviceSynchronize();
            }

            // Copy result into output (device to device)
            cudaMemcpy(dev_odata, dev_buffer, n * sizeof(int), cudaMemcpyDeviceToDevice);

            cudaFree(dev_buffer);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int* odata, const int* idata) {

            int* dev_idata;
            int* dev_odata;
            int* dev_flagged;
            int* dev_scanned;

            int round_n = 1 << ilog2ceil(n);

            // Create buffers.
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc failed to create buffer.");

            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc failed to create buffer.");

            cudaMalloc((void**)&dev_flagged, n * sizeof(int));
            checkCUDAError("cudaMalloc failed to create flagged array.");

            cudaMalloc((void**)&dev_scanned, n * sizeof(int));
            checkCUDAError("cudaMalloc failed to create scanned array.");

            // Copy data from idata input to dev_idata.
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // TODO
            // Step 1: Flag data that belongs w/ temp array flagged.
            // Call kernMapToBoolean w/ dev_flagged array.
            StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, dev_flagged, dev_idata);
            cudaDeviceSynchronize();

            // Step 2: Call efficient scan function, output = dev_scanned, input = dev_flagged.
            scanDevice(n, dev_scanned, dev_flagged);
            cudaDeviceSynchronize();

            // Step 3: Scatter.
            StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, dev_odata, dev_idata, dev_flagged, dev_scanned);
            cudaDeviceSynchronize();

            timer().endGpuTimer();

            int lastFlag = 0;
            int lastScan = 0;

            cudaMemcpy(&lastFlag, dev_flagged + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastScan, dev_scanned + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);

            int counter = lastScan + lastFlag;

            // Copy data from dev_odata to odata.
            cudaMemcpy(odata, dev_odata, counter * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_flagged);
            cudaFree(dev_scanned);

            return counter;
        }
    }
}
