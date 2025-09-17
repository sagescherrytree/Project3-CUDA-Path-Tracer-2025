#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        // Scan that solely runs on device and does not copy from host to device to mitigate runtime issues.
        void scanDevice(int n, int* odata, const int* idata);

        int compact(int n, int *odata, const int *idata);
    }
}
