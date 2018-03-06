#include <cuda_runtime.h>
#include "GPUTiktok.h"

namespace util
{
    GPUTiktok::GPUTiktok() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    GPUTiktok::~GPUTiktok() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void GPUTiktok::tik() {
        cudaEventRecord(start_);
    }

    void GPUTiktok::tok() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
    }

    double GPUTiktok::toSeconds() {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_, stop_);
        return (double)milliseconds/1000.f;
    }

    double GPUTiktok::toMilliseconds() {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_, stop_);
        return (double)milliseconds;
    }

}
