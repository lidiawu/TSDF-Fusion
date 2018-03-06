#pragma once
#include <cuda.h>
#include <driver_types.h>

namespace util
{

    class GPUTiktok {
    private:

        cudaEvent_t start_, stop_;

    public:

        GPUTiktok();

        ~GPUTiktok();

        void tik();

        void tok();

        double toSeconds();

        double toMilliseconds();

    };

}

