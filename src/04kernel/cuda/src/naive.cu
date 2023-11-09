#include "common.h"
#include "kernel/cuda/naive.cuh"
#include "kernel/cuda/threads_distributer.cuh"
#include <chrono>
#include <thrust/device_vector.h>

__global__ static void sigmoidKernel(float *out, const float *in, const int N) {
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N) {
        out[tid] = 1.0 / (1.0 + expf(-in[tid]));
    }
}

struct SigmoidFunctor {
    __device__ __host__ __forceinline__ float operator()(float x) const {
        return 1.0 / (1.0 + expf(-x));
    }
};

struct SigmoidFactory {
    __device__ __host__ __forceinline__ SigmoidFunctor operator()() const {
        return SigmoidFunctor();
    }
};

namespace refactor::kernel::cuda {

    template<class T, int N>
    struct Packed {
        using Phantom = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;

        union {
            T elem[N];
            Phantom _phantom;
        };

        __device__ Packed() = default;
    };

    template<class T>
    constexpr size_t nPack() {
        static_assert(sizeof(T) > 1);
        constexpr size_t AccessGrainBits = 128;
        return AccessGrainBits / 8 / sizeof(T);
    }

    template<class T, class U, class... Args>
    constexpr size_t nPack() {
        return std::min(nPack<T>(), nPack<U, Args...>());
    }

    template<size_t>
    constexpr bool isAlignedForPack() {
        return true;
    }

    template<size_t N, class T, class... Others>
    constexpr bool isAlignedForPack(T const *ptr, Others const *...others) {
        return reinterpret_cast<uintptr_t>(ptr) % sizeof(Packed<T, N>) == 0 &&
               isAlignedForPack<N>(others...);
    }

    template<int N, class FunctorT> class UseApply2 {
        struct A {};
        struct B {};

        template<class C> static A test(decltype(&C::Apply2));
        template<class C> static B test(...);

    public:
        constexpr static bool value =
            N % 2 == 0 &&
            std::is_same_v<decltype(test<FunctorT>(0)), A>;
    };

    template<class Ans, class FunctorT, int N, class... Args>
    __device__ typename std::enable_if_t<UseApply2<N, FunctorT>::value, Packed<Ans, N>>
    applyPack(FunctorT const &functor, Packed<Args, N> const... args) {
        Packed<Ans, N> ans;
#pragma unroll
        for (int j = 0; j < N; j += 2) { functor.Apply2(ans.elem + j, (args.elem + j)...); }
        return ans;
    }

    template<class Ans, class FunctorT, int N, class... Args>
    __device__ typename std::enable_if_t<!UseApply2<N, FunctorT>::value, Packed<Ans, N>>
    applyPack(const FunctorT &functor, Packed<Args, N> const... args) {
        Packed<Ans, N> ans;
#pragma unroll
        for (int j = 0; j < N; ++j) { ans.elem[j] = functor((args.elem[j])...); }
        return ans;
    }

    template<int N, class Ans, class FactoryT, class... Args>
    __global__ void applyGeneric(
        FactoryT factory,
        int64_t n, int64_t nPack,
        Ans *ans, Args const *...args) {
        auto functor = factory();
        size_t tid = blockIdx.x * blockDim.x + threadIdx.x,
               tail = tid + nPack * N,
               step = blockDim.x * gridDim.x;
        for (; tid < nPack; tid += step) {
            reinterpret_cast<Packed<Ans, N> *>(ans)[tid] = applyPack<Ans>(functor, (reinterpret_cast<Packed<Args, N> const *>(args)[tid])...);
        }
        if (tail < n) {
            ans[tail] = functor((args[tail])...);
        }
    }

    template<class FactoryT, class Ans, class... Args>
    cudaError_t launch(ThreadsDistributer const &distributer,
                       FactoryT factory,
                       int64_t n, Ans *ans, Args const *...args,
                       cudaStream_t stream) {
        constexpr auto NPack = nPack<Ans, Args...>();
        if (isAlignedForPack<NPack>(ans, args...)) {
            int64_t const nPack = n / NPack;
            auto [grid, block] = distributer(nPack);
            applyGeneric<NPack, Ans><<<grid, block, 0, stream>>>(
                factory, n, nPack,
                ans, args...);
        } else {
            auto [grid, block] = distributer(n);
            applyGeneric<1, Ans><<<grid, block, 0, stream>>>(
                factory, n, n,
                ans, args...);
        }

        return cudaPeekAtLastError();
    }

    void sigmoid(float *out, float const *in, unsigned long long n) {
        thrust::device_vector<float>
            in_(in, in + n),
            out_(n);

        ThreadsDistributer distributer;
        auto [grid, block] = distributer(n);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < 1000; ++i)
            launch<SigmoidFactory, float, float>(distributer, SigmoidFactory(), n, out_.data().get(), in_.data().get(), 0);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        fmt::println("gpu: {} ns", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());

        cudaMemcpy(out, out_.data().get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    }

}// namespace refactor::kernel::cuda
