#include "cpu_kernel.hh"
#include <cmath>

namespace refactor::kernel {
    using K = RoPECpu;

    K::RoPECpu(decltype(info) info_) noexcept
        : Kernel(), info(info_) {}

    auto K::build(decltype(info) info, Tensor const &x) noexcept -> KernelBox {
        if (x.dataType != DataType::F32) {
            return nullptr;
        }
        return std::make_unique<K>(info);
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing rotary position embedding on cpu";
    }


    auto K::lower(Resources &) const -> RoutineWorkspace {
        return [batchsize = this->info.batchsize,
                seq_len = this->info.seq_len,
                n_heads = this->info.n_heads,
                head_dim = this->info.head_dim,
                theta = this->info.theta]//
            (Resources &, void *, void const *const *inputs, void *const *outputs) {
                auto input = reinterpret_cast<float const *>(inputs[0]);
                auto pos_ids = reinterpret_cast<int64_t const *>(inputs[1]);
                auto output = reinterpret_cast<float *>(outputs[0]);
                auto half_dim = head_dim / 2;

                for (unsigned int batch_id = 0; batch_id < batchsize; batch_id++) {
                    for (unsigned int pos = 0; pos < seq_len; pos++) {
                        auto pos_id = pos_ids[batch_id * seq_len + pos];
                        for (unsigned int head = 0; head < n_heads; head++) {
                            auto offset = batch_id * seq_len * n_heads * head_dim + pos * n_heads * head_dim + head * head_dim;
                            for (unsigned int i = 0; i < head_dim; i++) {
                                if (i < half_dim) {
                                    float freq = pos_id * powf(theta, -float(i * 2) / head_dim);
                                    float cos_freq = cos(freq);
                                    float sin_freq = sin(freq);
                                    output[offset + i] =
                                        input[offset + i] * float(cos_freq) - input[offset + i + half_dim] * float(sin_freq);
                                } else {
                                    float freq = pos_id * powf(theta, -float((i - half_dim) * 2) / head_dim);
                                    float cos_freq = cos(freq);
                                    float sin_freq = sin(freq);
                                    output[offset + i] =
                                        input[offset + i] * float(cos_freq) + input[offset + i - half_dim] * float(sin_freq);
                                }
                            }
                        }
                    }
                }
            };
    }

}// namespace refactor::kernel
