#include "kernel/attributes/topk_info.h"
#include <numeric>

namespace refactor::kernel {

TopKInfo::TopKInfo(uint32_t topk, uint32_t axis, Tensor const &input){
    this->topk =topk;
    auto tmpStride =  axis == 0 ? 0 : input.strides()[axis - 1];
    this->stride = {input.strides()[axis],\
                tmpStride,\
                tmpStride/input.shape[axis]*topk};
    this->size = {input.shape[axis], \
                input.elementsSize()/input.shape[axis]};
}        

}
