#include "kernel/attributes/moe_info.h"
#include <numeric>

namespace refactor::kernel {

AssignPosInfo::AssignPosInfo(uint32_t top, uint32_t expert_num, Tensor const &gate):\
    top(top), expert_num(expert_num),elementSize(gate.elementsSize()){}      

ReorderInfo::ReorderInfo(bool scatter, uint32_t top, TensorRefs inputs):\
    scatter(scatter), top(top),blockNum(inputs[1].get().elementsSize()), blockSize(inputs[0].get().strides()[0]){}  


}
