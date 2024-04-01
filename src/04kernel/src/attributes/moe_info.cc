#include "kernel/attributes/moe_info.h"
#include <numeric>

namespace refactor::kernel {

AssignPosInfo::AssignPosInfo(int64_t top, int64_t expert_num, Tensor const &gate):\
    top(top), expert_num(expert_num),elementSize(gate.elementsSize()){}      

ReorderInfo::ReorderInfo(bool scatter, int64_t top, TensorRefs inputs):\
    scatter(scatter), top(top),blockNum(inputs[1].get().elementsSize()), blockSize(inputs[0].get().strides()[0]){}  


}
