#include "../../../src/kernels/moe/cpu_kernel.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, AssignPosCpu) {
    // build routine    
    //auto inputTensor = Tensor::share(DataType::F32, Shape{4, 1024});
    auto gate = Tensor::share(DataType::U32, Shape{8, 2});
    auto expert_cnt = Tensor::share(DataType::U32, Shape{4});
    auto pos = Tensor::share(DataType::U32, Shape{16});

    auto kernel = AssignPosCpu::build(AssignPosInfo(2,4, *gate));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<uint8_t> ins = {3,2, 0,1, 2,1, 1,3, 2,0, 1,3, 1,0, 1,2};
    std::vector<uint8_t>  out0(expert_cnt->elementsSize());
    std::vector<uint8_t> out1(pos->elementsSize());

    // inference
    void const *inputs[]{ins.data()};
    void *outputs[]{out0.data(), out1.data()};
    routine(res, nullptr, inputs, outputs);    
  
    // check
    std::vector<uint32_t> expectExpertCnt = {3,6,4,3};
    std::vector<uint32_t> expectPos = {13,9,2, 14,12,10,6,5,3, 15,8,4,1, 11,7,0};
    //std::for_each(out0.begin(), out0.end(),[](const float &val){std::cout<<val<<" ";});

    for(size_t i=0;i< expectPos.size(); ++i){
        EXPECT_EQ(expectPos[i], out1[i]);
    }
    for(size_t i=0;i< expectExpertCnt.size(); ++i){
        EXPECT_EQ(expectExpertCnt[i], out0[i]);
    }
}

TEST(kernel, ReorderScatterCpu) {
    // build routine    
    const int seq = 8, hid = 4, top = 2;
    auto input = Tensor::share(DataType::U32, Shape{seq, hid});
    auto pos = Tensor::share(DataType::U32, Shape{seq * top});
    std::vector<Arc<Tensor>> inputTensors{input, pos};
    TensorRefs inputs_;
    inputs_.reserve(inputTensors.size());
    std::transform(inputTensors.begin(), inputTensors.end(),
                   std::back_inserter(inputs_),
                   [](auto const &it) { return std::cref(*it); });

    auto kernel = ReorderCpu::build(ReorderInfo(true, top, inputs_));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float>  ins0(input->elementsSize());
    std::iota(ins0.begin(), ins0.end(), 0);
    std::vector<uint32_t> ins1 = {13,9,2, 14,12,10,6,5,3, 15,8,4,1, 11,7,0};
    std::vector<float>  out(input->elementsSize() * top);

    // inference
    void const *inputs[]{ins0.data(), ins1.data()};
    void *outputs[]{out.data()};
    routine(res, nullptr, inputs, outputs);    
    std::for_each(out.begin(), out.end(),[](const float &val){std::cout<<val<<" ";});
    // check    
    for(size_t i=0;i< seq; ++i){
        int row = ins1[i]/top;
        for(size_t j = 0; j<hid; j++)
            EXPECT_EQ(ins0[row *hid + j], out[i*hid + j]);
    }
}
