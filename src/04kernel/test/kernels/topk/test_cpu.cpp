#include "../../../src/kernels/topk/cpu_kernel.hh"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace kernel;

TEST(kernel, TopKCpu) {
    // build routine    
    auto inputTensor = Tensor::share(DataType::F32, Shape{3, 4});
    auto outputTensor0 = Tensor::share(DataType::F32, Shape{3, 3});
    auto outputTensor1 = Tensor::share(DataType::I64, Shape{3, 3});

    auto kernel = TopKCpu::build(TopKInfo(3,1, *inputTensor));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float> ins(inputTensor->elementsSize());
    std::vector<float>  out0(outputTensor0->elementsSize());
    std::vector<uint32_t> out1(outputTensor1->elementsSize());

    std::iota(ins.begin(), ins.end(), 0);
    // inference
    void const *inputs[]{ins.data()};
    void *outputs[]{out0.data(), out1.data()};
    routine(res, nullptr, inputs, outputs);    
  
    // check
    std::vector<float> expectVal = {3,2,1,7,6,5,11,10,9};
    std::vector<int64_t> expectIdx = {3,2,1,3,2,1,3,2,1};
    std::for_each(out0.begin(), out0.end(),[](const float &val){std::cout<<val<<" ";});

    for(size_t i=0;i< expectVal.size(); ++i){
        EXPECT_EQ(expectVal[i], out0[i]);
        EXPECT_EQ(expectIdx[i], out1[i]);
    }
}

TEST(kernel, TopKCpu1) {
    // build routine    
    auto inputTensor = Tensor::share(DataType::F32, Shape{2, 4, 2});
    auto outputTensor0 = Tensor::share(DataType::F32, Shape{2, 3, 2});
    auto outputTensor1 = Tensor::share(DataType::U32, Shape{2, 3, 2});

    auto kernel = TopKCpu::build(TopKInfo(3,1, *inputTensor));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float> ins(inputTensor->elementsSize());
    std::vector<float>  out0(outputTensor0->elementsSize());
    std::vector<uint32_t> out1(outputTensor1->elementsSize());

    std::iota(ins.begin(), ins.end(), 0);
    // inference
    void const *inputs[]{ins.data()};
    void *outputs[]{out0.data(), out1.data()};
    routine(res, nullptr, inputs, outputs);    
    std::for_each(out0.begin(), out0.end(),[](const float &val){std::cout<<val<<" ";});

    // check
    std::vector<float> expectVal = {6,7,4,5,2,3,14,15,12,13,10,11};
    std::vector<uint32_t> expectIdx = {3,3,2,2,1,1,3,3,2,2,1,1};
    

    for(size_t i=0;i< expectVal.size(); ++i){
        EXPECT_EQ(expectVal[i], out0[i]);
        EXPECT_EQ(expectIdx[i], out1[i]);
    }
}

TEST(kernel, TopKCpu2) {
    // build routine    
    auto inputTensor = Tensor::share(DataType::F32, Shape{2, 4, 2});
    auto outputTensor0 = Tensor::share(DataType::F32, Shape{1, 4, 2});
    auto outputTensor1 = Tensor::share(DataType::U32, Shape{1, 4, 2});

    auto kernel = TopKCpu::build(TopKInfo(1,0, *inputTensor));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float> ins(inputTensor->elementsSize());
    std::vector<float>  out0(outputTensor0->elementsSize());
    std::vector<uint32_t> out1(outputTensor1->elementsSize());

    std::iota(ins.begin(), ins.end(), 0);
    // inference
    void const *inputs[]{ins.data()};
    void *outputs[]{out0.data(), out1.data()};
    routine(res, nullptr, inputs, outputs);    
    std::for_each(out0.begin(), out0.end(),[](const float &val){std::cout<<val<<" ";});

    // check
    std::vector<float> expectVal = {8,9,10,11,12,13,14,15};
    std::vector<uint32_t> expectIdx = {1,1,1,1,1,1,1,1};
    

    for(size_t i=0;i< expectVal.size(); ++i){
        EXPECT_EQ(expectVal[i], out0[i]);
        EXPECT_EQ(expectIdx[i], out1[i]);
    }
}


TEST(kernel, TopKCpu3) {
    // build routine    
    auto inputTensor = Tensor::share(DataType::F32, Shape{2, 3, 2, 2});
    auto outputTensor0 = Tensor::share(DataType::F32, Shape{1, 3, 2, 2});
    auto outputTensor1 = Tensor::share(DataType::U32, Shape{1, 3, 2, 2});

    auto kernel = TopKCpu::build(TopKInfo(1,0, *inputTensor));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float> ins(inputTensor->elementsSize());
    std::vector<float>  out0(outputTensor0->elementsSize());
    std::vector<uint32_t> out1(outputTensor1->elementsSize());

    std::iota(ins.begin(), ins.end(), 0);
    // inference
    void const *inputs[]{ins.data()};
    void *outputs[]{out0.data(), out1.data()};
    routine(res, nullptr, inputs, outputs);    
    std::for_each(out0.begin(), out0.end(),[](const float &val){std::cout<<val<<" ";});

    // check
    std::vector<float> expectVal = {12, 13, 14, 15, 16, 17, 18,  19, 20,21, 22,23};
    std::vector<uint32_t> expectIdx = {1,1,1,1,1,1,1,1,1,1,1,1};
    

    for(size_t i=0;i< expectVal.size(); ++i){
        EXPECT_EQ(expectVal[i], out0[i]);
        EXPECT_EQ(expectIdx[i], out1[i]);
    }
}

TEST(kernel, TopKCpu4) {
    // build routine    
    auto inputTensor = Tensor::share(DataType::F32, Shape{2, 3, 2, 2});
    auto outputTensor0 = Tensor::share(DataType::F32, Shape{2, 2, 2, 2});
    auto outputTensor1 = Tensor::share(DataType::U32, Shape{2, 2, 2, 2});

    auto kernel = TopKCpu::build(TopKInfo(2,1, *inputTensor));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float> ins(inputTensor->elementsSize());
    std::vector<float>  out0(outputTensor0->elementsSize());
    std::vector<uint32_t> out1(outputTensor1->elementsSize());

    std::iota(ins.begin(), ins.end(), 0);
    // inference
    void const *inputs[]{ins.data()};
    void *outputs[]{out0.data(), out1.data()};
    routine(res, nullptr, inputs, outputs);    
    std::for_each(out0.begin(), out0.end(),[](const float &val){std::cout<<val<<" ";});

    // check
    std::vector<float> expectVal = {8, 9, 10, 11,4,5,6,7,20,21,22,23,16,17,18,19};
    std::vector<uint32_t> expectIdx = {2,2,2,2,1,1,1,1,2,2,2,2,1,1,1,1};
    

    for(size_t i=0;i< expectVal.size(); ++i){
        EXPECT_EQ(expectVal[i], out0[i]);
        EXPECT_EQ(expectIdx[i], out1[i]);
    }
}


TEST(kernel, TopKCpu5) {
    // build routine    
    auto inputTensor = Tensor::share(DataType::F32, Shape{2, 3, 2, 2});
    auto outputTensor0 = Tensor::share(DataType::F32, Shape{2, 3, 1, 2});
    auto outputTensor1 = Tensor::share(DataType::U32, Shape{2, 3, 1, 2});

    auto kernel = TopKCpu::build(TopKInfo(1,2, *inputTensor));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float> ins(inputTensor->elementsSize());
    std::vector<float>  out0(outputTensor0->elementsSize());
    std::vector<uint32_t> out1(outputTensor1->elementsSize());

    std::iota(ins.begin(), ins.end(), 0);
    // inference
    void const *inputs[]{ins.data()};
    void *outputs[]{out0.data(), out1.data()};
    routine(res, nullptr, inputs, outputs);    
    std::for_each(out0.begin(), out0.end(),[](const float &val){std::cout<<val<<" ";});

    // check
    std::vector<float> expectVal = {2,3,6,7,10,11,14,15,18,19,22,23};
    std::vector<uint32_t> expectIdx = {1,1,1,1,1,1,1,1,1,1,1,1};
    

    for(size_t i=0;i< expectVal.size(); ++i){
        EXPECT_EQ(expectVal[i], out0[i]);
        EXPECT_EQ(expectIdx[i], out1[i]);
    }
}

TEST(kernel, TopKCpu6) {
    // build routine    
    auto inputTensor = Tensor::share(DataType::F32, Shape{2, 3, 2, 2});
    auto outputTensor0 = Tensor::share(DataType::F32, Shape{2, 3, 2, 1});
    auto outputTensor1 = Tensor::share(DataType::U32, Shape{2, 3, 2, 1});

    auto kernel = TopKCpu::build(TopKInfo(1,3, *inputTensor));
    ASSERT_TRUE(kernel);
    auto res = runtime::Resources();
    auto routine = kernel->lower(res).routine;
    // put input data
    std::vector<float> ins(inputTensor->elementsSize());
    std::vector<float>  out0(outputTensor0->elementsSize());
    std::vector<uint32_t> out1(outputTensor1->elementsSize());

    std::iota(ins.begin(), ins.end(), 0);
    // inference
    void const *inputs[]{ins.data()};
    void *outputs[]{out0.data(), out1.data()};
    routine(res, nullptr, inputs, outputs);    
    std::for_each(out0.begin(), out0.end(),[](const float &val){std::cout<<val<<" ";});

    // check
    std::vector<float> expectVal = {1,3,5,7,9,11,13,15,17,19,21,23};
    std::vector<uint32_t> expectIdx = {1,1,1,1,1,1,1,1,1,1,1,1};
    

    for(size_t i=0;i< expectVal.size(); ++i){
        EXPECT_EQ(expectVal[i], out0[i]);
        EXPECT_EQ(expectIdx[i], out1[i]);
    }
}