#ifdef USE_CUDA

#include "nvrtc_repo.h"
#include "hardware/device_manager.h"
#include <nvrtc.h>

#define NVRTC_ASSERT(CALL)                                                 \
    if (auto status = CALL; status != NVRTC_SUCCESS) {                     \
        RUNTIME_ERROR(fmt::format("nvrtc failed on \"" #CALL "\" with {}", \
                                  nvrtcGetErrorString(status)));           \
    }

namespace refactor::kernel::nvrtc {

    Handler::Handler(std::string_view name,
                     std::string_view code,
                     std::string_view symbol) {
        std::string header;
        if (code.find("half") != code.npos) {
            header += "#include <cuda_fp16.h>\n";
        }
        if (code.find("nv_bfloat16") != code.npos) {
            header += "#include <cuda_bf16.h>\n";
        }

        nvrtcProgram prog;
        if (!header.empty()) {
            code = (header += code.data());
        }
        NVRTC_ASSERT(nvrtcCreateProgram(&prog, code.data(), name.data(), 0, nullptr, nullptr));

        std::vector<std::string> opts{"--std=c++20", "--gpu-architecture=compute_80"};
#ifdef CUDA_INCLUDE_PATH
        opts.emplace_back(fmt::format("-I{}", CUDA_INCLUDE_PATH));
#endif
        std::vector<const char *> optsPtr(opts.size());
        std::transform(opts.begin(), opts.end(), optsPtr.begin(),
                       [](auto &s) { return s.c_str(); });
        auto compileResult = nvrtcCompileProgram(
            prog, optsPtr.size(), optsPtr.data());
        // Obtain compilation log from the program.
        {
            size_t logSize;
            NVRTC_ASSERT(nvrtcGetProgramLogSize(prog, &logSize));
            if (logSize > 1) {
                std::vector<char> log(logSize);
                NVRTC_ASSERT(nvrtcGetProgramLog(prog, log.data()));
                fmt::println("{}", log.data());
            }
        }
        if (compileResult != NVRTC_SUCCESS) {
            fmt::println("wrong code:");
            fmt::println("{}", code);
            abort();
        }
        // Obtain PTX from the program.
        std::string ptx;
        {
            size_t ptxSize;
            NVRTC_ASSERT(nvrtcGetPTXSize(prog, &ptxSize));
            ptx.resize(ptxSize);
            NVRTC_ASSERT(nvrtcGetPTX(prog, ptx.data()));
        }
        // Destroy the program.
        NVRTC_ASSERT(nvrtcDestroyProgram(&prog));

        hardware::device::fetch(hardware::Device::Type::Nvidia);
        CUDA_ASSERT(cuModuleLoadData(&_module, ptx.data()));
        CUDA_ASSERT(cuModuleGetFunction(&_kernel, _module, symbol.data()));
    }

    Handler::~Handler() {
        cuModuleUnload(_module);
    }

    Arc<Handler> Handler::compile(
        std::string_view name,
        std::string_view code,
        std::string_view symbol) {
        static std::unordered_map<std::string, Arc<Handler>> REPO;
        auto it = REPO.find(name.data());
        if (it == REPO.end()) {
            std::tie(it, std::ignore) = REPO.emplace(name, Arc<Handler>(new Handler(name, code, symbol)));
        }
        return it->second;
    }

    CUfunction Handler::kernel() const {
        return _kernel;
    }

    std::string_view memCopyType(size_t size) {
        // clang-format off
        static const std::unordered_map<size_t, std::string_view> TABLE {
            { 1, "uchar1" },
            { 2, "half"   },
            { 3, "uchar3" },
            { 4, "float"  },
            { 6, "ushort3"},
            { 8, "double" },
            {12, "uint3"  },
            {16, "float4" },
            {24, "double3"},
            {32, "double4"},
        };
        // clang-format on
        return TABLE.at(size);
    }

    std::string_view dataType(DataType dt) {
        using DT = DataType;
        // clang-format off
        static const std::unordered_map<uint8_t, std::string_view> TABLE {
            {DT::U8  , "unsigned char"     },
            {DT::U16 , "unsigned short"    },
            {DT::U32 , "unsigned int"      },
            {DT::U64 , "unsigned long long"},
            {DT::I8  , "char"              },
            {DT::I16 , "short"             },
            {DT::I32 , "int"               },
            {DT::I64 , "long long"         },
            {DT::FP16, "half"              },
            {DT::BF16, "nv_bfloat16"       },
            {DT::F32 , "float"             },
            {DT::F64 , "double"            },
            {DT::Bool, "bool"              },
        };
        // clang-format on
        return TABLE.at(dt);
    }

}// namespace refactor::kernel::nvrtc

#endif
