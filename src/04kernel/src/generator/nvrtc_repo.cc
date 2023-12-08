#ifdef USE_CUDA

#include "nvrtc_repo.h"
#include <nvrtc.h>

#define NVRTC_ASSERT(CALL)                                                 \
    if (auto status = CALL; status != NVRTC_SUCCESS) {                     \
        RUNTIME_ERROR(fmt::format("nvrtc failed on \"" #CALL "\" with {}", \
                                  nvrtcGetErrorString(status)));           \
    }

namespace refactor::kernel::nvrtc {

    Handler::Handler(const char *name, const char *code, const char *symbol) {

        nvrtcProgram prog;
        NVRTC_ASSERT(nvrtcCreateProgram(&prog, code, name, 0, nullptr, nullptr));
        // Compile the program with fmad disabled.
        // Note: Can specify GPU target architecture explicitly with '-arch' flag.
        const char *opts[] = {"--fmad=false"};
        auto compileResult = nvrtcCompileProgram(prog, 1, opts);
        // Obtain compilation log from the program.
        {
            size_t logSize;
            NVRTC_ASSERT(nvrtcGetProgramLogSize(prog, &logSize));
            std::vector<char> log(logSize);
            NVRTC_ASSERT(nvrtcGetProgramLog(prog, log.data()));
            fmt::println("{}", log.data());
        }
        if (compileResult != NVRTC_SUCCESS) {
            exit(1);
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

        CUDA_ASSERT(cuModuleLoadDataEx(&_module, ptx.c_str(), 0, 0, 0));
        CUDA_ASSERT(cuModuleGetFunction(&_kernel, _module, symbol));
    }

    Handler::~Handler() {
        cuModuleUnload(_module);
    }

    Arc<Handler> Handler::compile(const char *name, const char *code, const char *symbol) {
        static std::unordered_map<std::string, Arc<Handler>> REPO;
        auto it = REPO.find(name);
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
            {16, "half4"  },
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
