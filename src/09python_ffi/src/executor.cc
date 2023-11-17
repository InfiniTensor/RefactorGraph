#include "executor.h"
#include <filesystem>
#include <fstream>

#ifdef USE_CUDA
#include "kernel/cuda/functions.cuh"
#endif// USE_CUDA

namespace refactor::python_ffi {

    Executor::Executor(computation::Graph graph, runtime::Stream stream)
        : _graph(std::move(graph)),
          _stream(std::move(stream)) {}

    void Executor::setInput(count_t i, pybind11::array data) {
        _stream.setInput(i, data.data(), data.nbytes());
    }

    auto Executor::getOutput(count_t i) -> pybind11::array {
        auto globalOutputs = _graph.internal().contiguous().topology.globalOutputs();
        ASSERT(i < globalOutputs.size(), "input index out of range");

        auto const &tensor = *_graph.internal().contiguous().edges[globalOutputs[i]].tensor;
        auto ans = pybind11::array(buildNumpyDType(tensor.dataType), std::move(tensor.shape), nullptr);
        _stream.getOutput(i, ans.mutable_data(), ans.nbytes());
        return ans;
    }

    auto Executor::prepare() -> std::vector<count_t> {
        return _stream.prepare();
    }

    void Executor::run() {
        _stream.run();
    }

    void Executor::bench(bool sync) {
#ifdef USE_CUDA
        auto ans = _stream.bench(sync ? kernel::cuda::sync : nullptr);
#else
        auto ans = _stream.bench(nullptr);
#endif// USE_CUDA
        auto const &nodes = _graph.internal().contiguous().nodes;
        for (auto i : range0_(nodes.size())) {
            fmt::println("{} {} {}",
                         i,
                         nodes[i].name,
                         std::chrono::duration_cast<std::chrono::microseconds>(ans[i]).count());
        }
    }


    static void writeBin(std::ofstream os, char const *ptr, size_t size) {
        os.write(ptr, size);
    }

    static void writeNpy(std::ofstream os, char const *ptr, size_t size,
                         DataType dataType, computation::Shape const &shape) {
        std::stringstream ss;
        for (auto d : shape) { ss << d << ","; }
        const char *dtstr;
        switch (dataType) {
                // clang-format off
            case DataType::F32       : dtstr = "<f4"; break;
            case DataType::U8        : dtstr = "|u1"; break;
            case DataType::I8        : dtstr = "|i1"; break;
            case DataType::U16       : dtstr = "<u2"; break;
            case DataType::I16       : dtstr = "<i2"; break;
            case DataType::I32       : dtstr = "<i4"; break;
            case DataType::I64       : dtstr = "<i8"; break;
            case DataType::Bool      : dtstr = "|b1"; break;
            case DataType::FP16      : dtstr = "<f2"; break;
            case DataType::F64       : dtstr = "<f8"; break;
            case DataType::U32       : dtstr = "<u4"; break;
            case DataType::U64       : dtstr = "<u8"; break;
            case DataType::Complex64 : dtstr = "<c8"; break;
            case DataType::Complex128: dtstr = "<c16"; break;
            // clang-format on
            default:
                UNREACHABLE();
                break;
        }
        constexpr static auto ALIGN = 64;
        auto dictionary = fmt::format("{{'descr': '{}', 'fortran_order': False, 'shape': ({}), }}", dtstr, ss.str());
        auto aligned = (dictionary.size() + 11 + ALIGN - 1) & ~(ALIGN - 1);
        auto headerLen = dictionary.size();
        auto space = std::vector<char>(headerLen - 1, ' ');
        os.write("\x93NUMPY\x01\x00", 8);
        os.put(static_cast<char>(headerLen));
        os.put(static_cast<char>(headerLen >> 8));
        os.write(dictionary.data(), dictionary.size());
        os.write(space.data(), space.size());
        os.put('\n');
        os.write(ptr, size);
    }

    void Executor::trace(std::string path_, std::string format) {
        namespace fs = std::filesystem;

        auto path = fs::path(std::move(path_));
        fs::create_directories(path);
        ASSERT(fs::is_directory(path), "Failed to create \"{}\"", path.c_str());

        auto npy = format == "npy";

        auto it = _graph.internal().contiguous().topology.begin();
        _stream.trace([&](count_t nodeIdx, void const *const *inputs, void const *const *outputs) {
            auto [nodeIdx_, i_, o_] = *it++;
            ASSERT(nodeIdx_ == nodeIdx, "node index mismatch");
            auto nodeName = _graph.internal().contiguous().nodes[nodeIdx].name;
            std::replace(nodeName.begin(), nodeName.end(), '/', '_');
            std::replace(nodeName.begin(), nodeName.end(), '.', '-');

            std::vector<char> buffer;
            auto const &edges = _graph.internal().contiguous().edges;
            auto fn = [&](char dir, count_t idx, count_t edgeIdx, void const *const *addresses) {
                if (!addresses[idx]) { return; }
                auto const &edge = edges[edgeIdx];

                auto edgeName = edge.name;
                std::replace(edgeName.begin(), edgeName.end(), '/', '_');
                std::replace(edgeName.begin(), edgeName.end(), '.', '-');
                auto file = path / fmt::format("{}({}_{}{}).{}", edgeName, nodeName, dir, idx, format);
                fs::remove(file);
                std::ofstream os(file, std::ios::binary);

                auto size = edge.tensor->bytesSize();
                buffer.resize(size);
#ifdef USE_CUDA
                kernel::cuda::copyOut(buffer.data(), addresses[idx], size);
#endif
                if (npy) {
                    writeNpy(std::move(os), buffer.data(), size,
                             edge.tensor->dataType, edge.tensor->shape);
                } else {
                    writeBin(std::move(os), buffer.data(), size);
                }
            };

            for (auto i : range0_(i_.size())) { fn('i', i, i_[i], inputs); }
            for (auto i : range0_(o_.size())) { fn('o', i, o_[i], outputs); }
        });
    }

    void Executor::debugInfo() const noexcept {
        auto const &nodes = _graph.internal().contiguous().nodes;
        for (auto i : range0_(nodes.size())) {
            fmt::println("{}. {}", i, nodes[i].name);
        }
    }

}// namespace refactor::python_ffi
