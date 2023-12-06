#include "executor.h"
#include "kernel/allocators.h"
#include <filesystem>
#include <fstream>

#ifdef USE_CUDA
#include "kernel/cuda/functions.cuh"
#endif// USE_CUDA

namespace refactor::python_ffi {

    Executor::Executor(computation::Graph graph, runtime::Stream stream)
        : _graph(std::move(graph)),
          _stream(std::move(stream)) {}

    void Executor::dispatch(Arc<hardware::Device> device, std::string allocator) {
        auto stream = _graph
                          .lower(device->type())
                          .lower(std::move(device),
                                 allocator == "flat"
                                     ? kernel::flatAllocate
                                     : kernel::reusableAllocate);
        std::swap(_stream, stream);
        std::vector<uint8_t> buffer;
        auto const &graph = _graph.internal().contiguous();
        for (auto i : graph.topology.globalInputs()) {
            auto size = graph.edges[i].tensor->bytesSize();
            buffer.resize(size);
            if (stream.getData(i, buffer.data(), size)) {
                _stream.setData(i, buffer.data(), size);
            }
        }
    }

    void Executor::setInput(count_t i, pybind11::array data) {
        i = _stream.graph().topology.globalInputs().at(i);

        auto const &name = _stream.graph().edges[i].name;
        auto const &edges = _graph.internal().contiguous().edges;
        auto const &tensor = *std::find_if(edges.begin(), edges.end(), [&](auto const &e) { return e.name == name; })->tensor;
        ASSERT(tensor.bytesSize() == static_cast<size_t>(data.nbytes()), "input size mismatch");
        _stream.setData(i, data.data(), data.nbytes());
    }

    auto Executor::getOutput(count_t i) -> pybind11::array {
        i = _stream.graph().topology.globalOutputs().at(i);

        auto const &name = _stream.graph().edges[i].name;
        auto const &edges = _graph.internal().contiguous().edges;
        auto const &tensor = *std::find_if(edges.begin(), edges.end(), [&](auto const &e) { return e.name == name; })->tensor;
        auto ans = pybind11::array(buildNumpyDType(tensor.dataType), std::move(tensor.shape));
        _stream.getData(i, ans.mutable_data(), ans.nbytes());
        return ans;
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

        auto header = std::vector<char>(aligned, ' ');
        constexpr static char MAGIC[]{static_cast<char>(0x93), 'N', 'U', 'M', 'P', 'Y', 1, 0};
        std::memcpy(header.data(), MAGIC, sizeof(MAGIC));
        auto headerLen = aligned - 10;
        header[8] = static_cast<char>(headerLen);
        header[9] = static_cast<char>(headerLen >> 8);
        std::memcpy(header.data() + 10, dictionary.data(), dictionary.size());
        header[aligned - 1] = '\n';

        os.write(header.data(), aligned);
        os.write(ptr, size);
    }

    void Executor::trace(std::string path_, std::string format) {
        namespace fs = std::filesystem;

        auto path = fs::path(std::move(path_));
        fs::create_directories(path);
        ASSERT(fs::is_directory(path), "Failed to create \"{}\"", path.c_str());

        auto const npy = format == "npy";
        size_t dataIdx = 0;

        auto const &graph = _graph.internal().contiguous();
        auto it = graph.topology.begin();
        _stream.trace([&](count_t nodeIdx, void const *const *inputs, void const *const *outputs) {
            auto [nodeIdx_, i_, o_] = *it++;
            ASSERT(nodeIdx_ == nodeIdx, "node index mismatch");

            std::stringstream meta;
            std::vector<char> buffer;
            auto fn = [&](char dir, count_t idx, count_t edgeIdx, void const *const *addresses)
                -> std::string {
                if (!addresses[idx] || (reinterpret_cast<size_t>(addresses[idx]) & 1)) {
                    return "";
                }
                auto const &edge = graph.edges[edgeIdx];
                if (!edge.tensor) { return ""; }

                auto size = edge.tensor->bytesSize();
                buffer.resize(size);
#ifdef USE_CUDA
                kernel::cuda::copyOut(buffer.data(), addresses[idx], size);
#endif

                auto file = path / fmt::format("data{:06}.{}", dataIdx++, format);
                fs::remove(file);
                std::ofstream os(file, std::ios::binary);
                if (npy) {
                    writeNpy(std::move(os), buffer.data(), size,
                             edge.tensor->dataType, edge.tensor->shape);
                } else {
                    writeBin(std::move(os), buffer.data(), size);
                }
                return file.string();
            };

            meta << "node\t" << nodeIdx << '\t'
                 << graph.nodes[nodeIdx].name << std::endl;
            for (auto i : range0_(i_.size())) {
                auto edgeIdx = i_[i];
                meta << "input\t" << i << '\t'
                     << graph.edges[edgeIdx].name << '\t'
                     << fn('i', i, edgeIdx, inputs) << std::endl;
            }
            for (auto i : range0_(o_.size())) {
                auto edgeIdx = o_[i];
                meta << "output\t" << i << '\t'
                     << graph.edges[edgeIdx].name << '\t'
                     << fn('o', i, edgeIdx, outputs) << std::endl;
            }
            std::ofstream(path / fmt::format("node{:06}.meta", nodeIdx)) << meta.str();
        });
    }

    void Executor::debugInfo() const noexcept {
        auto const &nodes = _graph.internal().contiguous().nodes;
        for (auto i : range0_(nodes.size())) {
            fmt::println("{}. {}", i, nodes[i].name);
        }
    }

}// namespace refactor::python_ffi
