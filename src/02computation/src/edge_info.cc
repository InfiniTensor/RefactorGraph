#include "graph/edge_info.h"
#include "common/error_handler.h"
#include <numeric>

namespace refactor::graph {

    DimExpr::DimExpr(int64_t val) : expr(val) {}
    DimExpr::DimExpr(std::string &&name) : expr(std::forward<std::string>(name)) {}
    bool DimExpr::operator==(DimExpr const &rhs) const {
        if (expr.index() != rhs.expr.index()) {
            return false;
        } else {
            switch (expr.index()) {
                case 0:
                    return std::get<0>(expr) == std::get<0>(rhs.expr);
                case 1:
                    return std::get<1>(expr) == std::get<1>(rhs.expr);
                default:
                    RUNTIME_ERROR("Unreachable");
            }
        }
    }
    bool DimExpr::operator!=(DimExpr const &rhs) const { return !operator==(rhs); }
    bool DimExpr::isValue() const { return std::holds_alternative<int64_t>(expr); }
    bool DimExpr::isVariable() const { return std::holds_alternative<std::string>(expr); }
    int64_t DimExpr::value() const { return std::get<int64_t>(expr); }
    std::string const &DimExpr::variable() const { return std::get<std::string>(expr); }

    Blob::Blob(void *ptr_) : ptr(ptr_) {}
    Blob::~Blob() { std::free(ptr); }

    Tensor::Tensor(common::DataType dt_, Shape shape_, std::shared_ptr<Blob> data_)
        : dataType(dt_),
          shape(std::move(shape_)),
          data(std::move(data_)) {}
    bool Tensor::operator==(Tensor const &rhs) const {
        return dataType == rhs.dataType &&
               shape == rhs.shape &&
               data.get() == rhs.data.get();
    }
    bool Tensor::operator!=(Tensor const &rhs) const { return !operator==(rhs); }
    bool Tensor::hasData() const { return data.get(); }
    size_t Tensor::elementsSize() const {
        return std::accumulate(shape.begin(), shape.end(), 1,
                               [](auto acc, auto const &it) { return acc * it.value(); });
    }
    size_t Tensor::bytesSize() const {
        return common::dataTypeSize(dataType) * elementsSize();
    }

}// namespace refactor::graph
