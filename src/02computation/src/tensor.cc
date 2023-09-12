#include "computation/tensor.h"
#include "common/error_handler.h"
#include <numeric>

namespace refactor::computation {

    DimVariableInternal::DimVariableInternal(std::string name_, std::optional<int64_t> value_)
        : name(std::move(name_)), value(value_) {}

    DimExpr::DimExpr(int64_t val) : expr(val) {}
    DimExpr::DimExpr(std::string name)
        : expr(std::make_shared<DimVariableInternal>(std::move(name))) {}

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
    bool DimExpr::isVariable() const { return std::holds_alternative<DimVariable>(expr); }
    bool DimExpr::hasValue() const { return isValue() || variable()->value; }
    int64_t DimExpr::value() const { return isValue() ? std::get<int64_t>(expr) : variable()->value.value(); }
    DimVariable DimExpr::variable() const { return std::get<DimVariable>(expr); }

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

}// namespace refactor::computation
