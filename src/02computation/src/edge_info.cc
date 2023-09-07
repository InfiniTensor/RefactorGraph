#include "graph/edge_info.h"
#include "common/error_handler.h"

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

    // bool EmptyEdgeInfo::operator==(EmptyEdgeInfo const &rhs) const { return false; }
    // bool EmptyEdgeInfo::operator!=(EmptyEdgeInfo const &rhs) const { return !operator==(rhs); }

    // bool Tensor::operator==(Tensor const &rhs) const { return dataType == rhs.dataType && shape == rhs.shape; }
    // bool Tensor::operator!=(Tensor const &rhs) const { return !operator==(rhs); }

    // bool ShapeVariable::operator==(ShapeVariable const &rhs) const { return shape == rhs.shape; }
    // bool ShapeVariable::operator!=(ShapeVariable const &rhs) const { return !operator==(rhs); }

    // EdgeInfo::EdgeInfo() : info(EmptyEdgeInfo()) {}
    // EdgeInfo::EdgeInfo(Tensor tensor_) : info(std::move(tensor_)) {}
    // EdgeInfo::EdgeInfo(ShapeVariable shape_) : info(std::move(shape_)) {}

    // bool EdgeInfo::isEmpty() const { return std::holds_alternative<EmptyEdgeInfo>(info); }
    // bool EdgeInfo::isTensor() const { return std::holds_alternative<Tensor>(info); }
    // bool EdgeInfo::isShapeVariable() const { return std::holds_alternative<ShapeVariable>(info); }

    // Tensor &EdgeInfo::tensor() {
    //     if (std::holds_alternative<Tensor>(info)) {
    //         return std::get<Tensor>(info);
    //     } else {
    //         RUNTIME_ERROR("Edge type error");
    //     }
    // }

    // Tensor const &EdgeInfo::tensor() const {
    //     if (std::holds_alternative<Tensor>(info)) {
    //         return std::get<Tensor>(info);
    //     } else {
    //         RUNTIME_ERROR("Edge type error");
    //     }
    // }

    // ShapeVariable &EdgeInfo::shapeVariable() {
    //     if (std::holds_alternative<ShapeVariable>(info)) {
    //         return std::get<ShapeVariable>(info);
    //     } else {
    //         RUNTIME_ERROR("Edge type error");
    //     }
    // }

    // ShapeVariable const &EdgeInfo::shapeVariable() const {
    //     if (std::holds_alternative<ShapeVariable>(info)) {
    //         return std::get<ShapeVariable>(info);
    //     } else {
    //         RUNTIME_ERROR("Edge type error");
    //     }
    // }

    // bool EdgeInfo::operator==(EdgeInfo const &rhs) const {
    //     if (info.index() != rhs.info.index()) {
    //         return false;
    //     } else {
    //         switch (info.index()) {
    //             case 0:
    //                 return std::get<0>(info) == std::get<0>(rhs.info);
    //             case 1:
    //                 return std::get<1>(info) == std::get<1>(rhs.info);
    //             case 2:
    //                 return std::get<2>(info) == std::get<2>(rhs.info);
    //             default:
    //                 RUNTIME_ERROR("Unreachable");
    //         }
    //     }
    // }
    // bool EdgeInfo::operator!=(EdgeInfo const &rhs) const { return !operator==(rhs); }

}// namespace refactor::graph
