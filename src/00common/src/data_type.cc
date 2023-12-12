#include "common/data_type.h"
#include "common/error_handler.h"
#include <unordered_set>

namespace refactor {
    using DT = DataType;
    using Enum = decltype(DT::internal);

    std::optional<DT> DT::parse(uint8_t value) noexcept {
        switch (value) {
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 9:
            case 10:
            case 11:
            case 12:
            case 13:
                return {(Enum) value};
            default:
                return {};
        }
    }

    std::string_view DT::name() const noexcept {
        switch (internal) {
            case DT::F32:
                return "F32";
            case DT::U8:
                return "U8 ";
            case DT::I8:
                return "I8 ";
            case DT::U16:
                return "U16";
            case DT::I16:
                return "I16";
            case DT::I32:
                return "I32";
            case DT::I64:
                return "I64";
            case DT::Bool:
                return "Bool";
            case DT::FP16:
                return "FP16";
            case DT::F64:
                return "F64";
            case DT::U32:
                return "U32";
            case DT::U64:
                return "U64";
            case DT::Complex64:
                return "Complex64";
            case DT::Complex128:
                return "Complex128";
            case DT::BF16:
                return "BF16";
            default:
                UNREACHABLE();
        }
    }

    bool DT::isIeee754() const noexcept {
        static const std::unordered_set<Enum> set{DT::F32, DT::FP16, DT::F64};
        return set.contains(internal);
    }
    bool DT::isFloat() const noexcept {
        static const std::unordered_set<Enum> set{DT::F32, DT::FP16, DT::F64, DT::BF16};
        return set.contains(internal);
    }
    bool DT::isSignedLarge() const noexcept {
        static const std::unordered_set<Enum> set{
            DT::F32, DT::FP16, DT::BF16, DT::F64, DT::I32, DT::I64};
        return set.contains(internal);
    }
    bool DT::isSigned() const noexcept {
        static const std::unordered_set<Enum> set{
            DT::F32, DT::FP16, DT::BF16, DT::F64,
            DT::I8, DT::I16, DT::I32, DT::I64};
        return set.contains(internal);
    }
    bool DT::isUnsigned() const noexcept {
        static const std::unordered_set<Enum> set{
            DT::U8, DT::U16, DT::U32, DT::U64};
        return set.contains(internal);
    }
    bool DT::isNumberic() const noexcept {
        static const std::unordered_set<Enum> set{
            DT::F32, DT::U8, DT::I8, DT::U16, DT::I16,
            DT::I32, DT::I64, DT::FP16, DT::F64,
            DT::U32, DT::U64, DT::BF16};
        return set.contains(internal);
    }
    bool DT::isCpuNumberic() const noexcept {
        static const std::unordered_set<Enum> set{
            DT::F32, DT::U8, DT::I8, DT::U16, DT::I16,
            DT::I32, DT::I64, DT::F64, DT::U32, DT::U64};
        return set.contains(internal);
    }
    bool DT::isBool() const noexcept {
        return internal == DT::Bool;
    }

    size_t DT::size() const noexcept {
#define RETURN_SIZE(TYPE) \
    case DT::TYPE:        \
        return sizeof(primitive<DT::TYPE>::type)

        switch (internal) {
            RETURN_SIZE(F32);
            RETURN_SIZE(U8);
            RETURN_SIZE(I8);
            RETURN_SIZE(U16);
            RETURN_SIZE(I16);
            RETURN_SIZE(I32);
            RETURN_SIZE(I64);
            RETURN_SIZE(Bool);
            RETURN_SIZE(FP16);
            RETURN_SIZE(F64);
            RETURN_SIZE(U32);
            RETURN_SIZE(U64);
            RETURN_SIZE(Complex64);
            RETURN_SIZE(Complex128);
            RETURN_SIZE(BF16);
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor
