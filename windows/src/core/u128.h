#pragma once

#include <cstdint>

namespace ninfer {

// Portable unsigned 128-bit integer for overflow-checked arithmetic. GCC/Clang use the native
// type; MSVC (x64) has no __int128, so a split 64x64 emulation provides identical semantics.
// Only the operations used by the runtime cost model are implemented: u64 construction, +, -, *,
// /u64, <<, >>, ~, comparisons, and narrowing conversion back to u64 (the caller must have
// verified the value fits). Division by zero is undefined, as with the native type.
#if defined(__SIZEOF_INT128__)

using U128 = unsigned __int128;

#else

struct U128 {
    std::uint64_t hi = 0;
    std::uint64_t lo = 0;

    constexpr U128() noexcept            = default;
    constexpr U128(std::uint64_t value) noexcept : hi(0), lo(value) {}
    // Declaring constructors removes aggregate status, so the {hi, lo} brace form used throughout
    // the operators needs this constructor too.
    constexpr U128(std::uint64_t hi_value, std::uint64_t lo_value) noexcept
        : hi(hi_value), lo(lo_value) {}

    [[nodiscard]] explicit constexpr operator std::uint64_t() const noexcept { return lo; }

    friend constexpr U128 operator~(U128 value) noexcept { return U128{~value.hi, ~value.lo}; }

    friend constexpr U128 operator+(U128 left, U128 right) noexcept {
        const std::uint64_t lo = left.lo + right.lo;
        return U128{left.hi + right.hi + (lo < left.lo ? 1ULL : 0ULL), lo};
    }

    friend constexpr U128 operator-(U128 left, U128 right) noexcept {
        const std::uint64_t lo = left.lo - right.lo;
        return U128{left.hi - right.hi - (lo > left.lo ? 1ULL : 0ULL), lo};
    }

    // Low 128 bits of the product. Exact for operands below 2^64, the only domain used here.
    friend constexpr U128 operator*(U128 left, U128 right) noexcept {
        const U128 low     = mul64(left.lo, right.lo);
        const U128 cross_a = mul64(left.hi, right.lo);
        const U128 cross_b = mul64(left.lo, right.hi);
        return U128{low.hi + cross_a.lo + cross_b.lo, low.lo};
    }

    // 128/64 division; the quotient fits in u64 and is returned as a U128 so callers keep the
    // full operand type through chained expressions.
    friend constexpr U128 operator/(U128 dividend, std::uint64_t divisor) noexcept {
        std::uint64_t quotient  = 0;
        std::uint64_t remainder = 0; // invariant: remainder < divisor
        for (int bit_index = 127; bit_index >= 0; --bit_index) {
            const std::uint64_t bit =
                (bit_index >= 64) ? ((dividend.hi >> (bit_index - 64)) & 1ULL)
                                  : ((dividend.lo >> bit_index) & 1ULL);
            if ((remainder >> 63) != 0) {
                // 2*remainder + bit >= 2^64 > divisor: quotient bit set, and the new remainder is
                // (2^64 + low_bits - divisor), which u64 subtraction wraps to exactly.
                remainder = ((remainder << 1) | bit) - divisor;
                quotient  = (quotient << 1) | 1ULL;
            } else {
                const std::uint64_t value = (remainder << 1) | bit;
                if (value >= divisor) {
                    remainder = value - divisor;
                    quotient  = (quotient << 1) | 1ULL;
                } else {
                    remainder = value;
                    quotient <<= 1;
                }
            }
        }
        return U128{0, quotient};
    }

    friend constexpr U128 operator<<(U128 value, unsigned shift) noexcept {
        if (shift >= 128U) { return U128{}; }
        if (shift == 0U) { return value; }
        if (shift >= 64U) { return U128{value.lo << (shift - 64U), 0}; }
        return U128{(value.hi << shift) | (value.lo >> (64U - shift)), value.lo << shift};
    }

    friend constexpr U128 operator>>(U128 value, unsigned shift) noexcept {
        if (shift >= 128U) { return U128{}; }
        if (shift == 0U) { return value; }
        if (shift >= 64U) { return U128{0, value.hi >> (shift - 64U)}; }
        return U128{value.hi >> shift, (value.lo >> shift) | (value.hi << (64U - shift))};
    }

    [[nodiscard]] friend constexpr bool operator==(U128 left, U128 right) noexcept {
        return left.hi == right.hi && left.lo == right.lo;
    }
    [[nodiscard]] friend constexpr bool operator!=(U128 left, U128 right) noexcept {
        return !(left == right);
    }
    [[nodiscard]] friend constexpr bool operator<(U128 left, U128 right) noexcept {
        return left.hi != right.hi ? left.hi < right.hi : left.lo < right.lo;
    }
    [[nodiscard]] friend constexpr bool operator>(U128 left, U128 right) noexcept {
        return right < left;
    }
    [[nodiscard]] friend constexpr bool operator<=(U128 left, U128 right) noexcept {
        return !(right < left);
    }
    [[nodiscard]] friend constexpr bool operator>=(U128 left, U128 right) noexcept {
        return !(left < right);
    }

private:
    // Full 64x64 -> 128 schoolbook multiply.
    static constexpr U128 mul64(std::uint64_t left, std::uint64_t right) noexcept {
        // Block-scope `static` storage duration is not allowed in a C++20 constexpr body
        // (MSVC enforces [dcl.constexpr] with C3615; GCC/Clang accept it as an extension).
        constexpr std::uint64_t low32 = 0xFFFFFFFFULL;
        const std::uint64_t lo_l = left & low32;
        const std::uint64_t hi_l = left >> 32;
        const std::uint64_t lo_r = right & low32;
        const std::uint64_t hi_r = right >> 32;
        const std::uint64_t p0   = lo_l * lo_r;
        const std::uint64_t p1   = hi_l * lo_r;
        const std::uint64_t p2   = lo_l * hi_r;
        const std::uint64_t p3   = hi_l * hi_r;
        const std::uint64_t x1   = (p1 & low32) << 32;
        const std::uint64_t t1   = p0 + x1;
        std::uint64_t carry      = (t1 < p0) ? 1ULL : 0ULL;
        const std::uint64_t x2   = (p2 & low32) << 32;
        const std::uint64_t lo   = t1 + x2;
        carry                   += (lo < t1) ? 1ULL : 0ULL;
        return U128{p3 + (p1 >> 32) + (p2 >> 32) + carry, lo};
    }
};

#endif

} // namespace ninfer
