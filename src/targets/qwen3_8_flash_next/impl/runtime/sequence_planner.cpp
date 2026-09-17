// Flash-Next P9 — sequence planner + capacity curve (v1).
//
// One QSA KV page per main page group (64 tokens x 2 KV heads x 32 dims x {K,V} x BF16
// = 16,384 B). The per-sequence persistent state (GDN recurrent state + causal-conv1d
// state) and the one-shot round scratch live in the Program-owned workspace arena, so
// the curve reservation stays exactly affine in page groups.

#include "targets/qwen3_8_flash_next/impl/runtime/family.h"

#include "ninfer/ops/sparse_moe_nvfp4.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next {

namespace {

constexpr std::uint32_t kPagedKVPageSize = 64;
constexpr std::size_t kQsaKvPageBytes =
    static_cast<std::size_t>(kPagedKVPageSize) * 2U * 32U * 2U * 2U; // 16,384
// 3 GDN layers x [128,128,Hv=4] FP32 state (786,432 B) + 3 conv states x
// [256,3] BF16 (4,608 B) = 791,040 B per sequence.
constexpr std::size_t kPersistentBytesPerSeq =
    3U * 128U * 128U * 4U * sizeof(float) + 3U * 256U * 3U * 2U;
// One-shot round scratch for the largest v1 round (T <= 64).
constexpr std::size_t kRoundScratchBytes = 2U * 1024U * 1024U;
// Mini MoE geometry (E=8, K=2, K_in=256, I=64) — inside the sparse_moe_nvfp4 v1 domain.
constexpr ops::SparseMoeNvfp4Geometry kMoeGeometry{8, 2, 256, 64};
constexpr int kMaxRoundTokens = 64;

std::size_t compute_workspace_capacity(std::uint32_t max_concurrency) {
    const auto moe_workspace = ops::sparse_moe_nvfp4_workspace_capacity_bytes(
        kMoeGeometry, 1, kMaxRoundTokens);
    return static_cast<std::size_t>(max_concurrency) * kPersistentBytesPerSeq +
           kRoundScratchBytes + moe_workspace;
}

}  // namespace

struct detail::SequencePlanImpl {
    std::uint32_t capacity               = 0;
    std::uint32_t kv_capacity            = 0;
    std::uint32_t max_concurrency        = 0;
    std::size_t device_reservation_bytes = 0;
    std::size_t workspace_capacity       = 0;
};

struct detail::SequencePlannerImpl {
    runtime::SequenceCapacityCurve curve{};
    std::uint32_t capacity             = 0;
    std::uint32_t max_concurrency      = 0;
    std::size_t workspace_capacity     = 0;
};

SequencePlan::SequencePlan(std::unique_ptr<detail::SequencePlanImpl> impl) noexcept
    : impl_(std::move(impl)) {}

SequencePlan::~SequencePlan() = default;

// Move ctor/assign are defined here (not inline-defaulted in family.h) because
// the defaulted definition instantiates ~unique_ptr<SequencePlanImpl>, which
// requires the complete type — only available in this TU (family.h forward-
// declares it). Any TU that moves a SequencePlan by value (e.g. registry.cpp's
// planner.finalize(...)) odr-uses these out-of-line definitions.
SequencePlan::SequencePlan(SequencePlan&&) noexcept = default;
SequencePlan& SequencePlan::operator=(SequencePlan&&) noexcept = default;

std::uint32_t SequencePlan::capacity() const {
    if (!impl_) {
        throw std::logic_error("flash_next SequencePlan is empty");
    }
    return impl_->capacity;
}

std::uint32_t SequencePlan::kv_capacity() const {
    if (!impl_) {
        throw std::logic_error("flash_next SequencePlan is empty");
    }
    return impl_->kv_capacity;
}

std::uint32_t SequencePlan::max_concurrency() const {
    if (!impl_) {
        throw std::logic_error("flash_next SequencePlan is empty");
    }
    return impl_->max_concurrency;
}

std::size_t SequencePlan::device_reservation_bytes() const {
    if (!impl_) {
        throw std::logic_error("flash_next SequencePlan is empty");
    }
    return impl_->device_reservation_bytes;
}

std::size_t SequencePlan::workspace_capacity_bytes() const {
    if (!impl_) {
        throw std::logic_error("flash_next SequencePlan is empty");
    }
    return impl_->workspace_capacity;
}

SequencePlanner::SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl> impl) noexcept
    : impl_(std::move(impl)) {}

SequencePlanner::~SequencePlanner() = default;

const runtime::SequenceCapacityCurve& SequencePlanner::capacity_curve() const {
    if (!impl_) {
        throw std::logic_error("flash_next SequencePlanner is empty");
    }
    return impl_->curve;
}

SequencePlan SequencePlanner::finalize(std::uint32_t main_page_groups) && {
    if (!impl_) {
        throw std::logic_error("flash_next SequencePlanner is empty");
    }
    const auto& curve = impl_->curve;
    if (main_page_groups < curve.minimum_main_page_groups ||
        main_page_groups > curve.maximum_main_page_groups) {
        throw std::logic_error("flash_next main page groups outside the capacity curve");
    }
    auto plan = std::make_unique<detail::SequencePlanImpl>();
    plan->capacity                 = impl_->capacity;
    plan->kv_capacity              = curve.resolved_tokens(main_page_groups);
    plan->max_concurrency          = impl_->max_concurrency;
    plan->device_reservation_bytes = curve.reservation_bytes(main_page_groups);
    plan->workspace_capacity       = impl_->workspace_capacity;
    return SequencePlan(std::move(plan));
}

SequencePlanner make_sequence_planner(DeviceContext& device, const EngineOptions& options,
                                      WeightsProfile) {
    (void)device;
    if (options.max_context == 0 || options.max_concurrency == 0) {
        throw std::invalid_argument(
            "flash_next planner requires non-zero max_context and max_concurrency");
    }
    auto impl = std::make_unique<detail::SequencePlannerImpl>();
    impl->capacity             = options.max_context;
    impl->max_concurrency      = options.max_concurrency;
    impl->workspace_capacity   = compute_workspace_capacity(options.max_concurrency);
    const auto logical_pages = (options.max_context + kPagedKVPageSize - 1) / kPagedKVPageSize;
    impl->curve.main_page_tokens = kPagedKVPageSize;
    impl->curve.minimum_main_page_groups = std::max(logical_pages, options.max_concurrency);
    impl->curve.maximum_main_page_groups = logical_pages * options.max_concurrency;
    impl->curve.minimum_device_reservation_bytes =
        static_cast<std::size_t>(impl->curve.minimum_main_page_groups) * kQsaKvPageBytes;
    impl->curve.bytes_per_additional_main_page_group = kQsaKvPageBytes;
    return SequencePlanner(std::move(impl));
}

}  // namespace ninfer::targets::qwen3_8_flash_next
