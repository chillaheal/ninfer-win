// Flash-Next P9 — PressurePlanningSession throw-stubs.
//
// The header declares the exact signatures because the generic
// runtime::MaterializationPlanner / ResourceManager templates (instantiated in
// engine.cpp for this family) reference them. For the v1 mini forward these
// branches are unreachable — the identity/pressure path is never taken — so the
// bodies throw by design (family.h "v1-unreachable; exact signatures, bodies
// throw"). The HONESTY BLOCKER: real pressure modeling is unrecoverable; this
// is a v1 contract, not an implementation.

#include "targets/qwen3_8_flash_next/impl/runtime/family.h"

#include <optional>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next {

namespace {
[[noreturn]] void pressure_unreachable() {
    throw std::logic_error("flash_next v1: pressure planning is unreachable (throw-stub)");
}
}  // namespace

PressurePlanningSession::~PressurePlanningSession() = default;

PressureTargetHandle
PressurePlanningSession::identity_target(const AdmissionCandidate&) const {
    pressure_unreachable();
}

PressureTargetHandle
PressurePlanningSession::root_maximal_target(const AdmissionCandidate&) {
    pressure_unreachable();
}

runtime::PressureTargetAssessment
PressurePlanningSession::assess(PressureTargetHandle) const {
    pressure_unreachable();
}

PreparedPressureExpansion
PressurePlanningSession::prepare_expansion(PressureTargetHandle) {
    pressure_unreachable();
}

PressureExpansionView
PressurePlanningSession::commit_expansion(PreparedPressureExpansion&&) {
    pressure_unreachable();
}

void PressurePlanningSession::discard_expansion(PreparedPressureExpansion&&) noexcept {
    // noexcept: cannot throw. v1-unreachable; there is nothing to release.
}

std::optional<ResourcePlan>
PressurePlanningSession::seal(PressureTargetHandle, const PreparedPrompt&) {
    pressure_unreachable();
}

}  // namespace ninfer::targets::qwen3_8_flash_next
