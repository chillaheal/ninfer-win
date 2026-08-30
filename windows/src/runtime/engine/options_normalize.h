#pragma once

#include "ninfer/types.h"

namespace ninfer {

// Normalizes EngineOptions context-cache optionals into concrete capacities so downstream
// layout / sequence-planner code can rely on non-null values. Shared by the Engine constructor
// and the lightweight VRAM probe (targets::probe_kv_capacity), which must build a capacity
// curve without constructing an Engine.
EngineOptions normalize_engine_options(EngineOptions options);

} // namespace ninfer
