#pragma once

#include "artifact/v3/binder.h"
#include "artifact/v3/materializer.h"
#include "core/weight_view.h"

namespace ninfer::artifact::v3 {

[[nodiscard]] WeightView bind_view(const ParameterReference& reference,
                                   const MaterializedArtifact& materialized);

} // namespace ninfer::artifact::v3
