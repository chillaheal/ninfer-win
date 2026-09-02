#pragma once

#include "artifact/binder.h"
#include "core/tensor.h"

#include <cstdint>
#include <initializer_list>
#include <string_view>

namespace ninfer::artifact {

class MaterializedArtifact;

[[nodiscard]] ObjectHandle bind_tensor(Binder& binder, std::string_view name, NumericFormat format,
                                       std::initializer_list<std::uint64_t> shape,
                                       TensorPlacement placement);

[[nodiscard]] ObjectHandle bind_device_tensor(Binder& binder, std::string_view name,
                                              NumericFormat format,
                                              std::initializer_list<std::uint64_t> shape);

[[nodiscard]] ObjectHandle bind_raw_resource(Binder& binder, std::string_view name);

[[nodiscard]] Tensor materialized_tensor(const MaterializedArtifact& materialized,
                                         ObjectHandle handle, NumericFormat format,
                                         std::initializer_list<std::int32_t> internal_shape);

[[nodiscard]] Weight materialized_weight(const MaterializedArtifact& materialized,
                                         ObjectHandle handle, NumericFormat format,
                                         std::int32_t rows, std::int32_t columns);

// Host-resident variants: the Weight/Tensor point into the artifact's retained host bytes
// (resource_bytes) instead of device memory. Used by the CPU vision path, whose weights are
// placed Host and read by CPU compute rather than materialized to the device.
[[nodiscard]] Tensor materialized_host_tensor(const MaterializedArtifact& materialized,
                                             ObjectHandle handle, NumericFormat format,
                                             std::initializer_list<std::int32_t> internal_shape);
[[nodiscard]] Weight materialized_host_weight(const MaterializedArtifact& materialized,
                                              ObjectHandle handle, NumericFormat format,
                                              std::int32_t rows, std::int32_t columns);

} // namespace ninfer::artifact
