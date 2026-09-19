#include "artifact/v3/layouts.h"

#include "artifact/v3/formats.h"
#include "artifact/v3/schema.h"

namespace ninfer::artifact::v3 {

WeightGeometry describe_tensor(const TensorObject& object) {
    try {
        auto geometry =
            weight_geometry(parse_format(object.format), parse_layout(object.layout), object.shape);
        if (geometry.bytes != object.bytes || object.offset % geometry.alignment) {
            throw ArtifactError("encoded size or object alignment differs from layout");
        }
        return geometry;
    } catch (const std::exception& error) { throw ArtifactError(object.id + ": " + error.what()); }
}

} // namespace ninfer::artifact::v3
