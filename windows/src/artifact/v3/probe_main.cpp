// Standalone inspector for .ninfer V3 artifacts (magic NINFER\0\3).
// Dumps the parsed manifest summary and per-tensor geometry without materializing weights,
// so it verifies the framing/manifest parse against a real artifact in a couple of seconds.
//
// Optional: --dump-bindings <path> additionally writes every binding (name -> parts with
// object id/format/layout/shape + begin/end) plus every use (parameter -> input) to a file,
// so the v3 bind can be authored from the real manifest instead of guessed names.
#include "artifact/v3/binder.h"
#include "artifact/v3/reader.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

namespace {

using namespace ninfer::artifact::v3;

void print_hex(const std::byte* p, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        char buf[3];
        std::snprintf(buf, sizeof buf, "%02x", static_cast<unsigned int>(std::to_integer<unsigned char>(p[i])));
        std::cout << buf;
    }
}

std::string describe_part(const Directory& dir, const Part& part) {
    if (part.object.index >= dir.objects.size()) return "<?>";
    const Object& obj = dir.objects[part.object.index];
    std::string s;
    if (const auto* t = std::get_if<TensorObject>(&obj)) {
        s = t->id + " fmt=" + t->format + "/" + t->layout + "[";
        for (std::uint64_t d : t->shape) {
            s += std::to_string(d) + ",";
        }
        s += "]";
    } else if (const auto* r = std::get_if<ResourceObject>(&obj)) {
        s = r->id + " enc=" + r->encoding;
    } else {
        s = "<object>";
    }
    return s + " [" + std::to_string(part.begin) + ":" + std::to_string(part.end) + "]";
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: ninfer_v3_probe <model.ninfer> [--dump-bindings <path>]\n";
        return 2;
    }
    std::string dump_path;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump-bindings") == 0 && i + 1 < argc) {
            dump_path = argv[++i];
        }
    }
    using namespace ninfer::artifact::v3;
    try {
        Reader reader(argv[1]);
        const Directory& dir = reader.directory();
        std::cout << "file_bytes=" << reader.file_bytes() << "\n";
        std::cout << "payload_bytes=" << dir.payload_bytes
                  << " file_records=" << dir.files.size() << "\n";
        std::cout << "objects=" << dir.objects.size()
                  << " components=" << dir.components.size()
                  << " bindings=" << dir.bindings.size() << "\n";

        std::cout << "artifact_id=";
        const auto& id = reader.artifact_id();
        print_hex(id.data(), id.size());
        std::cout << "\n";

        if (dir.metadata.is_object()) {
            std::cout << "metadata:";
            for (const auto& entry : dir.metadata.items()) {
                std::cout << " " << entry.key() << "="
                    << (entry.value().is_string() ? entry.value().get<std::string>() : entry.value().dump());
            }
            std::cout << "\n";
        }

        std::size_t tensors = 0, resources = 0, object_bytes_total = 0;
        for (const auto& obj : dir.objects) {
            if (const auto* t = std::get_if<TensorObject>(&obj)) {
                ++tensors;
                object_bytes_total += t->bytes;
            } else {
                ++resources;
            }
        }
        std::cout << "tensors=" << tensors << " resources=" << resources
                  << " object_bytes_total=" << object_bytes_total << "\n";

        std::cout << "components:";
        for (const auto& comp : dir.components) std::cout << ' ' << comp.first;
        std::cout << "\n";

        std::cout << "component resources:\n";
        for (const auto& [name, comp] : dir.components) {
            std::cout << "  [" << name << "]";
            if (comp.target) std::cout << " target=" << *comp.target;
            if (!comp.resources.empty()) std::cout << " resources=" << comp.resources.size();
            std::cout << "\n";
            for (const auto& [role, handle] : comp.resources) {
                std::string desc = "<missing>";
                if (handle.index < dir.objects.size()) {
                    if (const auto* r = std::get_if<ResourceObject>(&dir.objects[handle.index])) {
                        desc = r->id + " enc=" + r->encoding;
                    } else if (const auto* t = std::get_if<TensorObject>(&dir.objects[handle.index])) {
                        desc = t->id + " (tensor)";
                    }
                }
                std::cout << "    " << role << " -> idx" << handle.index << " " << desc << "\n";
            }
        }

        std::size_t shown = 0;
        std::cout << "tensors (first 12):\n";
        for (const auto& obj : dir.objects) {
            const auto* t = std::get_if<TensorObject>(&obj);
            if (!t) {
                continue;
            }
            std::cout << "  " << t->id << " fmt=" << t->format << "/" << t->layout
                   << " bytes=" << t->bytes << " offset=" << t->offset;
            for (auto d : t->shape) std::cout << "[" << d << "]";
            std::cout << "\n";
            if (++shown >= 12) {
                break;
            }
        }

        std::size_t whole = 0, multi = 0, mx_parts = 0;
        for (const auto& kv : dir.bindings) {
            if (kv.second.parts.size() > 1) {
                ++multi;
                mx_parts = std::max(mx_parts, kv.second.parts.size());
            } else {
                ++whole;
            }
        }
        std::cout << "bindings_whole=" << whole << " multi=" << multi
                  << " max_parts=" << mx_parts << "\n";

        int nb = 0;
        std::cout << "bindings (first 25):\n";
        for (const auto& kv : dir.bindings) {
            const Binding& b = kv.second;
            std::cout << "  " << kv.first;
            if (b.whole_object) std::cout << " <whole>";
            for (const auto& part : b.parts) {
                std::cout << " | " << describe_part(dir, part);
            }
            std::cout << "\n";
            if (++nb >= 25) {
                break;
            }
        }

        if (!dump_path.empty()) {
            std::ofstream out(dump_path);
            if (!out) {
                std::cerr << "cannot open " << dump_path << " for writing\n";
                return 1;
            }
            out << "# v3 binding dump: " << argv[1] << "\n";
            out << "total_bindings=" << dir.bindings.size() << "\n";
            for (const auto& kv : dir.bindings) {
                const Binding& b = kv.second;
                out << "BINDING " << kv.first;
                if (b.whole_object) out << " <whole>";
                out << " elements=" << b.elements << "\n";
                for (const auto& part : b.parts) {
                    out << "  part " << part.object.index << " " << describe_part(dir, part) << "\n";
                }
            }
            for (const auto& kv : dir.uses) {
                const Use& u = kv.second;
                out << "USE " << kv.first.first << "/" << kv.first.second
                    << " param=" << u.parameter << " input=" << u.input;
                if (!u.auxiliaries.empty()) {
                    out << " aux={";
                    bool first = true;
                    for (const auto& au : u.auxiliaries) {
                        if (!first) out << ",";
                        first = false;
                        out << au.first << "(" << au.second.parts.size() << ")";
                }
                out << "}";
                }
                out << "\n";
            }
            out << "\n# uses end\n";
            out.flush();
            std::cout << "dumped " << dir.bindings.size() << " bindings to " << dump_path << "\n";
        }

        std::cout << "PROBE_OK\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PROBE_ERROR: " << e.what() << "\n";
        return 1;
    }
}
