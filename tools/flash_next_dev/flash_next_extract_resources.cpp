// Flash-Next M2 — artifact resource extractor (pure mmap, no GPU / VRAM / CUDA
// context).
//
// Opens the real .ninfer artifact, walks objects(), and for every
// ResourceDescriptor (the 6 frontend JSON / Jinja resources) copies its bytes to
// <outdir>/<resource-name>. Prints the identity + object/resource counts + a
// per-resource summary table. Reader::payload is a plain mmap read, so this runs
// while the 27B serve holds VRAM — it is the tokenizer-extraction gate for the
// real-model serve: it lands tokenizer.json / tokenizer_config.json /
// chat_template.jinja / generation_config.json on disk WITHOUT the 27B serve
// having to stop.
//
//   usage: flash_next_extract_resources [artifact.ninfer] [outdir]
// (defaults: the real models/ artifact + ninfer-win/out/flash_next_dev/
//  frontend_resources; argv > env NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS > default).

#include "artifact/reader.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::filesystem::path resolve_path(const char* env, const char* fallback, int argc, int idx,
                                   char** argv) {
  if (argc > idx) return std::filesystem::path(argv[idx]);
  if (env != nullptr && *env) return std::filesystem::path(env);
  return std::filesystem::path(fallback);
}

}  // namespace

int main(int argc, char** argv) {
  // Unbuffered so a crash preserves every progress line (a redirected stdout is
  // block-buffered and otherwise lost on a fastfail).
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv
  const std::filesystem::path artifact =
      resolve_path(std::getenv("NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS"),
                   "C:/Users/Micke/Documents/Ninfer/models/qwen3_8_flash_next.ninfer", argc, 1, argv);
  const std::filesystem::path outdir =
      resolve_path(nullptr,
                   "C:/Users/Micke/Documents/Ninfer/ninfer-win/out/flash_next_dev/frontend_resources",
                   argc, 2, argv);
#pragma warning(pop)

  if (!std::filesystem::exists(artifact)) {
    std::fprintf(stderr, "ARTIFACT_MISSING: %s\n", artifact.string().c_str());
    return 1;
  }
  std::filesystem::create_directories(outdir);

  const ninfer::artifact::Reader reader(artifact);
  const auto& id = reader.identity();
  const auto& objects = reader.objects();

  std::size_t tensor_count = 0;
  std::size_t resource_count = 0;
  std::fprintf(stdout, "artifact : %s\n", artifact.string().c_str());
  std::fprintf(stdout, "model_id : %s\n", id.model_id.c_str());
  std::fprintf(stdout, "weights  : %s\n", id.weights_id.c_str());
  std::fprintf(stdout, "objects  : %zu  (file %llu bytes)\n", objects.size(),
               static_cast<unsigned long long>(reader.file_bytes()));
  std::fprintf(stdout, "dump dir : %s\n\n", outdir.string().c_str());

  for (const auto& object : objects) {
    const auto* res = std::get_if<ninfer::artifact::ResourceDescriptor>(&object);
    if (res == nullptr) {
      ++tensor_count;
      continue;
    }
    ++resource_count;
    const auto span = reader.payload(res->name);
    const std::filesystem::path dest = outdir / res->name;
    std::filesystem::create_directories(dest.parent_path());
    {
      std::ofstream out(dest, std::ios::binary | std::ios::trunc);
      const char* p = reinterpret_cast<const char*>(span.data.data());
      const std::streamsize n = static_cast<std::streamsize>(span.data.size());
      out.write(p, n);
      out.flush();
      if (!out) {
        std::fprintf(stderr, "FAIL writing %s\n", dest.string().c_str());
        return 2;
      }
    }
    std::fprintf(stdout, "  %-46s off=%14llu bytes=%12llu -> %s\n", res->name.c_str(),
                 static_cast<unsigned long long>(span.absolute_offset),
                 static_cast<unsigned long long>(res->bytes), dest.string().c_str());
  }

  std::fprintf(stdout, "\ntensors  : %zu\n", tensor_count);
  std::fprintf(stdout, "resources: %zu\n", resource_count);
  std::fprintf(stdout, "EXTRACT_OK\n");
  return 0;
}
