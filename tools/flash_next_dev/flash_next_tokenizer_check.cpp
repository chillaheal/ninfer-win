// Flash-Next M2 — real-tokenizer verification (pure CPU / mmap, no GPU / VRAM).
//
// Constructs the production qwen3_6::frontend_internal::Tokenizer (compiled into
// ninfer_engine, fully format-driven) from the REAL flash_next artifact's embedded
// resources (frontend/tokenizer.json + tokenizer_config.json + generation_config.json,
// read via Reader::payload -- a plain mmap read, no CUDA context) and proves it on the
// real 248320-vocab Qwen4-Exp BPE: special-token ids, the default stop ids, and
// encode/decode round-trips over English / Swedish / CJK / emoji / a rendered
// chat-template prompt. Runs while the 27B serve holds VRAM -- the de-risking gate
// for the single-lane real serve (the tokenizer is the shared foundation for both
// the single-lane and concurrent paths).
//
//   usage: flash_next_tokenizer_check [artifact.ninfer]
// (default: the real models/ artifact; env NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS).
// Exit 0 = all checks pass; 2 = one or more failures.

#include "targets/qwen3_6/impl/frontend/tokenizer.h"
#include "artifact/reader.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace {
using DecodeOptions = ninfer::targets::qwen3_6::frontend_internal::DecodeOptions;
using Tokenizer = ninfer::targets::qwen3_6::frontend_internal::Tokenizer;
using TokenizerResources = ninfer::targets::qwen3_6::frontend_internal::TokenizerResources;

int g_failures = 0;

void check(bool ok, const char* what) {
  std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) { ++g_failures; }
}

std::string read_resource(const ninfer::artifact::Reader& reader, const char* name) {
  const auto span = reader.payload(name);
  return std::string(reinterpret_cast<const char*>(span.data.data()), span.data.size());
}

void round_trip(const Tokenizer& tok, const char* label, const std::string& text) {
  const std::vector<int> ids = tok.encode(text);
  const std::string back     = tok.decode(ids);
  const bool ok = (back == text);
  std::printf("  [%s] round-trip %-14s (%zu tokens)\n", ok ? "PASS" : "FAIL", label, ids.size());
  if (!ok) {
    std::printf("    in  : %s\n", text.c_str());
    std::printf("    back: %s\n", back.c_str());
    ++g_failures;
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv
  const char* env = std::getenv("NINFER_QWEN3_8_FLASH_NEXT_WEIGHTS");
  const std::filesystem::path artifact =
      (argc > 1) ? std::filesystem::path(argv[1])
                 : ((env != nullptr && *env != '\0') ? std::filesystem::path(env)
                                                     : std::filesystem::path(
                                                           "C:/Users/Micke/Documents/Ninfer/models/"
                                                           "qwen3_8_flash_next.ninfer"));
#pragma warning(pop)
  if (!std::filesystem::exists(artifact)) {
    std::fprintf(stderr, "ARTIFACT_MISSING: %s\n", artifact.string().c_str());
    return 1;
  }
  std::printf("artifact : %s\n", artifact.string().c_str());

  const ninfer::artifact::Reader reader(artifact);
  std::string tokenizer_json, tokenizer_config, generation_config;
  try {
    tokenizer_json    = read_resource(reader, "frontend/tokenizer.json");
    tokenizer_config  = read_resource(reader, "frontend/tokenizer_config.json");
    generation_config = read_resource(reader, "frontend/generation_config.json");
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "RESOURCE_READ_FAIL: %s\n", ex.what());
    return 1;
  }

  std::printf("reading embedded tokenizer resources: tokenizer.json %zu B, config %zu B, "
              "generation %zu B\n",
              tokenizer_json.size(), tokenizer_config.size(), generation_config.size());

  Tokenizer tok(TokenizerResources{tokenizer_json, tokenizer_config, generation_config});
  std::printf("tokenizer constructed OK (248320-vocab Qwen4-Exp BPE)\n\n");

  std::printf("special tokens + stops:\n");
  const std::vector<int> im_start{248045}, im_end{248046}, endoftext{248044};
  check(tok.decode(im_start) == "<|im_start|>", "decode(248045) == <|im_start|>");
  check(tok.decode(im_end) == "<|im_end|>", "decode(248046) == <|im_end|>");
  check(tok.decode(endoftext) == "<|endoftext|>", "decode(248044) == <|endoftext|>");
  check(tok.encode("<|im_end|>") == std::vector<int>{248046}, "encode(<|im_end|>) == [248046]");
  check(tok.is_special_token(248046), "is_special_token(248046)");
  const std::vector<int> stop = tok.default_stop_token_ids();
  check(stop.size() == 2 && stop[0] == 248046 && stop[1] == 248044,
        "default stop ids == [248046, 248044]");

  std::printf("round-trips:\n");
  round_trip(tok, "english", "The quick brown fox jumps over the lazy dog.");
  round_trip(tok, "swedish",
             "Hej! Hur m\u00e5r du idag? \u00c4r du p\u00e5 st\u00e4mning? Jag vill ha en "
             " fungerande modell p\u00e5 100k context.");
  round_trip(tok, "cjk",
             "\u4f60\u597d\uff0c\u4e16\u754c\u3002\u3053\u308c\u306f\u30c6\u30b9\u30c8\u3067\u3059\u3002");
  round_trip(tok, "emoji+punct",
             "\U0001F604\U0001F389 Test 123 \u2014 (brackets) [brackets] \"quotes\" 'apostrophes'");
  const std::string tpl =
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\nHello, how are you?<|im_end|>\n"
      "<|im_start|>assistant\n<think>\n";
  round_trip(tok, "chat-template", tpl);

  // Optional decode of an explicit token stream (argv[2..]) for coherence checks:
  //   flash_next_tokenizer_check.exe <artifact> 3241 10432 ...
  // Purely additive; the TOKENIZER_OK/FAIL verdict below is unchanged by it.
  if (argc > 2) {
    std::vector<int> ids;
    for (int i = 2; i < argc; ++i) {
      char* end = nullptr;
      const long v = std::strtol(argv[i], &end, 10);
      if (end == argv[i]) {
        std::fprintf(stderr, "BAD_TOKEN_ID: '%s'\n", argv[i]);
        return 3;
      }
      ids.push_back(static_cast<int>(v));
    }
    std::printf("\ndecode(%zu tokens) = %s\n", ids.size(), tok.decode(ids).c_str());
  }

  std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "TOKENIZER_OK" : "TOKENIZER_FAIL",
              g_failures, g_failures == 1 ? "" : "s");
  return g_failures == 0 ? 0 : 2;
}
