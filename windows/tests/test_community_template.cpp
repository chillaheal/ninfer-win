// The community chat template (D7b) resolves to the ReasoningEffort semantics and carries
// its digest hex out through PromptCapabilities (Task 3 plumbing).

#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include <fstream>
#include <iostream>
#include <istream>
#include <stdexcept>
#include <string>
#include <iterator>

namespace {

using namespace ninfer::targets::qwen3_6::frontend_internal;

const char* kExpectedSha =
    "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67";

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

int test_community_template_digest() {
    int failures = 0;
    const std::string source = read_file(
        std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/frontend/community_chat_template.jinja");
    failures += (source.size() == 7816) ? 0 : 1;

    const CompiledChatTemplate compiled = CompiledChatTemplate::resolve(source);
    const ninfer::PromptCapabilities caps = compiled.capabilities();
    failures += (caps.template_sha256_hex == kExpectedSha) ? 0 : 1;
    failures += caps.reasoning_effort.low ? 0 : 1;
    failures += caps.reasoning_effort.medium ? 0 : 1;
    failures += caps.reasoning_effort.xhigh ? 0 : 1;
    failures += (caps.reasoning_effort.default_effort == ninfer::ReasoningEffort::XHigh) ? 0 : 1;

    // Any byte change to the template source (e.g. a trailing newline) must break resolution.
    bool threw = false;
    try {
        (void) CompiledChatTemplate::resolve(source + "\n");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    failures += threw ? 0 : 1;
    return failures;
}
} // namespace

int main() {
    int failures = 0;
    try {
        failures = test_community_template_digest();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
