// Serve options: --system-prompt-file parse + fail-fast file read (Task 1), and the
// /health payload (Task 3 appends its tests below the parse tests).

#include "serve/http_server.h"
#include "serve/serve_options.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer::serve;

using Json = nlohmann::json;

namespace {

int check(bool ok, const char* what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        return 1;
    }
    return 0;
}

ServeOptions parse(const std::vector<std::string>& args) {
    std::vector<std::string> stored = args;
    std::vector<char*> argv;
    for (std::string& s : stored) { argv.push_back(s.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

bool throws_invalid_argument(const std::vector<std::string>& args) {
    try {
        parse(args);
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

int test_parse_system_prompt_file() {
    int failures = 0;
    const std::filesystem::path present =
        std::filesystem::temp_directory_path() / "ninfer_sp_test_present.md";
    const std::filesystem::path whitespace =
        std::filesystem::temp_directory_path() / "ninfer_sp_test_ws.md";
    {
        std::ofstream out(present, std::ios::binary);
        out << "You are a careful engineer.\n";
    }
    {
        std::ofstream out(whitespace, std::ios::binary);
        out << "   \n\t\n";
    }
    const std::string present_arg   = present.generic_string();
    const std::string whitespace_arg = whitespace.generic_string();
    const std::string absent_arg =
        (std::filesystem::temp_directory_path() / "ninfer_sp_test_absent_9x41.md")
            .generic_string();

    // Absent file: valid no-default configuration, parses cleanly.
    const ServeOptions absent = parse({"serve", "m.ninfer", "--system-prompt-file", absent_arg});
    failures += (absent.system_prompt_file == absent_arg) ? 0 : 1; // path stored
    failures += absent.system_prompt_default.empty() ? 0 : 1;      // empty default

    // Present file: content loaded verbatim (trailing whitespace trimmed).
    const ServeOptions loaded = parse({"serve", "m.ninfer", "--system-prompt-file", present_arg});
    failures +=
        (loaded.system_prompt_default == "You are a careful engineer.") ? 0 : 1;

    // Whitespace-only file: valid no-default configuration.
    const ServeOptions ws = parse({"serve", "m.ninfer", "--system-prompt-file", whitespace_arg});
    failures += ws.system_prompt_default.empty() ? 0 : 1;

    // Flag without a value / empty value: hard errors.
    failures += throws_invalid_argument({"serve", "m.ninfer", "--system-prompt-file"}) ? 0 : 1;
    failures += throws_invalid_argument({"serve", "m.ninfer", "--system-prompt-file", ""}) ? 0 : 1;

    std::filesystem::remove(present);
    std::filesystem::remove(whitespace);
    return failures;
}

int test_health_payload() {
    int failures = 0;

    // Reasoning-effort template (the deployed default), active system prompt.
    ninfer::PromptCapabilities caps;
    caps.enable_thinking = true;
    caps.template_sha256_hex =
        "7f0e529032c25183bcd66c7f238da2d377f43be754a94e2725a58c4e16d2ed67";
    caps.reasoning_effort.low            = true;
    caps.reasoning_effort.medium         = true;
    caps.reasoning_effort.xhigh          = true;
    caps.reasoning_effort.default_effort = ninfer::ReasoningEffort::XHigh;
    ServeOptions options;
    options.system_prompt_file    = "C:\\Ninfer\\system-prompt.md";
    options.system_prompt_default = "You are a careful engineer.\n";

    const Json body = Json::parse(HttpServer::health_payload("qwen3.8-27b", caps, options));
    failures += check(body.at("status").get<std::string>() == "ok", "status ok");
    failures += check(body.at("model_id").get<std::string>() == "qwen3.8-27b", "model_id echoed");
    const Json& ct = body.at("chat_template");
    failures += check(ct.at("sha256").get<std::string>() == caps.template_sha256_hex,
                      "template sha256 carried");
    failures += check(ct.at("semantics").get<std::string>() == "reasoning_effort",
                      "semantics name");
    failures += check(ct.at("enable_thinking").get<bool>(), "enable_thinking true");
    const Json& effort = ct.at("reasoning_effort");
    failures += check(effort.at("default").get<std::string>() == "xhigh", "default effort xhigh");
    const Json supported = effort.at("supported");
    failures += check(supported.is_array() && supported.size() == 3 && supported.at(0) == "low" &&
                          supported.at(1) == "medium" && supported.at(2) == "xhigh",
                      "supported list low/medium/xhigh in order");
    const Json& sp = body.at("system_prompt");
    failures += check(sp.at("file").get<std::string>() == options.system_prompt_file,
                      "system_prompt file");
    failures += check(sp.at("active").get<bool>(), "system_prompt active");
    failures += check(
        sp.at("bytes").get<std::size_t>() == options.system_prompt_default.size(),
        "system_prompt bytes");

    // Flag absent: system_prompt omitted entirely.
    ServeOptions no_flag;
    const Json no_sp = Json::parse(HttpServer::health_payload("m", caps, no_flag));
    failures += check(!no_sp.contains("system_prompt"), "system_prompt omitted without flag");

    // Flag present, empty default (absent/empty file): active=false, bytes=0.
    ServeOptions empty_default;
    empty_default.system_prompt_file = "empty.md";
    const Json empty_sp = Json::parse(HttpServer::health_payload("m", caps, empty_default));
    failures += check(empty_sp.at("system_prompt").at("active").get<bool>() == false &&
                          empty_sp.at("system_prompt").at("bytes").get<std::size_t>() == 0,
                      "empty default: active=false bytes=0");

    // Thinking-toggle template: semantics name, no reasoning_effort block.
    ninfer::PromptCapabilities toggle;
    toggle.enable_thinking     = true;
    toggle.template_sha256_hex =
        "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259";
    const Json toggle_body = Json::parse(HttpServer::health_payload("m", toggle, no_flag));
    failures += check(
        toggle_body.at("chat_template").at("semantics").get<std::string>() == "thinking_toggle",
        "toggle semantics name");
    failures += check(!toggle_body.at("chat_template").contains("reasoning_effort"),
                      "toggle: no reasoning_effort block");
    return failures;
}
} // namespace

int main() {
    int failures = test_parse_system_prompt_file();
    failures += test_health_payload();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
