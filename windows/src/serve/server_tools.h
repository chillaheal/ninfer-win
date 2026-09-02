#pragma once

#include <string>
#include <string_view>

namespace ninfer::serve {

// Executes the emulated Anthropic server-side tools. Every entry point returns
// human-readable text and never throws: on failure the error text is fed back
// to the model as the tool result so it can react (retry, answer without
// results, ...).
std::string run_web_search(std::string_view query);
std::string run_web_fetch(std::string_view url);

} // namespace ninfer::serve
