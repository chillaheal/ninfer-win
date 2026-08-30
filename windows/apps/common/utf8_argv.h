// Windows-only argv re-encoding (port shim, not upstream logic).
//
// The MSVC CRT converts the process command line to narrow argv using the ANSI code page
// (CP_ACP), which corrupts non-ASCII text; the upstream contract is UTF-8 throughout. On
// Windows the apps therefore define wmain and re-encode the wide argv as UTF-8 before
// calling into the shared run_app() logic. Verified empirically: with /utf-8, "på" still
// arrives as 0xE5 (CP1252) in narrow argv, not C3 B5 (UTF-8).

#pragma once

#if defined(_WIN32)

// Pre-define _WINSOCKAPI_ so that <windows.h> does not pull in the legacy <winsock.h>;
// a TU that later uses Winsock 2 (serve via httplib's <winsock2.h>) must not mix both.
#ifndef _WINSOCKAPI_
#define _WINSOCKAPI_
#endif

#include <windows.h>

#include <cstddef>
#include <string>
#include <vector>

namespace ninfer::port {

inline std::vector<std::string> argv_to_utf8(int argc, wchar_t* const* argv) {
    std::vector<std::string> out;
    out.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        const int needed = ::WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string arg(needed > 0 ? static_cast<std::size_t>(needed - 1) : 0, '\0');
        if (needed > 1) {
            ::WideCharToMultiByte(CP_UTF8, 0, argv[i], -1, arg.data(), needed, nullptr, nullptr);
        }
        out.push_back(std::move(arg));
    }
    return out;
}

// Adapts a wide-argv entry point to the narrow (UTF-8) run_app signature used by upstream.
template <typename RunApp>
int run_wmain(int argc, wchar_t** argv, RunApp&& run_app) {
    std::vector<std::string> utf8 = argv_to_utf8(argc, argv);  // non-const: data() must be char*
    std::vector<char*> c_args;
    c_args.reserve(utf8.size());
    for (auto& arg : utf8) { c_args.push_back(arg.data()); }
    return run_app(static_cast<int>(c_args.size()), c_args.data());
}

}  // namespace ninfer::port

#endif  // _WIN32
