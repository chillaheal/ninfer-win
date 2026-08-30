// Ninfer.exe — native Win32 GUI: serve launcher + System-prompt / Chat-template
// editors (spec D3/D4/D5).
//
// Serve launches ninfer-serve.exe in its own console window (CREATE_NEW_CONSOLE)
// so the server's live output — "listening on http://host:port", per-request
// results, 5-second prefill/decode tok/s stats — stays visible like a terminal
// (unsloth-style). The UI thread never blocks on the child: a watcher thread
// reaps the process and posts WM_APP_DONE. Exit (and the window X) terminate
// the child and every other ninfer engine process so no orphaned GPU work
// lingers. The two editor windows (D4/D5) are non-modal top-level windows that
// reuse this window class, routed by a creation tag (EditorInfo in lpParam).
// Zero dependencies beyond Win32 + CRT + winhttp (+ header-only nlohmann for
// the /health JSON).
//
// Command-line prefill (used by the automated GUI e2e, .logs/gui_serve_e2e.ps1):
//   Ninfer.exe <model.ninfer> [--max-new N] [--max-ctx N]
// The first non-flag argument fills the model field; --max-new/--max-ctx fill
// their edits. (The old prompt-words prefill was removed with the one-shot Run
// mode.)

#include <windows.h>
#include <commctrl.h>
#include <shellapi.h>
#include <tlhelp32.h>
#include <winhttp.h>

#include <nlohmann/json.hpp>

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <istream>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <wchar.h>

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "comctl32.lib")
#pragma comment(lib, "comdlg32.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "winhttp.lib")

namespace {

// ---------------------------------------------------------------------------
// Control IDs and messages
// ---------------------------------------------------------------------------

enum : UINT {
    IDC_MODEL_EDIT = 101,
    IDC_MODEL_BROWSE,             // 102
    IDC_MESSAGES_EDIT_RETIRED,    // 103 retired (one-shot Run mode removed)
    IDC_MESSAGES_BROWSE_RETIRED,  // 104 retired
    IDC_PROMPT_EDIT_RETIRED,      // 105 retired
    IDC_MAX_CONTEXT_EDIT,         // 106
    IDC_MAX_NEW_EDIT,             // 107
    IDC_KV_DTYPE_COMBO,           // 108
    IDC_TEMPERATURE_EDIT,         // 109
    IDC_TOP_P_EDIT,               // 110
    IDC_TOP_K_EDIT,               // 111
    IDC_GREEDY_CHECK,             // 112
    IDC_THINKING_CHECK,           // 113
    IDC_VISION_CHECK,             // 114
    IDC_SPEC_COMBO,               // 115
    IDC_DRAFT_TOKENS_EDIT,        // 116
    IDC_SEED_EDIT,                // 117
    IDC_RUN_BUTTON_RETIRED,       // 118 retired
    IDC_CANCEL_BUTTON_RETIRED,    // 119 retired
    IDC_EXIT_BUTTON,              // 120
    IDC_ANSWER_EDIT_RETIRED,      // 121 retired
    IDC_LOG_EDIT_RETIRED,         // 122 retired
    IDC_STATUS_TEXT,              // 123
    // The historical control IDs (hardcoded in the automated e2e) stay stable
    // from here on; new IDs are appended after 128.
    IDC_HOST_EDIT = 124,
    IDC_PORT_EDIT,                // 125
    IDC_SERVE_BUTTON,             // 126
    IDC_LM_HEAD_DRAFT_CHECK,      // 127
    IDC_AUTO_BUTTON,              // 128 — fills Max context with the probed VRAM fit
    // New (D3/D4/D5):
    IDC_SYSTEM_PROMPT_BUTTON,     // 129
    IDC_TEMPLATE_BUTTON,          // 130
    IDC_SP_EDIT,                  // 131
    IDC_SP_SAVE_BUTTON,           // 132
    IDC_SP_STATUS,                // 133
    IDC_CT_EDIT,                  // 134
    IDC_CT_REFRESH_BUTTON,        // 135
    IDC_CT_OPEN_BUTTON,           // 136
    IDC_CT_SAVE_BUTTON,           // 137
    IDC_CT_STATUS_LOADED,         // 138
    IDC_CT_STATUS_EDITOR,         // 139
    IDC_CT_STATUS_MATCH,          // 140
    // Follow-up round (2026-08-30):
    IDC_STOP_BUTTON,              // 141 — stop the serve this GUI launched; the GUI stays open
};

constexpr UINT WM_APP_DONE = WM_APP + 1;  // wParam: serve exit code (main window only)

constexpr wchar_t kWindowClass[] = L"NinferGuiWindow";

// Per-window routing tag (set from the CreateWindowExW lpParam into GWLP_USERDATA
// at WM_CREATE): nullptr = main window; kind 1 = system-prompt editor; kind 2 =
// chat-template editor (host/port remembered for /health refreshes).
struct EditorInfo {
    int kind = 0;
    std::wstring host;
    std::wstring port;
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

std::wstring utf8_to_wide(std::string_view bytes) {
    if (bytes.empty()) { return {}; }
    const int needed = ::MultiByteToWideChar(CP_UTF8, 0, bytes.data(), static_cast<int>(bytes.size()),
                                             nullptr, 0);
    std::wstring out(static_cast<std::size_t>(needed), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, bytes.data(), static_cast<int>(bytes.size()), out.data(),
                          needed);
    return out;
}

std::string wide_to_utf8(std::wstring_view text) {
    if (text.empty()) { return {}; }
    const int needed =
        ::WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0,
                              nullptr, nullptr);
    std::string out(static_cast<std::size_t>(needed), '\0');
    ::WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), out.data(), needed,
                          nullptr, nullptr);
    return out;
}

// The EDIT controls on this machine store LF-only text but do NOT render the
// line breaks (captured: .logs/sp_repro_edit.png — the whole file shows as one
// continuous line), while CRLF line breaks do render (sp_repro_crlf_edit.png).
// So the display form is CRLF and the on-disk form stays canonical LF-only:
// normalize on the way into the control, denormalize before writing or hashing.
std::string to_display_lf_crlf(std::string_view bytes) {
    std::string out;
    out.reserve(bytes.size());
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        if (bytes[i] == '\r') {
            if (i + 1 < bytes.size() && bytes[i + 1] == '\n') { ++i; }
            out += "\r\n";
        } else if (bytes[i] == '\n') {
            out += "\r\n";
        } else {
            out += bytes[i];
        }
    }
    return out;
}

std::string to_canonical_crlf_lf(std::string_view bytes) {
    std::string out;
    out.reserve(bytes.size());
    for (const char c : bytes) {
        if (c != '\r') { out += c; }
    }
    return out;
}

// Declared here (defined below, after the SHA-256 section): serve_child's
// pre-launch port-busy check calls it before its definition.
std::string http_get(const std::wstring& host, const std::wstring& port, const std::string& path);

std::wstring get_control_text(HWND hwnd) {
    const int len = ::GetWindowTextLengthW(hwnd);
    std::wstring out(static_cast<std::size_t>(len), L'\0');
    if (len > 0) { ::GetWindowTextW(hwnd, out.data(), len + 1); }
    return out;
}

// Quote one argument for a Windows command line: wrap in quotes when it
// contains whitespace or a quote; escape embedded quotes per CreateProcess rules.
std::wstring quote_arg(std::wstring_view arg) {
    if (!arg.empty() && arg.find_first_of(L" \t\"") == std::wstring_view::npos) {
        return std::wstring(arg);
    }
    std::wstring out(L"\"");
    unsigned backslashes = 0;
    for (wchar_t ch : arg) {
        if (ch == L'\\') {
            ++backslashes;
            continue;
        }
        if (ch == L'"') {
            out.append(backslashes * 2 + 1, L'\\');
            out.push_back(L'"');
            backslashes = 0;
            continue;
        }
        out.append(backslashes, L'\\');
        backslashes = 0;
        out.push_back(ch);
    }
    out.append(backslashes * 2, L'\\');
    out.push_back(L'"');
    return out;
}

std::wstring browse_for_file(HWND owner, std::wstring title, std::wstring filter,
                             std::wstring initial_dir = {}) {
    wchar_t buffer[MAX_PATH] {};
    OPENFILENAMEW ofn {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner   = owner;
    ofn.lpstrFilter = filter.data();
    ofn.nMaxFile    = std::size(buffer);
    ofn.lpstrFile   = buffer;
    ofn.lpstrTitle  = title.data();
    ofn.Flags       = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    if (!initial_dir.empty()) {
        ofn.lpstrInitialDir = initial_dir.c_str();
        ofn.Flags          |= OFN_EXPLORER;
    }
    return ::GetOpenFileNameW(&ofn) ? std::wstring(buffer) : std::wstring{};
}

std::wstring browse_save_file(HWND owner, std::wstring title, std::wstring filter,
                              std::wstring initial_dir = {}) {
    wchar_t buffer[MAX_PATH] {};
    OPENFILENAMEW ofn {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner   = owner;
    ofn.lpstrFilter = filter.data();
    ofn.nMaxFile    = std::size(buffer);
    ofn.lpstrFile   = buffer;
    ofn.lpstrTitle  = title.data();
    ofn.Flags       = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (!initial_dir.empty()) {
        ofn.lpstrInitialDir = initial_dir.c_str();
        ofn.Flags          |= OFN_EXPLORER;
    }
    return ::GetSaveFileNameW(&ofn) ? std::wstring(buffer) : std::wstring{};
}

// The directory containing this executable (with trailing slash).
std::wstring module_dir() {
    wchar_t module[MAX_PATH] {};
    const DWORD got = ::GetModuleFileNameW(nullptr, module, MAX_PATH);
    if (got == 0 || got >= MAX_PATH) { return {}; }
    std::wstring path(module, got);
    const auto slash = path.find_last_of(L"\\/");
    return slash == std::wstring::npos ? std::wstring{} : path.substr(0, slash + 1);
}

// Locate a sibling executable next to this one.
std::wstring find_sibling(const wchar_t* exe_name) {
    const std::wstring dir = module_dir();
    if (dir.empty()) { return {}; }
    const std::wstring path = dir + exe_name;
    return ::GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES ? path : std::wstring{};
}

std::wstring find_sibling_cli() { return find_sibling(L"ninfer-cli.exe"); }

std::optional<std::string> read_text_file(const std::wstring& path) {
    std::ifstream in(path, std::ios::binary);  // MSVC fstreams take wide paths natively
    if (!in) { return std::nullopt; }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

bool write_text_file(const std::wstring& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) { return false; }
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(out);
}

// Prefill the form from the command line (see file header for usage). Called
// after CreateWindowExW — the controls exist by then — so the automated e2e
// can launch a fully populated window without any interactive input.
void apply_command_line_prefill(HWND hwnd, int argc, LPWSTR* argv) {
    std::wstring model;
    std::wstring max_new;
    std::wstring max_ctx;
    for (int i = 1; i < argc; ++i) {
        const std::wstring_view arg = argv[i];
        if (arg == L"--max-new" && i + 1 < argc) { max_new = argv[++i]; continue; }
        if (arg == L"--max-ctx" && i + 1 < argc) { max_ctx = argv[++i]; continue; }
        if (!arg.empty() && arg[0] == L'-') { continue; }  // unknown flag: ignore
        if (model.empty()) { model.assign(arg); }
    }
    if (!model.empty()) { ::SetWindowTextW(GetDlgItem(hwnd, IDC_MODEL_EDIT), model.c_str()); }
    if (!max_new.empty()) { ::SetWindowTextW(GetDlgItem(hwnd, IDC_MAX_NEW_EDIT), max_new.c_str()); }
    if (!max_ctx.empty()) { ::SetWindowTextW(GetDlgItem(hwnd, IDC_MAX_CONTEXT_EDIT), max_ctx.c_str()); }
}

// ---------------------------------------------------------------------------
// Child process state (serve only). `running` is touched by the UI thread and
// the watcher thread — the atomic keeps them consistent without a mutex.
// ---------------------------------------------------------------------------

struct ChildProcess {
    HANDLE process = nullptr;
    std::atomic<bool> running {false};
    std::thread watcher;  // joinable while running
};

ChildProcess g_child {};

// Cached VRAM probe: the last model probed and its KV fit in tokens.
// Re-probing only happens when the selected model changes, so repeated Serve
// clicks against the same artifact stay instant.
std::wstring g_probe_model;
std::uint32_t g_probe_fit = 0;

void set_status(HWND hwnd, std::wstring_view text) {
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_STATUS_TEXT), std::wstring(text).c_str());
}

// Parse a wide decimal string to an unsigned value; 0 on empty/malformed input.
std::uint64_t parse_wide_u64(const std::wstring& text) {
    if (text.empty()) { return 0; }
    errno = 0;
    wchar_t* end = nullptr;
    const unsigned long long value = std::wcstoull(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str()) { return 0; }
    return static_cast<std::uint64_t>(value);
}

// Append the sampling flags read from the form controls. Empty fields are
// omitted so engine/model defaults apply. When `fit` is nonzero (a successful
// VRAM probe), the probed KV fit is pinned as an explicit --kv-capacity and the
// requested max-context is clamped down to it — the pool must hold the whole
// sequence window, so a context larger than the pool would be rejected. The
// capacity curve also caps the pool at max_context x max_concurrency, so the
// window must be at least the pool: serve_child widens the window control to
// the probed fit before calling here (the Auto button does the same).
void append_sampling_args(HWND hwnd, std::vector<std::wstring>& args, std::uint32_t fit = 0) {
    std::wstring max_ctx      = get_control_text(GetDlgItem(hwnd, IDC_MAX_CONTEXT_EDIT));
    const std::wstring temp    = get_control_text(GetDlgItem(hwnd, IDC_TEMPERATURE_EDIT));
    const std::wstring top_p   = get_control_text(GetDlgItem(hwnd, IDC_TOP_P_EDIT));
    const std::wstring top_k   = get_control_text(GetDlgItem(hwnd, IDC_TOP_K_EDIT));
    const std::wstring seed    = get_control_text(GetDlgItem(hwnd, IDC_SEED_EDIT));
    const std::wstring draft   = get_control_text(GetDlgItem(hwnd, IDC_DRAFT_TOKENS_EDIT));

    const int kv_idx   = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_KV_DTYPE_COMBO), CB_GETCURSEL, 0, 0));
    const int spec_idx = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_SPEC_COMBO), CB_GETCURSEL, 0, 0));
    const bool greedy        = ::SendMessageW(GetDlgItem(hwnd, IDC_GREEDY_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED;
    const bool thinking      = ::SendMessageW(GetDlgItem(hwnd, IDC_THINKING_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED;
    const bool vision        = ::SendMessageW(GetDlgItem(hwnd, IDC_VISION_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED;
    const bool lm_head_draft =
        ::SendMessageW(GetDlgItem(hwnd, IDC_LM_HEAD_DRAFT_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED;

    if (fit > 0) {
        // Pin the KV pool to the probed fit; clamp the requested context to it so the
        // engine's kv_capacity >= max_context invariant holds and launch proceeds.
        args.push_back(L"--kv-capacity");
        args.push_back(std::to_wstring(fit));
        if (!max_ctx.empty() && parse_wide_u64(max_ctx) > fit) {
            max_ctx = std::to_wstring(fit);
        }
    }
    if (!max_ctx.empty()) { args.push_back(L"--max-context"); args.push_back(max_ctx); }
    if (kv_idx > 0) {
        static const wchar_t* kKvDtypes[] = {L"bf16", L"int8", L"fp8"};
        args.push_back(L"--kv-dtype");
        args.push_back(kKvDtypes[kv_idx - 1]);
    }
    if (!temp.empty()) { args.push_back(L"--temperature"); args.push_back(temp); }
    if (!top_p.empty()) { args.push_back(L"--top-p"); args.push_back(top_p); }
    if (!top_k.empty()) { args.push_back(L"--top-k"); args.push_back(top_k); }
    if (greedy) { args.push_back(L"--greedy"); }
    if (!thinking) { args.push_back(L"--no-thinking"); }
    if (vision) { args.push_back(L"--vision"); }
    if (spec_idx > 0) {
        static const wchar_t* kSpecs[] = {L"mtp", L"dflash"};
        args.push_back(L"--spec");
        args.push_back(kSpecs[spec_idx - 1]);
        if (!draft.empty()) { args.push_back(L"--draft-tokens"); args.push_back(draft); }
        if (lm_head_draft) { args.push_back(L"--lm-head-draft"); }
    }
    if (!seed.empty()) { args.push_back(L"--seed"); args.push_back(seed); }
}

std::wstring build_command_line(const std::vector<std::wstring>& args) {
    std::wstring command_line;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (i != 0) { command_line.push_back(L' '); }
        command_line += quote_arg(args[i]);
    }
    return command_line;
}

// ---------------------------------------------------------------------------
// VRAM probe. Runs the sibling CLI in --probe mode, which reports the largest
// KV capacity that fits current free VRAM without loading weights. The result
// pins the engine's KV pool (--kv-capacity) and drives the Auto button.
// ---------------------------------------------------------------------------

// Read a pipe to EOF into a UTF-8 string. Probe output is a handful of lines,
// so one-shot sequential reads are safe (well under the 4 KiB pipe buffer).
std::string read_pipe_all(HANDLE read_end) {
    std::string out;
    char chunk[4096];
    for (;;) {
        DWORD got = 0;
        if (!::ReadFile(read_end, chunk, sizeof(chunk), &got, nullptr) || got == 0) { break; }
        out.append(chunk, got);
    }
    return out;
}

std::wstring format_gib(std::uint64_t bytes) {
    wchar_t buffer[32];
    ::swprintf(buffer, std::size(buffer), L"%.2f GiB",
               static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return buffer;
}

// Extract an unsigned value following `key` in a "key=value" line of probe output.
// Returns 0 when the key is absent or the value is malformed (failures surface via status).
std::uint64_t parse_probe_field(const std::string& text, const char* key) {
    const auto pos = text.find(key);
    if (pos == std::string::npos) { return 0; }
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text.c_str() + pos + std::char_traits<char>::length(key), &end, 10);
    if (errno == ERANGE || end == text.c_str() + pos + std::char_traits<char>::length(key)) { return 0; }
    return static_cast<std::uint64_t>(value);
}

// Synchronously launch the sibling CLI in probe mode and return the largest KV
// capacity (tokens) that fits current free VRAM after a 3% margin. Completes in
// seconds (no weights, no model build). Returns 0 on failure — the status line
// carries the reason.
std::uint32_t run_probe(HWND hwnd, const std::wstring& model) {
    const std::wstring cli = find_sibling_cli();
    if (cli.empty()) {
        set_status(hwnd, L"Error: ninfer-cli.exe not found next to Ninfer.exe");
        return 0;
    }

    const std::vector<std::wstring> args {cli, model, L"--probe"};
    const std::wstring command_line = build_command_line(args);

    SECURITY_ATTRIBUTES sa {};
    sa.nLength        = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE out_read = nullptr, out_write = nullptr, err_read = nullptr, err_write = nullptr;
    if (!::CreatePipe(&out_read, &out_write, &sa, 0) || !::CreatePipe(&err_read, &err_write, &sa, 0)) {
        set_status(hwnd, L"Error: could not create probe pipes");
        return 0;
    }

    STARTUPINFOW si {};
    si.cb         = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdOutput = out_write;
    si.hStdError  = err_write;

    std::wstring mutable_command = command_line;
    PROCESS_INFORMATION pi {};
    if (!::CreateProcessW(cli.c_str(), mutable_command.data(), nullptr, nullptr, TRUE,
                          CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
        ::CloseHandle(out_read); ::CloseHandle(out_write);
        ::CloseHandle(err_read); ::CloseHandle(err_write);
        set_status(hwnd, L"Error: failed to start the VRAM probe");
        return 0;
    }
    ::CloseHandle(pi.hThread);
    // Drop the write ends so the readers observe EOF.
    ::CloseHandle(out_write);
    ::CloseHandle(err_write);

    const std::string stdout_text = read_pipe_all(out_read);
    const std::string stderr_text = read_pipe_all(err_read);
    ::WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 0;
    ::GetExitCodeProcess(pi.hProcess, &exit_code);
    ::CloseHandle(out_read);
    ::CloseHandle(err_read);
    ::CloseHandle(pi.hProcess);

    if (exit_code != 0) {
        set_status(hwnd, L"Probe failed (exit " + std::to_wstring(exit_code) + L"): " +
                           utf8_to_wide(stderr_text));
        return 0;
    }

    const std::uint64_t fit      = parse_probe_field(stdout_text, "kv_fit_tokens=");
    const std::uint64_t free_after = parse_probe_field(stdout_text, "vram_free_after_weights_bytes=");
    if (fit == 0) {
        set_status(hwnd, L"Probe returned no KV fit: " + utf8_to_wide(stderr_text));
        return 0;
    }

    g_probe_model = model;
    g_probe_fit   = static_cast<std::uint32_t>(fit);
    set_status(hwnd, L"Auto: ~" + std::to_wstring(fit) + L" tokens fit (free after weights: " +
                       format_gib(free_after) + L")");
    return g_probe_fit;
}

// Return the cached probe fit for `model`, re-probing only when the model changed.
std::uint32_t ensure_probe(HWND hwnd, const std::wstring& model) {
    if (model == g_probe_model && g_probe_fit > 0) { return g_probe_fit; }
    set_status(hwnd, L"Measuring VRAM…");
    ::UpdateWindow(hwnd);
    return run_probe(hwnd, model);
}

// On exit, terminate every ninfer engine process (ninfer-cli.exe and
// ninfer-serve.exe) in any session so no orphaned GPU work lingers after this
// GUI closes. Sibling Ninfer GUI windows are left untouched — only the engines
// die. Main-window only: the editor windows must never reach this.
void kill_all_ninfer_engines() {
    const wchar_t* names[] = {L"ninfer-cli.exe", L"ninfer-serve.exe"};
    HANDLE snapshot = ::CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) { return; }

    PROCESSENTRY32W entry {};
    entry.dwSize = sizeof(entry);
    if (::Process32FirstW(snapshot, &entry)) {
        do {
            for (const wchar_t* name : names) {
                if (_wcsicmp(entry.szExeFile, name) != 0) { continue; }
                HANDLE process = ::OpenProcess(PROCESS_TERMINATE, FALSE, entry.th32ProcessID);
                if (process != nullptr) {
                    ::TerminateProcess(process, 1);
                    ::CloseHandle(process);
                }
            }
        } while (::Process32NextW(snapshot, &entry));
    }
    ::CloseHandle(snapshot);
}

// Launch ninfer-serve.exe as a long-lived server in its own console window so
// the user watches "listening on http://host:port" and the live tok/s stats
// directly — unsloth-style. The message loop keeps pumping while it runs.
void launch_serve(HWND hwnd, const std::wstring& serve, const std::wstring& command_line,
                  std::wstring_view extra_status = {}) {
    // Reap a previous run's watcher thread: it stays joinable after posting
    // WM_APP_DONE, and assigning a new joinable thread to g_child.watcher would
    // std::terminate.
    if (g_child.watcher.joinable()) { g_child.watcher.join(); }

    STARTUPINFOW si {};
    si.cb = sizeof(si);

    // CreateProcessW takes an LPWSTR and may modify the buffer; work on a copy.
    std::wstring mutable_command = command_line;
    PROCESS_INFORMATION pi {};
    if (!::CreateProcessW(serve.c_str(), mutable_command.data(), nullptr, nullptr, TRUE,
                          CREATE_NEW_CONSOLE, nullptr, nullptr, &si, &pi)) {
        const DWORD err = ::GetLastError();
        set_status(hwnd, L"Error: failed to start ninfer-serve.exe (Win32 error " +
                            std::to_wstring(err) + L")");
        return;
    }
    ::CloseHandle(pi.hThread);

    g_child.process   = pi.hProcess;
    g_child.running.store(true);

    ::EnableWindow(GetDlgItem(hwnd, IDC_SERVE_BUTTON), FALSE);
    std::wstring status = L"Serving — watch the terminal window for status and stats…";
    if (!extra_status.empty()) { status += L' '; status += extra_status; }
    set_status(hwnd, status);

    g_child.watcher = std::thread([hwnd] {
        ::WaitForSingleObject(g_child.process, INFINITE);
        DWORD exit_code = 0;
        ::GetExitCodeProcess(g_child.process, &exit_code);
        ::CloseHandle(g_child.process);
        g_child.process = nullptr;
        g_child.running.store(false);
        ::PostMessageW(hwnd, WM_APP_DONE, static_cast<WPARAM>(exit_code), 0);
    });
}

// Build and launch the serve command line. "Max new" becomes the per-request
// default (clients may still override it). The fixed system-prompt path next
// to the executable is always passed (D3); a missing/empty file is a valid
// no-default configuration (D1), so no existence check is needed here.
void serve_child(HWND hwnd) {
    const std::wstring serve = find_sibling(L"ninfer-serve.exe");
    if (serve.empty()) {
        set_status(hwnd, L"Error: ninfer-serve.exe not found next to Ninfer.exe");
        return;
    }

    const std::wstring model   = get_control_text(GetDlgItem(hwnd, IDC_MODEL_EDIT));
    const std::wstring host    = get_control_text(GetDlgItem(hwnd, IDC_HOST_EDIT));
    const std::wstring port    = get_control_text(GetDlgItem(hwnd, IDC_PORT_EDIT));
    const std::wstring max_new = get_control_text(GetDlgItem(hwnd, IDC_MAX_NEW_EDIT));

    if (model.empty()) { set_status(hwnd, L"Error: choose a model artifact (.ninfer)"); return; }
    if (port.empty()) { set_status(hwnd, L"Error: enter a port number"); return; }
    for (wchar_t ch : port) {
        if (ch < L'0' || ch > L'9') {
            set_status(hwnd, L"Error: port must be numeric");
            return;
        }
    }

    // A serve already answering on the target port would make the new child die on
    // bind (or the probe would fail on VRAM pressure) with no useful status; detect
    // it up front via /health (the same helper the CT editor uses to refresh).
    if (!http_get(host, port, "/health").empty()) {
        set_status(hwnd, L"A serve is already running on " +
                            (host.empty() ? std::wstring(L"127.0.0.1") : host) + L":" + port +
                            L" — Stop or Exit it first, or pick another port.");
        return;
    }

    // Probe VRAM (cached per model) to pin the KV pool and warn when the requested
    // context exceeds what fits. A failed probe aborts the launch.
    const std::uint32_t fit = ensure_probe(hwnd, model);
    if (fit == 0) { return; }
    std::wstring extra_status;
    const std::uint64_t requested_ctx = parse_wide_u64(get_control_text(GetDlgItem(hwnd, IDC_MAX_CONTEXT_EDIT)));
    if (requested_ctx > fit) {
        extra_status = L"Warning: context " + std::to_wstring(requested_ctx) +
                       L" exceeds VRAM fit ~" + std::to_wstring(fit) + L"; clamped to fit.";
    }
    if (requested_ctx < fit) {
        // The other direction of the pool/window invariant: the engine's capacity
        // curve caps the KV pool at max_context x max_concurrency (concurrency is 1
        // here), so a pool wider than the context window is rejected at launch.
        // Widen the window to the probed fit - the same thing the Auto button does.
        ::SetWindowTextW(GetDlgItem(hwnd, IDC_MAX_CONTEXT_EDIT), std::to_wstring(fit).c_str());
    }

    std::vector<std::wstring> args {serve, model};
    args.push_back(L"--host");
    args.push_back(host.empty() ? std::wstring(L"127.0.0.1") : host);
    args.push_back(L"--port");
    args.push_back(port);
    if (!max_new.empty()) { args.push_back(L"--default-max-tokens"); args.push_back(max_new); }
    // Qwen's recommended server config for agentic use keeps closed-turn
    // reasoning content in responses; pass it unconditionally in serve mode.
    args.push_back(L"--preserve-thinking");
    const std::wstring dir = module_dir();
    if (!dir.empty()) {
        args.push_back(L"--system-prompt-file");
        args.push_back(dir + L"system-prompt.md");
    }
    append_sampling_args(hwnd, args, fit);

    launch_serve(hwnd, serve, build_command_line(args), extra_status);
}

// ---------------------------------------------------------------------------
// Self-contained SHA-256 (FIPS 180-4) for the template digest comparison (D5).
// ---------------------------------------------------------------------------

struct Sha256Hash {
    static constexpr std::uint32_t kInitial[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };
    static constexpr std::uint32_t kRound[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    };

    std::uint32_t state_[8] {};
    std::uint64_t total_len = 0;
    unsigned char buffer_[64] {};
    std::size_t buffer_len = 0;

    void reset() {
        for (int i = 0; i < 8; ++i) { state_[i] = kInitial[i]; }
        total_len = 0;
        buffer_len = 0;
    }

    static std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void transform(const unsigned char* block) {
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24) |
                   (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
                   (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
                   static_cast<std::uint32_t>(block[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t temp1 = h + S1 + ch + kRound[i] + w[i];
            const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = S0 + maj;
            h = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    void update(std::string_view data) {
        total_len += data.size();
        std::size_t offset = 0;
        if (buffer_len > 0) {
            const std::size_t fill = 64 - buffer_len;
            if (data.size() < fill) {
                for (std::size_t i = 0; i < data.size(); ++i) {
                    buffer_[buffer_len + i] = static_cast<unsigned char>(data[i]);
                }
                buffer_len += data.size();
                return;
            }
            for (std::size_t i = 0; i < fill; ++i) {
                buffer_[buffer_len + i] = static_cast<unsigned char>(data[i]);
            }
            transform(buffer_);
            offset = fill;
            buffer_len = 0;  // tail below now lands at buffer_[0..], not the stale offset
        }
        while (data.size() - offset >= 64) {
            transform(reinterpret_cast<const unsigned char*>(data.data() + offset));
            offset += 64;
        }
        for (std::size_t i = offset; i < data.size(); ++i) {
            buffer_[buffer_len + (i - offset)] = static_cast<unsigned char>(data[i]);
        }
        buffer_len = data.size() - offset;
    }

    std::array<std::uint8_t, 32> digest() {
        const std::uint64_t bits = total_len * 8;
        const unsigned char pad = 0x80;
        update(std::string_view(reinterpret_cast<const char*>(&pad), 1));
        const unsigned char zero = 0;
        while (buffer_len != 56) {
            update(std::string_view(reinterpret_cast<const char*>(&zero), 1));
        }
        for (int i = 7; i >= 0; --i) {
            const unsigned char byte = static_cast<unsigned char>((bits >> (i * 8)) & 0xFF);
            update(std::string_view(reinterpret_cast<const char*>(&byte), 1));
        }
        std::array<std::uint8_t, 32> out {};
        for (int i = 0; i < 8; ++i) {
            out[i * 4]     = static_cast<std::uint8_t>(state_[i] >> 24);
            out[i * 4 + 1] = static_cast<std::uint8_t>(state_[i] >> 16);
            out[i * 4 + 2] = static_cast<std::uint8_t>(state_[i] >> 8);
            out[i * 4 + 3] = static_cast<std::uint8_t>(state_[i]);
        }
        return out;
    }
};

std::string sha256_hex_of(std::string_view data) {
    Sha256Hash hasher;
    hasher.reset();
    hasher.update(data);
    const std::array<std::uint8_t, 32> digest = hasher.digest();
    static const char* const kHex = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (const unsigned char byte : digest) {
        out.push_back(kHex[byte >> 4]);
        out.push_back(kHex[byte & 0x0F]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// WinHttp GET (D3): short timeout, local HTTP only. Any failure collapses to an
// empty body — the editor window treats that as "serve unreachable".
// ---------------------------------------------------------------------------

std::string http_get(const std::wstring& host, const std::wstring& port, const std::string& path) {
    const std::wstring host_wide = host.empty() ? std::wstring(L"127.0.0.1") : host;
    const std::string narrow_port = wide_to_utf8(port);
    char* end = nullptr;
    unsigned long port_num = std::strtoul(narrow_port.c_str(), &end, 10);
    if (end == narrow_port.c_str() || port_num == 0) { port_num = 80; }

    HINTERNET session = ::WinHttpOpen(L"NInfer/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                      WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (session == nullptr) { return {}; }
    std::string body;
    const DWORD timeout_ms = 2000;  // a closed port must never hang the UI
    HINTERNET connect = ::WinHttpConnect(session, host_wide.c_str(), port_num, 0);
    if (connect != nullptr) {
        ::WinHttpSetTimeouts(connect, timeout_ms, timeout_ms, timeout_ms, timeout_ms);
        HINTERNET request =
            ::WinHttpOpenRequest(connect, L"GET", utf8_to_wide(path).c_str(), nullptr, WINHTTP_NO_REFERER,
                                 WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
        if (request != nullptr) {
            if (::WinHttpSendRequest(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                     WINHTTP_NO_REQUEST_DATA, 0, 0, 0) &&
                ::WinHttpReceiveResponse(request, nullptr)) {
                DWORD status_code   = 0;
                DWORD status_length = sizeof(status_code);
                if (::WinHttpQueryHeaders(
                        request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        WINHTTP_HEADER_NAME_BY_INDEX, &status_code, &status_length,
                        WINHTTP_NO_HEADER_INDEX) &&
                    status_code == 200) {
                    char chunk[4096];
                    DWORD got = 0;
                    while (::WinHttpReadData(request, chunk, sizeof(chunk), &got) && got > 0) {
                        body.append(chunk, got);
                    }
                }
            }
            ::WinHttpCloseHandle(request);
        }
        ::WinHttpCloseHandle(connect);
    }
    ::WinHttpCloseHandle(session);
    return body;
}

// ---------------------------------------------------------------------------
// D4 — System prompt editor (fixed path <exe-dir>\system-prompt.md)
// ---------------------------------------------------------------------------

void create_sp_controls(HWND hwnd) {
    const HFONT mono = ::CreateFontW(
        -12, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, FIXED_PITCH | FF_MODERN, L"Consolas");
    const HFONT gui = static_cast<HFONT>(::GetStockObject(DEFAULT_GUI_FONT));

    HWND edit = ::CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"",
                                  WS_CHILD | WS_VISIBLE | ES_MULTILINE | WS_VSCROLL | ES_AUTOVSCROLL,
                                  8, 8, 684, 360, hwnd, reinterpret_cast<HMENU>(IDC_SP_EDIT),
                                  GetModuleHandleW(nullptr), nullptr);
    ::SendMessageW(edit, WM_SETFONT, reinterpret_cast<WPARAM>(mono), TRUE);

    HWND save = ::CreateWindowExW(0, L"BUTTON", L"Save", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                                  8, 376, 100, 28, hwnd,
                                  reinterpret_cast<HMENU>(IDC_SP_SAVE_BUTTON),
                                  GetModuleHandleW(nullptr), nullptr);
    ::SendMessageW(save, WM_SETFONT, reinterpret_cast<WPARAM>(gui), TRUE);

    HWND status = ::CreateWindowExW(0, L"STATIC", L"",
                                    WS_CHILD | WS_VISIBLE | SS_LEFT | SS_NOPREFIX,
                                    116, 380, 576, 40, hwnd,
                                    reinterpret_cast<HMENU>(IDC_SP_STATUS),
                                    GetModuleHandleW(nullptr), nullptr);
    ::SendMessageW(status, WM_SETFONT, reinterpret_cast<WPARAM>(gui), TRUE);

    const std::wstring path = module_dir() + L"system-prompt.md";
    const std::optional<std::string> loaded = read_text_file(path);
    if (loaded && !loaded->empty()) {
        // The control renders CRLF line breaks (LF-only collapses to one line);
        // the limit follows the DISPLAY byte count (always >= the wchar count),
        // so no text is clipped and the user can still edit. The e2e oracle
        // reads it back with EM_GETLIMITTEXT (0x00D5), not WM_GETTEXTLENGTH
        // (0x000E = current text length, 2 smaller here because of the
        // 3-byte em dash).
        const std::string display = to_display_lf_crlf(*loaded);
        ::SetWindowTextW(edit, utf8_to_wide(display).c_str());
        ::SendMessageW(edit, EM_SETLIMITTEXT, static_cast<WPARAM>(display.size()), 0);
        ::SetWindowTextW(status,
                         (path + L" — " + std::to_wstring(loaded->size()) +
                          L" bytes. Active from the next serve restart (read at startup).").c_str());
    } else {
        ::SetWindowTextW(status, L"File does not exist yet — it is created on Save.");
    }
}

void open_system_prompt_editor(HWND main_hwnd) {
    const std::wstring path  = module_dir() + L"system-prompt.md";
    const std::wstring title = L"System prompt — " + path;
    // WS_POPUP + non-NULL parent = a TOP-LEVEL window owned by the main window (own caption
    // and taskbar entry, closes with the owner). EditorInfo goes in lpParam (12th argument)
    // — WM_CREATE reads CREATESTRUCTW.lpCreateParams. Passing it as hMenu instead makes
    // CreateWindowExW fail with ERROR_INVALID_MENU_HANDLE (1401): with a parent, hMenu is
    // validated and a heap pointer is not a menu.
    HWND hwnd = ::CreateWindowExW(
        0, kWindowClass, title.c_str(), WS_POPUP | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT, 700, 450, main_hwnd, nullptr,
        GetModuleHandleW(nullptr), reinterpret_cast<LPVOID>(new EditorInfo {1, {}, {}}));
    if (hwnd != nullptr) { ::ShowWindow(hwnd, SW_SHOW); }
    else {
        ::SetWindowTextW(GetDlgItem(main_hwnd, IDC_STATUS_TEXT),
                         (L"Error: could not open the system prompt editor (err=" +
                          std::to_wstring(::GetLastError()) + L")").c_str());
    }
}

// ---------------------------------------------------------------------------
// D5 — Chat template editor (digest comparison against the live serve /health)
// ---------------------------------------------------------------------------

// Canonical (LF-only) bytes of the CT editor's current text. The control holds
// the display form (CRLF line breaks); the loaded template and its /health
// digest are canonical LF-only, so every comparison hashes the normalized form.
std::string ct_editor_canonical(HWND hwnd) {
    return to_canonical_crlf_lf(wide_to_utf8(get_control_text(GetDlgItem(hwnd, IDC_CT_EDIT))));
}

void update_ct_editor_sha(HWND hwnd) {
    const std::string editor_sha = sha256_hex_of(ct_editor_canonical(hwnd));
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_CT_STATUS_EDITOR),
                     (L"Editor:  sha256 " + utf8_to_wide(editor_sha)).c_str());
}

void refresh_ct_status(HWND hwnd, const EditorInfo* info) {
    // Self-check the SHA-256 against the well-known empty-string oracle; a broken
    // digest would make every comparison silently lie.
    if (sha256_hex_of("") !=
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") {
        ::SetWindowTextW(GetDlgItem(hwnd, IDC_CT_STATUS_LOADED), L"Error: SHA-256 self-check failed");
        return;
    }
    update_ct_editor_sha(hwnd);
    const std::string body = http_get(info->host, info->port, "/health");
    nlohmann::json health = nlohmann::json::object();
    if (!body.empty()) {
        health = nlohmann::json::parse(body, nullptr, false);
    }
    if (health.is_discarded() || !health.contains("chat_template")) {
        ::SetWindowTextW(
            GetDlgItem(hwnd, IDC_CT_STATUS_LOADED),
            (L"Serve not reachable at " + (info->host.empty() ? std::wstring(L"127.0.0.1") : info->host) +
             L":" + info->port + L" — editor only.").c_str());
        ::SetWindowTextW(GetDlgItem(hwnd, IDC_CT_STATUS_MATCH), L"");
        return;
    }
    const nlohmann::json& ct = health.at("chat_template");
    const std::string loaded_sha = ct.value("sha256", std::string("-"));
    const std::string editor_sha = sha256_hex_of(ct_editor_canonical(hwnd));
    std::wstring line = L"Loaded:  sha256 " + utf8_to_wide(loaded_sha) +
                        L"  ·  semantics " + utf8_to_wide(ct.value("semantics", std::string("-")));
    if (ct.contains("reasoning_effort")) {
        const nlohmann::json& effort = ct.at("reasoning_effort");
        if (effort.contains("default")) {
            line += L"  ·  default effort " + utf8_to_wide(effort.at("default").get<std::string>());
        }
        if (effort.contains("supported") && effort.at("supported").is_array()) {
            std::string supported;
            for (const auto& level : effort.at("supported")) {
                if (!supported.empty()) { supported += ", "; }
                supported += level.get<std::string>();
            }
            line += L"  ·  supported: " + utf8_to_wide(supported);
        }
    }
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_CT_STATUS_LOADED), line.c_str());
    ::SetWindowTextW(
        GetDlgItem(hwnd, IDC_CT_STATUS_MATCH),
        (loaded_sha == editor_sha
             ? std::wstring(L"Matches the template loaded by the server.")
             : std::wstring(L"Differs from the loaded template — active only after artifact swap + server rebuild.")).c_str());
}

void create_ct_controls(HWND hwnd, const EditorInfo* info) {
    const HFONT mono = ::CreateFontW(
        -12, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, FIXED_PITCH | FF_MODERN, L"Consolas");
    const HFONT gui = static_cast<HFONT>(::GetStockObject(DEFAULT_GUI_FONT));

    HWND edit = ::CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"",
                                  WS_CHILD | WS_VISIBLE | ES_MULTILINE | WS_VSCROLL | ES_AUTOVSCROLL,
                                  8, 8, 744, 300, hwnd, reinterpret_cast<HMENU>(IDC_CT_EDIT),
                                  GetModuleHandleW(nullptr), nullptr);
    ::SendMessageW(edit, WM_SETFONT, reinterpret_cast<WPARAM>(mono), TRUE);

    auto button = [&](int id, std::wstring_view text, int x) {
        HWND b = ::CreateWindowExW(0, L"BUTTON", text.data(), WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                                   x, 316, 90, 28, hwnd, reinterpret_cast<HMENU>(id),
                                   GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(b, WM_SETFONT, reinterpret_cast<WPARAM>(gui), TRUE);
    };
    button(IDC_CT_OPEN_BUTTON, L"Open…", 8);
    button(IDC_CT_REFRESH_BUTTON, L"Refresh", 106);
    button(IDC_CT_SAVE_BUTTON, L"Save…", 204);

    auto status_line = [&](int id, int y) {
        HWND s = ::CreateWindowExW(0, L"STATIC", L"", WS_CHILD | WS_VISIBLE | SS_LEFT | SS_NOPREFIX,
                                   8, y, 744, 20, hwnd, reinterpret_cast<HMENU>(id),
                                   GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(s, WM_SETFONT, reinterpret_cast<WPARAM>(gui), TRUE);
    };
    status_line(IDC_CT_STATUS_LOADED, 352);
    status_line(IDC_CT_STATUS_EDITOR, 376);
    status_line(IDC_CT_STATUS_MATCH, 400);
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_CT_STATUS_LOADED),
                     L"Load a template file or paste the template source.");
    refresh_ct_status(hwnd, info);  // replaces the hint with live /health data when reachable
}

void open_template_editor(HWND main_hwnd) {
    const std::wstring host = get_control_text(GetDlgItem(main_hwnd, IDC_HOST_EDIT));
    const std::wstring port = get_control_text(GetDlgItem(main_hwnd, IDC_PORT_EDIT));
    const std::wstring display_host = host.empty() ? std::wstring(L"127.0.0.1") : host;
    const std::wstring title = L"Chat template — " + display_host + L":" + port;
    HWND hwnd = ::CreateWindowExW(
        0, kWindowClass, title.c_str(), WS_POPUP | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT, 760, 560, main_hwnd, nullptr,
        GetModuleHandleW(nullptr), reinterpret_cast<LPVOID>(new EditorInfo {2, host, port}));
    if (hwnd != nullptr) { ::ShowWindow(hwnd, SW_SHOW); }
    else {
        ::SetWindowTextW(GetDlgItem(main_hwnd, IDC_STATUS_TEXT),
                         (L"Error: could not open the chat template editor (err=" +
                          std::to_wstring(::GetLastError()) + L")").c_str());
    }
}

// ---------------------------------------------------------------------------
// Main window controls (D3: serve launcher layout, no one-shot panes)
// ---------------------------------------------------------------------------

void create_main_controls(HWND hwnd) {
    const HFONT font = static_cast<HFONT>(::GetStockObject(DEFAULT_GUI_FONT));
    auto label = [&](int id, std::wstring_view text, int x, int y) {
        HWND h = ::CreateWindowExW(0, L"STATIC", text.data(), WS_CHILD | WS_VISIBLE | SS_LEFT,
                                   x, y, 120, 18, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                                   GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(h, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
    };
    auto edit = [&](int id, int x, int y, int w, int h, DWORD extra = 0) {
        HWND e = ::CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"",
                                   WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL | extra,
                                   x, y, w, h, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                                   GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(e, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
        return e;
    };
    auto button = [&](int id, std::wstring_view text, int x, int y, int w) {
        HWND b = ::CreateWindowExW(0, L"BUTTON", text.data(), WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                                   x, y, w, 28, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                                   GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(b, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
        return b;
    };
    auto check = [&](int id, std::wstring_view text, int x, int y, bool checked) {
        HWND c = ::CreateWindowExW(0, L"BUTTON", text.data(), WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                                   x, y, 130, 20, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                                   GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(c, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
        if (checked) { ::SendMessageW(c, BM_SETCHECK, BST_CHECKED, 0); }
        return c;
    };
    auto combo = [&](int id, int x, int y, int w, std::vector<std::wstring_view> items) {
        HWND c = ::CreateWindowExW(0, L"COMBOBOX", L"",
                                   WS_CHILD | WS_VISIBLE | CBS_DROPDOWNLIST | WS_TABSTOP,
                                   x, y, w, 200, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                                   GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(c, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
        for (std::wstring_view item : items) {
            ::SendMessageW(c, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(item.data()));
        }
        ::SendMessageW(c, CB_SETCURSEL, 0, 0);
        return c;
    };

    // NOTE: labels use dialog ID 0. GetDlgItem resolves the FIRST child with a
    // given ID (Z-order), so a label sharing an edit's ID would make serve_child
    // read the label text instead of the user input.
    // Row 1: model artifact
    label(0, L"Model (.ninfer):", 8, 8);
    edit(IDC_MODEL_EDIT, 132, 6, 470, 22);
    button(IDC_MODEL_BROWSE, L"Browse…", 608, 5, 70);

    // Row 2: numeric parameters
    label(0, L"Max context:", 8, 36);
    edit(IDC_MAX_CONTEXT_EDIT, 100, 34, 70, 22);
    label(0, L"Max new:", 184, 36);
    edit(IDC_MAX_NEW_EDIT, 250, 34, 60, 22);
    label(0, L"KV dtype:", 324, 36);
    combo(IDC_KV_DTYPE_COMBO, 396, 34, 70, {L"(default)", L"bf16", L"int8", L"fp8"});
    button(IDC_AUTO_BUTTON, L"Auto", 480, 34, 50);

    // Row 3: sampling
    label(0, L"Temperature:", 8, 64);
    edit(IDC_TEMPERATURE_EDIT, 100, 62, 60, 22);
    label(0, L"Top-p:", 174, 64);
    edit(IDC_TOP_P_EDIT, 222, 62, 60, 22);
    label(0, L"Top-k:", 296, 64);
    edit(IDC_TOP_K_EDIT, 344, 62, 60, 22);
    check(IDC_GREEDY_CHECK, L"Greedy", 420, 64, false);

    // Row 4: behavior flags
    check(IDC_THINKING_CHECK, L"Thinking", 8, 92, true);
    check(IDC_VISION_CHECK, L"Vision", 150, 92, false);
    label(0, L"Spec:", 290, 94);
    combo(IDC_SPEC_COMBO, 330, 92, 90, {L"(none)", L"mtp", L"dflash"});
    label(0, L"Draft tokens:", 430, 94);
    edit(IDC_DRAFT_TOKENS_EDIT, 528, 92, 50, 22);
    check(IDC_LM_HEAD_DRAFT_CHECK, L"LM head draft", 600, 92, true);

    // Row 5: seed + defaults editors + serve endpoint + actions
    label(0, L"Seed:", 8, 122);
    edit(IDC_SEED_EDIT, 50, 120, 80, 22);
    button(IDC_SYSTEM_PROMPT_BUTTON, L"System prompt…", 160, 119, 110);
    button(IDC_TEMPLATE_BUTTON, L"Chat template…", 278, 119, 110);
    label(0, L"Host:", 430, 122);
    edit(IDC_HOST_EDIT, 470, 120, 100, 22);
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_HOST_EDIT), L"127.0.0.1");
    label(0, L"Port:", 580, 122);
    edit(IDC_PORT_EDIT, 620, 120, 60, 22);
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_PORT_EDIT), L"8080");
    button(IDC_SERVE_BUTTON, L"Serve", 690, 119, 70);
    button(IDC_STOP_BUTTON, L"Stop", 766, 119, 70);  // stops the child serve; GUI stays open
    button(IDC_EXIT_BUTTON, L"Exit", 842, 119, 60);

    // Qwen3.8-27B recommended defaults for coding (model card + engine presets):
    // 32k context, fp8 KV cache, thinking-mode sampling, MTP speculation with
    // an optimized proposal head. Users can override any of these per run.
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_MAX_CONTEXT_EDIT), L"32768");
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_MAX_NEW_EDIT), L"8192");
    ::SendMessageW(GetDlgItem(hwnd, IDC_KV_DTYPE_COMBO), CB_SETCURSEL, 3, 0);  // fp8
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_TEMPERATURE_EDIT), L"1.0");
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_TOP_P_EDIT), L"0.95");
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_TOP_K_EDIT), L"20");
    ::SendMessageW(GetDlgItem(hwnd, IDC_SPEC_COMBO), CB_SETCURSEL, 1, 0);      // mtp
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_DRAFT_TOKENS_EDIT), L"3");

    // Status bar (plain static)
    HWND status = ::CreateWindowExW(0, L"STATIC", L"Idle",
                                    WS_CHILD | WS_VISIBLE | SS_LEFT | SS_NOPREFIX,
                                    8, 150, 884, 18, hwnd, reinterpret_cast<HMENU>(IDC_STATUS_TEXT),
                                    GetModuleHandleW(nullptr), nullptr);
    ::SendMessageW(status, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
}

// ---------------------------------------------------------------------------
// Window procedure (main window + the two editor windows, same class)
// ---------------------------------------------------------------------------

LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_CREATE: {
            // WM_CREATE's lParam is a system-built CREATESTRUCTW* (always
            // non-null); the lpParam passed to CreateWindowExW lives at
            // CREATESTRUCTW.lpCreateParams.
            const CREATESTRUCTW* cs = reinterpret_cast<const CREATESTRUCTW*>(lParam);
            EditorInfo* info = reinterpret_cast<EditorInfo*>(cs->lpCreateParams);
            ::SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(info));
            if (info == nullptr) {
                create_main_controls(hwnd);
            } else if (info->kind == 1) {
                create_sp_controls(hwnd);
            } else {
                create_ct_controls(hwnd, info);
            }
            return 0;
        }

        case WM_COMMAND: {
            const int id = LOWORD(wParam);
            const auto* info = reinterpret_cast<const EditorInfo*>(
                ::GetWindowLongPtrW(hwnd, GWLP_USERDATA));
            if (id == IDC_MODEL_BROWSE) {
                // Open the dialog where the current model path lives (the
                // prefilled/selected artifact), else next to <exe>\..\models
                // (same fallback as the CT open button) — a stale default
                // folder showed an empty list under the *.ninfer filter.
                std::wstring initial_dir;
                const std::wstring model = get_control_text(GetDlgItem(hwnd, IDC_MODEL_EDIT));
                const auto slash = model.find_last_of(L"\\/");
                if (slash != std::wstring::npos) {
                    const std::wstring dir = model.substr(0, slash);
                    if (::GetFileAttributesW(dir.c_str()) & FILE_ATTRIBUTE_DIRECTORY) {
                        initial_dir = dir;
                    }
                }
                if (initial_dir.empty()) {
                    initial_dir = module_dir();
                    const std::wstring models_dir = initial_dir + L"../models";
                    if (::GetFileAttributesW(models_dir.c_str()) & FILE_ATTRIBUTE_DIRECTORY) {
                        initial_dir = models_dir;
                    }
                }
                std::wstring picked = browse_for_file(
                    hwnd, L"Select model artifact",
                    std::wstring(L"NInfer artifacts (*.ninfer)\0*.ninfer\0All files (*)\0*.*\0"),
                    initial_dir);
                if (!picked.empty()) { ::SetWindowTextW(GetDlgItem(hwnd, IDC_MODEL_EDIT), picked.c_str()); }
                return 0;
            }
            if (id == IDC_AUTO_BUTTON) {
                const std::wstring model = get_control_text(GetDlgItem(hwnd, IDC_MODEL_EDIT));
                if (model.empty()) {
                    set_status(hwnd, L"Select a model first");
                    return 0;
                }
                // Re-probe when the model changed (cached otherwise); fill Max context
                // with the probed fit so Serve pins the KV pool to what VRAM holds.
                const std::uint32_t fit = ensure_probe(hwnd, model);
                if (fit > 0) {
                    ::SetWindowTextW(GetDlgItem(hwnd, IDC_MAX_CONTEXT_EDIT),
                                     std::to_wstring(fit).c_str());
                }
                return 0;
            }
            if (id == IDC_SERVE_BUTTON && !g_child.running.load()) {
                serve_child(hwnd);
                return 0;
            }
            if (id == IDC_STOP_BUTTON) {
                // Stop the serve this GUI launched (its terminal window dies); the
                // GUI stays open. The watcher posts WM_APP_DONE, which re-enables
                // the button and reports the exit code. Exit / the window X keep
                // their old behavior (kill engines + close).
                if (g_child.running.load() && g_child.process != nullptr) {
                    set_status(hwnd, L"Stopping serve…");
                    ::EnableWindow(GetDlgItem(hwnd, IDC_STOP_BUTTON), FALSE);
                    ::TerminateProcess(g_child.process, 1);
                } else {
                    set_status(hwnd, L"No serve started from this GUI.");
                }
                return 0;
            }
            if (id == IDC_EXIT_BUTTON) {
                // Post WM_CLOSE (what the window X does) rather than calling DestroyWindow
                // directly: DestroyWindow tears the window down without invoking our WM_CLOSE
                // handler, so the kill_all_ninfer_engines() cleanup would be skipped.
                ::PostMessageW(hwnd, WM_CLOSE, 0, 0);
                return 0;
            }
            if (id == IDC_SYSTEM_PROMPT_BUTTON) {
                open_system_prompt_editor(hwnd);
                return 0;
            }
            if (id == IDC_TEMPLATE_BUTTON) {
                open_template_editor(hwnd);
                return 0;
            }
            if (id == IDC_SP_SAVE_BUTTON) {
                const std::wstring path = module_dir() + L"system-prompt.md";
                // The control holds the display form (CRLF line breaks); the file
                // stays canonical LF-only so the serve's bytes never drift. The
                // limit follows the display text so saving never clips the editor.
                const std::string display_content =
                    wide_to_utf8(get_control_text(GetDlgItem(hwnd, IDC_SP_EDIT)));
                const std::string content = to_canonical_crlf_lf(display_content);
                if (write_text_file(path, content)) {
                    ::SendMessageW(GetDlgItem(hwnd, IDC_SP_EDIT), EM_SETLIMITTEXT,
                                   static_cast<WPARAM>(display_content.size()), 0);
                    ::SetWindowTextW(
                        GetDlgItem(hwnd, IDC_SP_STATUS),
                        (path + L" — saved " + std::to_wstring(content.size()) +
                         L" bytes. Active from the next serve restart (read at startup).").c_str());
                } else {
                    ::SetWindowTextW(GetDlgItem(hwnd, IDC_SP_STATUS),
                                     (L"Error: could not write " + path).c_str());
                }
                return 0;
            }
            if (id == IDC_CT_OPEN_BUTTON) {
                std::wstring initial_dir = module_dir();
                const std::wstring models_dir = initial_dir + L"../models";
                if (::GetFileAttributesW(models_dir.c_str()) & FILE_ATTRIBUTE_DIRECTORY) {
                    initial_dir = models_dir;
                }
                const std::wstring picked = browse_for_file(
                    hwnd, L"Select chat template",
                    std::wstring(L"Jinja templates (*.jinja)\0*.jinja\0All files (*)\0*.*\0"),
                    initial_dir);
                if (!picked.empty()) {
                    const std::optional<std::string> content = read_text_file(picked);
                    if (content) {
                        // Display form: CRLF line breaks (LF-only renders as one line).
                        ::SetWindowTextW(GetDlgItem(hwnd, IDC_CT_EDIT),
                                         utf8_to_wide(to_display_lf_crlf(*content)).c_str());
                        update_ct_editor_sha(hwnd);
                    } else {
                        ::SetWindowTextW(GetDlgItem(hwnd, IDC_CT_STATUS_LOADED),
                                         (L"Error: could not read " + picked).c_str());
                    }
                }
                return 0;
            }
            if (id == IDC_CT_REFRESH_BUTTON && info != nullptr) {
                refresh_ct_status(hwnd, info);
                return 0;
            }
            if (id == IDC_CT_SAVE_BUTTON) {
                const std::wstring picked = browse_save_file(
                    hwnd, L"Save chat template",
                    std::wstring(L"Jinja templates (*.jinja)\0*.jinja\0All files (*)\0*.*\0"),
                    module_dir());
                if (picked.empty()) { return 0; }
                // Canonical LF-only bytes on disk (the control's display form is CRLF).
                const std::string content = ct_editor_canonical(hwnd);
                if (write_text_file(picked, content)) {
                    ::SetWindowTextW(GetDlgItem(hwnd, IDC_CT_STATUS_LOADED),
                                     (L"Saved to " + picked +
                                      L" — active after the next artifact swap + server rebuild.").c_str());
                } else {
                    ::SetWindowTextW(GetDlgItem(hwnd, IDC_CT_STATUS_LOADED),
                                     (L"Error: could not write " + picked).c_str());
                }
                return 0;
            }
            if (id == IDC_CT_EDIT && HIWORD(wParam) == EN_CHANGE) {
                update_ct_editor_sha(hwnd);
                return 0;
            }
            break;
        }

        case WM_APP_DONE: {
            ::EnableWindow(GetDlgItem(hwnd, IDC_SERVE_BUTTON), TRUE);
            ::EnableWindow(GetDlgItem(hwnd, IDC_STOP_BUTTON), TRUE);
            set_status(hwnd, wParam == 0 ? L"Server stopped (exit 0)"
                                         : L"Server exited with code " + std::to_wstring(wParam));
            return 0;
        }

        case WM_CLOSE: {
            const LONG_PTR user = ::GetWindowLongPtrW(hwnd, GWLP_USERDATA);
            if (user != 0) {
                // Editor window: closing it must never touch the engines.
                DestroyWindow(hwnd);
                return 0;
            }
            // Main window: kill every ninfer engine in any session before this GUI exits,
            // so closing one window never leaves orphaned GPU work behind. This also
            // terminates this GUI's own child; the cleanup below still joins its watcher.
            kill_all_ninfer_engines();
            if (g_child.running.load()) {
                if (g_child.process != nullptr) { ::TerminateProcess(g_child.process, 1); }
                if (g_child.watcher.joinable()) { g_child.watcher.join(); }
            }
            DestroyWindow(hwnd);
            return 0;
        }

        case WM_DESTROY: {
            const LONG_PTR user = ::GetWindowLongPtrW(hwnd, GWLP_USERDATA);
            delete reinterpret_cast<EditorInfo*>(user);
            if (user == 0) { ::PostQuitMessage(0); }  // main window only
            return 0;
        }
    }
    return ::DefWindowProcW(hwnd, msg, wParam, lParam);
}

}  // namespace

// Must have external linkage (outside the anonymous namespace) so the CRT
// entry point can find it.
int WINAPI WinMain(HINSTANCE h_instance, HINSTANCE, LPSTR, int show_cmd) {
    INITCOMMONCONTROLSEX icc {};
    icc.dwSize = sizeof(icc);
    icc.dwICC  = ICC_STANDARD_CLASSES;
    ::InitCommonControlsEx(&icc);

    WNDCLASSEXW wc {};
    wc.cbSize        = sizeof(wc);
    wc.style         = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc   = wnd_proc;
    wc.hInstance     = h_instance;
    wc.hCursor       = ::LoadCursorW(nullptr, MAKEINTRESOURCEW(IDC_ARROW));
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = kWindowClass;
    wc.hIcon         = ::LoadIconW(nullptr, MAKEINTRESOURCEW(IDI_APPLICATION));
    ::RegisterClassExW(&wc);

    // 920x240: the one-shot Answer/Log panes are gone, so the window is just the
    // parameter rows + status bar.
    HWND hwnd = ::CreateWindowExW(0, wc.lpszClassName, L"NInfer",
                                  WS_OVERLAPPEDWINDOW | WS_MINIMIZEBOX, CW_USEDEFAULT, CW_USEDEFAULT,
                                  920, 240, nullptr, nullptr, h_instance, nullptr);
    if (hwnd == nullptr) { return 1; }

    // Prefill the form from the command line, if any arguments were given.
    int argc = 0;
    LPWSTR* argv = ::CommandLineToArgvW(::GetCommandLineW(), &argc);
    if (argv != nullptr && argc > 1) {
        apply_command_line_prefill(hwnd, argc, argv);
        ::CoTaskMemFree(argv);
    }

    // The VRAM probe (Auto) needs the sibling CLI; surface its absence up front.
    if (find_sibling_cli().empty()) {
        ::MessageBoxW(hwnd, L"ninfer-cli.exe was not found next to Ninfer.exe.\n\n"
                            L"Place ninfer-cli.exe in the same folder and restart.",
                      L"NInfer", MB_ICONWARNING);
    }

    ::ShowWindow(hwnd, show_cmd);
    ::UpdateWindow(hwnd);

    MSG msg {};
    while (::GetMessageW(&msg, nullptr, 0, 0) > 0) {
        ::TranslateMessage(&msg);
        ::DispatchMessageW(&msg);
    }
    return static_cast<int>(msg.wParam);
}
