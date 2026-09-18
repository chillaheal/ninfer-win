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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iterator>
#include <istream>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <time.h>
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
    IDC_VISION_COMBO,             // 114 (was the Vision checkbox; now the Off/GPU/CPU combo)
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
    // Sampling preset fields (2026-09-01): the thinking checkbox drives these defaults.
    IDC_MIN_P_EDIT,               // 142
    IDC_PRESENCE_PENALTY_EDIT,    // 143
    // Usage tracking (D11):
    IDC_USAGE_TEXT,               // 144 — read-only usage/rate block below the status bar
    // Default thinking budget (2026-09-05): editable cap passed to the serve as
    // --default-thinking-budget; an empty field omits the flag (unlimited).
    IDC_THINKING_BUDGET_EDIT,         // 145
    // Frequency-penalty sampling override (2026-09-17): empty field omits the
    // --frequency-penalty flag so the serve default applies.
    IDC_FREQUENCY_PENALTY_EDIT,       // 146
};

constexpr UINT WM_APP_DONE = WM_APP + 1;  // wParam: serve exit code (main window only)

constexpr wchar_t kWindowClass[] = L"NinferGuiWindow";
// User-facing product identity (title bar + dialog captions). The on-disk exe
// name stays Ninfer.exe (deploy/e2e/kill-by-name logic reference it). v1.0.2 =
// fork changelog version for the "Ninfer AI" rename + usage-stats change.
constexpr wchar_t kAppIdentity[] = L"Ninfer AI v1.0.2";

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

// Cached VRAM probe: the last probe key and its KV fit in tokens. The key
// combines the model with every sizing control (vision regime, KV dtype,
// speculation) because the fit depends on where the ViT weights and encode
// workspace live (off/gpu/cpu), on the KV bytes per token, and on the MTP
// reservation. Re-probing only happens when the model or a sizing control
// changes, so repeated Serve clicks against the same artifact stay instant.
std::wstring g_probe_key;
std::uint32_t g_probe_fit = 0;

void set_status(HWND hwnd, std::wstring_view text) {
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_STATUS_TEXT), std::wstring(text).c_str());
}

// ---------------------------------------------------------------------------
// Settings persistence (gui-settings.ini next to this exe). The GUI saves the
// last USED form state when a serve is launched and restores it at startup.
// The command-line prefill runs after the restore (WinMain), so the CLI still
// beats the file. The system prompt / chat template are NOT stored here —
// they persist in their own files (system-prompt.md / user-picked template).
// ---------------------------------------------------------------------------

std::wstring settings_path() { return module_dir() + L"gui-settings.ini"; }

// Atomic replace: write <path>.tmp, then MoveFileExW over the target. A torn
// write would otherwise leave a truncated ini and silently reset the form
// (the GUI has a crash history — WER 2026-09-02 — so this matters).
bool write_text_file_atomic(const std::wstring& path, const std::string& bytes) {
    const std::wstring tmp = path + L".tmp";
    if (!write_text_file(tmp, bytes)) { return false; }
    if (!::MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        ::DeleteFileW(tmp.c_str());
        return false;
    }
    return true;
}

// Restore the saved form state. Runs in WM_CREATE right after the defaults are
// set, so each remembered key overwrites the default; a missing/unusable file
// leaves the defaults untouched (today's behavior). Unknown keys and out-of-
// range combo indices are skipped (forward-compatible).
void apply_saved_settings(HWND hwnd) {
    const std::optional<std::string> raw = read_text_file(settings_path());
    if (!raw || raw->empty()) { return; }
    bool restored_any = false;
    std::size_t pos = 0;
    while (pos < raw->size()) {
        const std::size_t eol = raw->find('\n', pos);
        const std::string line =
            raw->substr(pos, eol == std::string::npos ? std::string::npos : eol - pos);
        pos = (eol == std::string::npos) ? raw->size() : eol + 1;
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos || eq == 0) { continue; }
        // Views directly into `line` (alive for the whole loop body). NOT
        // `line.substr(...)` bound to a string_view -- that view would dangle
        // at the end of the declaration (std::string::substr returns a
        // temporary), and the key compare would read freed memory (UB that
        // broke the restore deterministically per binary, 2026-09-02).
        const std::string_view key = std::string_view(line).substr(0, eq);
        std::string_view value = std::string_view(line).substr(eq + 1);
        if (!value.empty() && value.back() == '\r') { value.remove_suffix(1); }
        auto set_edit = [&](int id, std::string_view v) {
            ::SetWindowTextW(GetDlgItem(hwnd, id), utf8_to_wide(v).c_str());
            restored_any = true;
        };
        auto set_combo = [&](int id, std::string_view v) {
            const int idx = std::atoi(std::string(v).c_str());
            const int count =
                static_cast<int>(::SendMessageW(GetDlgItem(hwnd, id), CB_GETCOUNT, 0, 0));
            if (idx >= 0 && idx < count) {
                ::SendMessageW(GetDlgItem(hwnd, id), CB_SETCURSEL, idx, 0);
                restored_any = true;
            }
        };
        auto set_check = [&](int id, std::string_view v) {
            const std::string on = std::string(v);
            ::SendMessageW(GetDlgItem(hwnd, id), BM_SETCHECK,
                           (on == "1" || on == "on" || on == "true") ? BST_CHECKED : BST_UNCHECKED, 0);
            restored_any = true;
        };
        // Scope (user decision 2026-09-02, extended 2026-09-05): persist the
        // model, last context, kv dtype (fp8), vision choice, the MTP draft spec
        // and the thinking budget. The remaining sampling fields follow the
        // thinking/vision toggles and are not persisted; unknown keys (incl. old
        // 18-key files) are ignored.
        if (key == "model") { set_edit(IDC_MODEL_EDIT, value); }
        else if (key == "max_context") { set_edit(IDC_MAX_CONTEXT_EDIT, value); }
        else if (key == "kv_dtype") { set_combo(IDC_KV_DTYPE_COMBO, value); }
        else if (key == "vision") { set_combo(IDC_VISION_COMBO, value); }
        else if (key == "spec") { set_combo(IDC_SPEC_COMBO, value); }
        else if (key == "draft_tokens") { set_edit(IDC_DRAFT_TOKENS_EDIT, value); }
        else if (key == "thinking_budget") { set_edit(IDC_THINKING_BUDGET_EDIT, value); }
        else if (key == "host") { set_edit(IDC_HOST_EDIT, value); }
        else if (key == "port") { set_edit(IDC_PORT_EDIT, value); }
        // unknown key: ignore
    }
    // A saved model path that no longer exists is still prefilled, but the
    // serve launch would fail obscurely — surface it in the status bar.
    if (!restored_any) { return; }
    const std::wstring model = get_control_text(GetDlgItem(hwnd, IDC_MODEL_EDIT));
    if (!model.empty() && ::GetFileAttributesW(model.c_str()) == INVALID_FILE_ATTRIBUTES) {
        set_status(hwnd, L"Model file not found: " + model);
    }
}
// Persist the current form state (best-effort: a write failure never blocks
// the serve launch). Called from the Serve-button handler before serve_child.
void save_current_settings(HWND hwnd) {
    const std::wstring model = get_control_text(GetDlgItem(hwnd, IDC_MODEL_EDIT));
    if (model.empty()) { return; }  // nothing launched without a model; nothing to persist
    const auto combo_idx = [&](int id) {
        const LRESULT i = ::SendMessageW(GetDlgItem(hwnd, id), CB_GETCURSEL, 0, 0);
        return i == CB_ERR ? 0 : static_cast<int>(i);
    };
    std::string out;
    auto line = [&](const char* k, std::string_view v) {
        out += k;
        out += '=';
        out += v;
        out += '\n';
    };
    // Scope (user decision 2026-09-02, extended 2026-09-05 with thinking_budget):
    // the same 7 keys apply_saved_settings restores — the other fields follow the
    // thinking/vision toggles.
    line("model", wide_to_utf8(model));
    line("max_context", wide_to_utf8(get_control_text(GetDlgItem(hwnd, IDC_MAX_CONTEXT_EDIT))));
    line("kv_dtype", std::to_string(combo_idx(IDC_KV_DTYPE_COMBO)));
    line("vision", std::to_string(combo_idx(IDC_VISION_COMBO)));
    line("spec", std::to_string(combo_idx(IDC_SPEC_COMBO)));
    line("draft_tokens",
         wide_to_utf8(get_control_text(GetDlgItem(hwnd, IDC_DRAFT_TOKENS_EDIT))));
    line("thinking_budget",
         wide_to_utf8(get_control_text(GetDlgItem(hwnd, IDC_THINKING_BUDGET_EDIT))));
    line("host", wide_to_utf8(get_control_text(GetDlgItem(hwnd, IDC_HOST_EDIT))));
    line("port", wide_to_utf8(get_control_text(GetDlgItem(hwnd, IDC_PORT_EDIT))));
    write_text_file_atomic(settings_path(), out);
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

// Strip spaces and tabs so a typed thousands separator ("130 000") parses as one
// number - wcstoull stops at the first space and would read 130.
std::wstring strip_wide_spaces(const std::wstring& text) {
    std::wstring out;
    out.reserve(text.size());
    for (const wchar_t ch : text) {
        if (ch != L' ' && ch != L'\t') { out.push_back(ch); }
    }
    return out;
}

// Qwen's recommended sampling presets (match the engine's per-target defaults in
// package.cpp). The thinking checkbox drives these: checked -> thinking preset,
// unchecked -> Instruct (non-thinking) preset. The fields stay user-editable; this
// only (re)populates them at startup and when the mode is toggled.
void apply_sampling_preset(HWND hwnd, bool thinking) {
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_TEMPERATURE_EDIT), thinking ? L"1.0" : L"0.7");
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_TOP_P_EDIT), thinking ? L"0.95" : L"0.80");
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_TOP_K_EDIT), L"20");
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_MIN_P_EDIT), L"0.0");
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_PRESENCE_PENALTY_EDIT), thinking ? L"0.0" : L"1.5");
    // Frequency penalty: neutral 0.0 by default (no-op); raising it (0.3-0.7 typical)
    // damps tokens that recur often in the context.
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_FREQUENCY_PENALTY_EDIT), L"0.0");
}

// Per-model sizing defaults (2026-09-12, updated 2026-09-13): when a browsed model is a
// dflash2 artifact, pin its serve-ready defaults - the dflash2 spec backend, an fp8 KV
// cache, the 262k window, and 7 draft tokens. Every other model keeps the Qwen3.8-27B
// defaults already seeded at startup (mtp spec, fp8 KV). Every field stays user-overridable;
// a typed context up to the probed fit launches as-is.
void apply_model_sizing_defaults(HWND hwnd, const std::wstring& model) {
    // Only the dflash2 family carries dedicated serving defaults; every other model
    // keeps the Qwen3.8-27B defaults seeded at startup.
    if (model.find(L"dflash2") != std::wstring::npos) {
        ::SendMessageW(GetDlgItem(hwnd, IDC_SPEC_COMBO), CB_SETCURSEL, 3, 0);    // dflash2
        ::SendMessageW(GetDlgItem(hwnd, IDC_KV_DTYPE_COMBO), CB_SETCURSEL, 3, 0);  // fp8
        ::SetWindowTextW(GetDlgItem(hwnd, IDC_MAX_CONTEXT_EDIT), L"262144");
        ::SetWindowTextW(GetDlgItem(hwnd, IDC_DRAFT_TOKENS_EDIT), L"7");
    } else {
        // Every other model keeps the Qwen3.8-27B defaults already seeded at
        // startup (mtp spec, fp8 KV).
        ::SendMessageW(GetDlgItem(hwnd, IDC_SPEC_COMBO), CB_SETCURSEL, 1, 0);    // mtp
        ::SendMessageW(GetDlgItem(hwnd, IDC_KV_DTYPE_COMBO), CB_SETCURSEL, 3, 0);  // fp8
    }
}

// Append the sampling flags read from the form controls. Empty fields are
// omitted so engine/model defaults apply. When `fit` is nonzero (a successful
// VRAM probe), the KV pool is pinned to the effective context window - the
// requested max-context (spaces stripped) clamped down to the fit. A context
// larger than the pool is rejected (the pool must hold the whole sequence
// window), and with concurrency 1 the capacity curve caps the pool at
// max_context, so pool == window; a pool no wider than the probed fit fits
// VRAM by construction. A user-set context below the fit therefore launches
// as-is with a smaller pool instead of being widened to the fit.
void append_sampling_args(HWND hwnd, std::vector<std::wstring>& args, std::uint32_t fit = 0) {
    std::wstring max_ctx      = get_control_text(GetDlgItem(hwnd, IDC_MAX_CONTEXT_EDIT));
    const std::wstring temp    = get_control_text(GetDlgItem(hwnd, IDC_TEMPERATURE_EDIT));
    const std::wstring top_p   = get_control_text(GetDlgItem(hwnd, IDC_TOP_P_EDIT));
    const std::wstring top_k   = get_control_text(GetDlgItem(hwnd, IDC_TOP_K_EDIT));
    const std::wstring min_p   = get_control_text(GetDlgItem(hwnd, IDC_MIN_P_EDIT));
    const std::wstring presence= get_control_text(GetDlgItem(hwnd, IDC_PRESENCE_PENALTY_EDIT));
    const std::wstring seed    = get_control_text(GetDlgItem(hwnd, IDC_SEED_EDIT));
    const std::wstring draft   = get_control_text(GetDlgItem(hwnd, IDC_DRAFT_TOKENS_EDIT));
    const std::wstring budget  = get_control_text(GetDlgItem(hwnd, IDC_THINKING_BUDGET_EDIT));

    const int kv_idx   = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_KV_DTYPE_COMBO), CB_GETCURSEL, 0, 0));
    const int spec_idx = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_SPEC_COMBO), CB_GETCURSEL, 0, 0));
    const bool greedy        = ::SendMessageW(GetDlgItem(hwnd, IDC_GREEDY_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED;
    const bool thinking      = ::SendMessageW(GetDlgItem(hwnd, IDC_THINKING_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED;
    // Vision regime: 0 = Off (no flag), 1 = GPU, 2 = CPU (the combo items).
    const int vision_mode    = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_VISION_COMBO), CB_GETCURSEL, 0, 0));
    const bool lm_head_draft =
        ::SendMessageW(GetDlgItem(hwnd, IDC_LM_HEAD_DRAFT_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED;

    if (fit > 0) {
        // Pin the KV pool to the effective window (requested context clamped to
        // the probed fit; empty or non-numeric field = the fit itself). This
        // keeps the engine's pool/window invariants (see the function comment)
        // while honoring a user-set context below the fit.
        std::uint64_t window =
            max_ctx.empty() ? fit : parse_wide_u64(strip_wide_spaces(max_ctx));
        if (window == 0 || window > fit) { window = fit; }
        args.push_back(L"--kv-capacity");
        args.push_back(std::to_wstring(window));
        if (!max_ctx.empty()) { max_ctx = std::to_wstring(window); }
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
    if (!min_p.empty()) { args.push_back(L"--min-p"); args.push_back(min_p); }
    if (!presence.empty()) { args.push_back(L"--presence-penalty"); args.push_back(presence); }
    if (greedy) { args.push_back(L"--greedy"); }
    if (!thinking) { args.push_back(L"--no-thinking"); }
    if (vision_mode > 0) {
        // Explicit regime (a bare --vision would mean GPU); Off omits the flag.
        static const wchar_t* kVisionRegimes[] = {L"gpu", L"cpu"};
        args.push_back(L"--vision");
        args.push_back(kVisionRegimes[vision_mode - 1]);
    }
    if (spec_idx > 0) {
        // dflash2+ngram (combo index 4) reuses the dflash2 backend and adds ngram
        // self-speculation; it reuses the draft-tokens (K) field.
        static const wchar_t* kSpecs[] = {L"mtp", L"dflash", L"dflash2", L"dflash2", L"dflash2"};
        args.push_back(L"--spec");
        args.push_back(kSpecs[spec_idx - 1]);
        if (spec_idx == 4) { args.push_back(L"--ngram"); }  // dflash2+ngram
        if (!draft.empty()) { args.push_back(L"--draft-tokens"); args.push_back(draft); }
        if (lm_head_draft) { args.push_back(L"--lm-head-draft"); }
    }
    if (!seed.empty()) { args.push_back(L"--seed"); args.push_back(seed); }
    // Validated up front in serve_child (digits, 1..u32max); empty omits the flag.
    if (!budget.empty()) { args.push_back(L"--default-thinking-budget"); args.push_back(budget); }
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
// carries the reason. The probe forwards the same sizing controls the serve
// child gets (vision regime, KV dtype, speculation) so the fitted pool matches
// the launch; `key` is the (model, sizing controls) combination the result is
// cached under.
std::uint32_t run_probe(HWND hwnd, const std::wstring& model, const std::wstring& key) {
    const std::wstring cli = find_sibling_cli();
    if (cli.empty()) {
        set_status(hwnd, L"Error: ninfer-cli.exe not found next to Ninfer.exe");
        return 0;
    }

    // Probe the same sizing regime the serve child will run, so the fitted pool
    // matches the launch (off/default items -> no flag; the probe loads no
    // weights, ~1 s). Verbatim parity with append_sampling_args for the sizing
    // flags; sampling-only flags do not size the pool and are omitted.
    const int vision_mode = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_VISION_COMBO), CB_GETCURSEL, 0, 0));
    const int kv_idx   = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_KV_DTYPE_COMBO), CB_GETCURSEL, 0, 0));
    const int spec_idx = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_SPEC_COMBO), CB_GETCURSEL, 0, 0));
    const std::wstring draft = get_control_text(GetDlgItem(hwnd, IDC_DRAFT_TOKENS_EDIT));
    const bool lm_head_draft =
        ::SendMessageW(GetDlgItem(hwnd, IDC_LM_HEAD_DRAFT_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED;
    std::vector<std::wstring> args {cli, model, L"--probe"};
    if (vision_mode > 0) {
        static const wchar_t* kVisionRegimes[] = {L"gpu", L"cpu"};
        args.push_back(L"--vision");
        args.push_back(kVisionRegimes[vision_mode - 1]);
    }
    if (kv_idx > 0) {
        static const wchar_t* kKvDtypes[] = {L"bf16", L"int8", L"fp8"};
        args.push_back(L"--kv-dtype");
        args.push_back(kKvDtypes[kv_idx - 1]);
    }
    if (spec_idx > 0) {
        // dflash2+ngram (combo index 4) reuses the dflash2 backend and adds ngram
        // self-speculation; it reuses the draft-tokens (K) field.
        static const wchar_t* kSpecs[] = {L"mtp", L"dflash", L"dflash2", L"dflash2", L"dflash2"};
        args.push_back(L"--spec");
        args.push_back(kSpecs[spec_idx - 1]);
        if (spec_idx == 4) { args.push_back(L"--ngram"); }  // dflash2+ngram
        if (!draft.empty()) { args.push_back(L"--draft-tokens"); args.push_back(draft); }
        if (lm_head_draft) { args.push_back(L"--lm-head-draft"); }
    }
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

    g_probe_key = key;
    g_probe_fit = static_cast<std::uint32_t>(fit);
    const std::wstring regime = vision_mode == 1    ? L" (vision gpu)"
                               : vision_mode == 2 ? L" (vision cpu)"
                               :                    L"";
    set_status(hwnd, L"AutoContext" + regime + L": ~" + std::to_wstring(fit) + L" tokens fit (free after weights: " +
                           format_gib(free_after) + L")");
    return g_probe_fit;
}

// The probe cache key: the model plus every sizing control the probe forwards
// (vision regime, KV dtype, speculation). Reads the same controls run_probe
// does, so a changed key means a different probe command line.
std::wstring build_probe_key(HWND hwnd, const std::wstring& model) {
    const int vision_mode = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_VISION_COMBO), CB_GETCURSEL, 0, 0));
    const int kv_idx   = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_KV_DTYPE_COMBO), CB_GETCURSEL, 0, 0));
    const int spec_idx = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_SPEC_COMBO), CB_GETCURSEL, 0, 0));
    const std::wstring draft = get_control_text(GetDlgItem(hwnd, IDC_DRAFT_TOKENS_EDIT));
    const bool lm_head_draft =
        ::SendMessageW(GetDlgItem(hwnd, IDC_LM_HEAD_DRAFT_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED;
    std::wstring key = model;
    key.push_back(L'\x01');
    key.append(std::to_wstring(vision_mode));
    key.push_back(L'\x01');
    key.append(std::to_wstring(kv_idx));
    key.push_back(L'\x01');
    key.append(std::to_wstring(spec_idx));
    key.push_back(L'\x01');
    key.append(draft);
    key.push_back(L'\x01');
    key.push_back(lm_head_draft ? L'1' : L'0');
    return key;
}

// Return the cached probe fit for the current (model, sizing controls),
// re-probing when any part of the key changed.
std::uint32_t ensure_probe(HWND hwnd, const std::wstring& model) {
    const std::wstring key = build_probe_key(hwnd, model);
    if (key == g_probe_key && g_probe_fit > 0) { return g_probe_fit; }
    set_status(hwnd, L"Measuring VRAM…");
    ::UpdateWindow(hwnd);
    return run_probe(hwnd, model, key);
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

    // The serve rejects a thinking budget of 0 or out of u32 range; surface it up
    // front like the port check. An empty field means "no default budget" and
    // simply omits --default-thinking-budget (thinking stays uncapped).
    const std::wstring budget = get_control_text(GetDlgItem(hwnd, IDC_THINKING_BUDGET_EDIT));
    if (!budget.empty()) {
        std::uint64_t v = 0;
        for (wchar_t ch : budget) {
            if (ch < L'0' || ch > L'9') { v = UINT64_MAX; break; }
            v = v * 10 + static_cast<std::uint64_t>(ch - L'0');
            if (v > 4294967295ULL) { v = UINT64_MAX; break; }  // u32 ceiling
        }
        if (v == 0 || v == UINT64_MAX) {
            set_status(hwnd, L"Error: thinking budget must be a number between 1 and 4294967295");
            return;
        }
    }

    // The engine rejects dflash/dflash2 + vision (cli/serve option validation);
    // surface it up front instead of dying at child launch.
    const int spec_sel   = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_SPEC_COMBO), CB_GETCURSEL, 0, 0));
    const int vision_sel = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_VISION_COMBO), CB_GETCURSEL, 0, 0));
    if ((spec_sel == 2 /* dflash */ || spec_sel == 3 /* dflash2 */ || spec_sel == 4 /* dflash2+ngram */) && vision_sel != 0 /* off */) {
        set_status(hwnd, L"dflash/dflash2 cannot be combined with vision: set Vision to Off or Spec to (none)/mtp");
        return;
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
    std::wstring max_ctx = get_control_text(GetDlgItem(hwnd, IDC_MAX_CONTEXT_EDIT));
    const std::uint64_t requested_ctx = parse_wide_u64(strip_wide_spaces(max_ctx));
    if (requested_ctx > fit) {
        extra_status = L"Warning: context " + std::to_wstring(requested_ctx) +
                       L" exceeds VRAM fit ~" + std::to_wstring(fit) + L"; clamped to fit.";
        max_ctx = std::to_wstring(fit);
        ::SetWindowTextW(GetDlgItem(hwnd, IDC_MAX_CONTEXT_EDIT), max_ctx.c_str());
    } else if (requested_ctx == 0) {
        // Empty (or non-numeric) field = auto: fill it with the probed fit, the
        // same thing the Auto button does. A user-set context below the fit is
        // kept as-is - append_sampling_args pins the KV pool to that window.
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

// ---------------------------------------------------------------------------
// Usage tracking (D11, reworked 2026-09-05 per user spec): the serve appends
// full-precision request records to ninfer-serve-request.jsonl next to its own
// binary by default (opt-out: --request-log-jsonl off). This block tails that
// ledger and folds request_done records into per-local-day buckets; a 3 s
// timer on the main window refreshes the read-only usage static. Every request
// the user runs flows through a serve, so this covers all of it.
//
// Display semantics (2026-09-05): the displayed totals (today/week/month/total)
// count DECODE tokens only — the prompt sum is the re-sent conversation history
// (hundreds of times larger than the output) and is not a meaningful usage
// total. The rates are rolling 10-minute averages: prompt tok/s = COMPUTED
// (non-cached) prefill tokens / 600 s — real prefill work, prefix-cache resends
// excluded ("riktig prefill/sek"); decode tok/s = completion / decode seconds
// (engine throughput).
//
// Persistence (2026-09-09): the day buckets and the consumed offset per
// ledger file live in gui-usage-state.json (atomic replace, next to this
// exe). The serve rotates its log at 10 MiB (base -> .1 -> .2; files older
// than 7 days are dropped), and the buckets must SURVIVE that — a ledger
// file alone can never reconstruct a month of usage. Each file is tracked
// by a fingerprint (its first 256 bytes, stable across the rotation
// renames), so every scan — and a GUI restart — folds only each file's new
// tail and never re-reads ledger history. week/month are calendar windows
// (the running week Mon-Sun / the running month), not rolling 7/30 days.
// Before this the tailer wiped all buckets when the base file shrank after
// a rotation, so week/month/total showed only the ~15 h in the base file
// (2026-09-09: 822K displayed vs 3.98M actually used).
// ---------------------------------------------------------------------------

struct UsageDay {
    std::uint64_t completion_tokens      = 0;
    std::uint64_t requests               = 0;
};

struct UsageTotals {
    std::uint64_t completion_tokens      = 0;
    std::uint64_t requests               = 0;
};

// Rolling 10-minute window for the live rates: one entry per request_done
// (ledger timestamp, computed prefill tokens, completion tokens, decode
// seconds). Entries age out at every 3 s scan; after a GUI restart the
// window warms from whatever new ledger tail arrives (the persisted
// per-file offsets mean history is NOT re-folded into it).
struct UsageRecent {
    std::time_t   ts;
    std::uint64_t computed_prefill_tokens;
    std::uint64_t completion_tokens;
    double        decode_seconds;
};

// Consumed ledger bytes per ledger file, plus the last scan the file was
// readable (prunes entries the serve has dropped).
struct UsageFileOffset {
    std::uint64_t offset    = 0;
    std::uint64_t last_seen = 0;  // epoch seconds of the last successful read
};

struct UsageState {
    // Ledger bytes consumed, keyed by the file's pipeline position ("base" =
    // the active file, "one" = .1 newest rotated, "two" = .2 oldest). Keyed by
    // position rather than the file's first 256 bytes: the three files can
    // share an identical head (files written before the serve started emitting
    // a per-file marker), which would collapse them into one offset and
    // re-fold the tails every scan. A rotation is detected from the base
    // shrinking below its consumed offset, and the offsets then shift
    // two<-one<-base<-0 so each file's consumed bytes follow its content down
    // the base -> .1 -> .2 pipeline. Offsets always sit on a '\n' boundary: a
    // trailing partial line is re-read on the next scan.
    std::map<std::string, UsageFileOffset> file_offsets;
    std::map<std::string, UsageDay> by_day;  // "YYYY-MM-DD", local time
    std::deque<UsageRecent> recent;            // last-10-min rate window
    // Rate-hold (user spec 2026-09-05: no call within 10 s -> rate stays as
    // before): the last-shown window sums; a rolling window would fade out the
    // moment traffic stops, so idle display holds these until new activity.
    std::uint64_t shown_prompt_tokens       = 0;
    std::uint64_t shown_completion_tokens   = 0;
    double        shown_decode_seconds      = 0.0;
    bool          shown_rate_valid          = false;
};

UsageState g_usage {};
HFONT g_usage_font = nullptr;  // larger face for the usage block (D11)
constexpr std::int64_t kUsageRateWindowSec = 600;  // "senaste 10 min"
constexpr std::int64_t kUsageActivityHoldSec = 10; // rate holds when idle (no
                                                    // request_done within 10 s)
constexpr std::uint64_t kUsageFileOffsetGraceSec = 8 * 86400;  // an offset is
                                                               // only needed
                                                               // while its file
                                                               // can exist (7
                                                               // days + 1)

std::wstring usage_state_path() { return module_dir() + L"gui-usage-state.json"; }

// Restore the persisted usage state (day buckets + per-file consumed
// offsets). A missing/corrupt file leaves everything fresh — the one-time
// fold of the existing ledger files then rebuilds the buckets.
void usage_state_load() {
    const std::optional<std::string> raw = read_text_file(usage_state_path());
    if (!raw || raw->empty()) { return; }
    nlohmann::json doc;
    try {
        doc = nlohmann::json::parse(raw->data());
    } catch (const nlohmann::json::exception&) {
        return;
    }
    if (!doc.is_object()) { return; }
    if (const nlohmann::json* days =
            doc.contains("days") ? &doc["days"] : nullptr;
        days && days->is_object()) {
        for (auto it = days->begin(); it != days->end(); ++it) {
            const nlohmann::json& v = it.value();
            if (!v.is_array() || v.size() != 2 || !v[0].is_number_unsigned() ||
                !v[1].is_number_unsigned()) {
                continue;
            }
            g_usage.by_day[it.key()] = {v[0].get<std::uint64_t>(),
                                        v[1].get<std::uint64_t>()};
        }
    }
    if (const nlohmann::json* files =
            doc.contains("files") ? &doc["files"] : nullptr;
        files && files->is_object()) {
        for (auto it = files->begin(); it != files->end(); ++it) {
            const nlohmann::json& v = it.value();
            if (!v.is_array() || v.size() != 2 || !v[0].is_number_unsigned() ||
                !v[1].is_number_unsigned()) {
                continue;
            }
            g_usage.file_offsets[it.key()] = {v[0].get<std::uint64_t>(),
                                              v[1].get<std::uint64_t>()};
        }
    }
    // A pre-2026-09-11 state keyed the offsets by the file's 256-byte head; the head
    // collision inflated those day buckets. If any loaded offset key is not a pipeline
    // tag, the whole state is from that buggy era: discard the buckets and the offsets
    // so the first scan re-folds the ledger from the start (the true totals).
    bool any_legacy_offset = false;
    for (const auto& entry : g_usage.file_offsets) {
        if (entry.first != "base" && entry.first != "one" && entry.first != "two") {
            any_legacy_offset = true;
            break;
        }
    }
    if (any_legacy_offset) {
        g_usage.file_offsets.clear();
        g_usage.by_day.clear();
    }
}

// Persist the usage state (best-effort, every scan): atomic replace so the
// GUI's crash history (WER 0xC0000409) can never leave a torn file. Day
// buckets are kept forever (a year is a few KB) so "total" stays a true
// all-time sum — the ledger itself only spans 7 days, so the buckets are the
// only long-term record.
void usage_state_save() {
    nlohmann::json days = nlohmann::json::object();
    for (const auto& entry : g_usage.by_day) {
        days[entry.first] = nlohmann::json::array(
            {entry.second.completion_tokens, entry.second.requests});
    }
    nlohmann::json files = nlohmann::json::object();
    const std::uint64_t now_epoch = static_cast<std::uint64_t>(std::time(nullptr));
    for (auto it = g_usage.file_offsets.begin(); it != g_usage.file_offsets.end();) {
        // An offset is only needed while its file can still exist: the
        // serve keeps rotated files 7 days, so an entry unseen for 8 days is
        // stale. The grace also spans transient open failures (the entry is
        // kept, not lost).
        if (it->second.last_seen + kUsageFileOffsetGraceSec < now_epoch) {
            it = g_usage.file_offsets.erase(it);
        } else {
            files[it->first] = nlohmann::json::array(
                {it->second.offset, it->second.last_seen});
            ++it;
        }
    }
    nlohmann::json doc = nlohmann::json::object();
    doc["days"] = std::move(days);
    doc["files"] = std::move(files);
    (void)write_text_file_atomic(usage_state_path(), doc.dump());
}

std::string usage_day_key_local(std::time_t seconds) {
    std::tm tm {};
    localtime_s(&tm, &seconds);
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d", tm.tm_year + 1900, tm.tm_mon + 1,
                  tm.tm_mday);
    return buf;
}

// Day keys for the running calendar window, oldest first: the week = this
// Monday through today (the week runs Mon-Sun); the month = the 1st through
// today. The week/month labels promise calendar windows, not rolling
// 7/30-day sums (2026-09-09).
std::vector<std::string> usage_calendar_day_keys(bool week) {
    const std::time_t now = std::time(nullptr);
    std::tm t {};
    localtime_s(&t, &now);
    std::tm start = t;
    if (week) {
        start.tm_mday -= (t.tm_wday + 6) % 7;  // Mon -> 0 back, Sun -> 6 back
    } else {
        start.tm_mday = 1;
    }
    const std::time_t start_ts = mktime(&start);  // normalizes the day walk
    std::vector<std::string> keys;
    for (std::time_t d = start_ts; d <= now; d += 86400) {
        keys.push_back(usage_day_key_local(d));
    }
    return keys;
}

std::wstring usage_fmt_tokens(std::uint64_t v) {
    wchar_t buf[24];
    if (v < 1000) {
        std::swprintf(buf, std::size(buf), L"%llu", static_cast<unsigned long long>(v));
    } else if (v < 1000000) {
        std::swprintf(buf, std::size(buf), L"%lluK", static_cast<unsigned long long>(
            std::llround(static_cast<double>(v) / 1000.0)));
    } else {
        std::swprintf(buf, std::size(buf), L"%.2fM", static_cast<double>(v) / 1000000.0);
    }
    return buf;
}

// Live rates over the rolling 10-minute window (user spec 2026-09-05):
//  decode tok/s = completion tokens / decode seconds of the window's requests
//                 (the engine throughput the serve stats show);
//  prompt tok/s = computed (non-cached) prefill tokens / 600 s — the real
//                 prefill work rate over the wall-clock window. Prompt resends
//                 that hit the prefix cache are NOT counted (user spec
//                 2026-09-05: "riktig prefill/sek"); the wall-clock
//                 denominator keeps the figure stable even when most of the
//                 prompt is cached.
struct UsageRate {
    std::uint64_t computed_prefill_tokens = 0;
    std::uint64_t completion_tokens       = 0;
    double        decode_seconds          = 0.0;
};

UsageRate usage_rate_window(const UsageState& state) {
    UsageRate rate;
    for (const UsageRecent& r : state.recent) {
        rate.computed_prefill_tokens += r.computed_prefill_tokens;
        rate.completion_tokens       += r.completion_tokens;
        rate.decode_seconds          += r.decode_seconds;
    }
    return rate;
}

std::wstring usage_fmt_decode_rate(const UsageRate& rate) {
    if (rate.decode_seconds <= 0.0) { return L"—"; }
    const long long tok_s = std::llround(
        static_cast<double>(rate.completion_tokens) / rate.decode_seconds);
    return std::to_wstring(tok_s);
}

// Full number with space thousands separators (no K/M shortening): 12480 ->
// "12 480" (user spec 2026-09-05: "inte avkortat som k utan tusental").
std::wstring usage_fmt_thousands(std::uint64_t v) {
    const std::wstring digits = std::to_wstring(v);
    std::wstring out;
    out.reserve(digits.size() + digits.size() / 3);
    for (std::size_t i = 0; i < digits.size(); ++i) {
        if (i > 0 && (digits.size() - i) % 3 == 0) { out += L' '; }
        out += digits[i];
    }
    return out;
}

UsageTotals usage_sum_days(const std::vector<std::string>& keys, const UsageState& state) {
    UsageTotals sum;
    for (const std::string& key : keys) {
        const auto it = state.by_day.find(key);
        if (it == state.by_day.end()) { continue; }
        sum.completion_tokens += it->second.completion_tokens;
        sum.requests          += it->second.requests;
    }
    return sum;
}

void usage_scan(HWND hwnd) {
    static bool state_loaded = false;
    if (!state_loaded) { usage_state_load(); state_loaded = true; }
    const std::wstring base = module_dir() + L"ninfer-serve-request.jsonl";
    // Oldest first: the rotated files (.2 oldest, .1 newest) are static — the
    // serve only appends to the base — so folding them first keeps the recent
    // ring (rate window) in ledger order.
    const std::array<std::wstring, 3> paths = {base + L".2", base + L".1", base};
    // Position tag per ledger file (oldest -> newest). Offsets are keyed by
    // these tags, not by file content, so identical heads never collapse two
    // files into one offset.
    const std::array<std::string, 3> tags = {"two", "one", "base"};
    const std::uint64_t now_epoch = static_cast<std::uint64_t>(std::time(nullptr));

    // Detect a rotation BEFORE reading: the serve appends only to the base, so
    // the base only ever grows -- if it is now smaller than the offset consumed
    // last scan, it rolled over and a fresh (smaller) base replaced it. On that
    // event shift the offsets down the pipeline (two<-one, one<-base, base<-0)
    // so each file's consumed bytes follow the content that moved. A base that
    // is absent this scan keeps its offset (a rotation may reappear next scan);
    // a base that reappears small then still trips this test.
    const bool base_present =
        ::GetFileAttributesW(base.c_str()) != INVALID_FILE_ATTRIBUTES;
    std::uint64_t base_size = 0;
    if (base_present) {
        HANDLE probe = ::CreateFileW(base.c_str(), GENERIC_READ,
                                     FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                     nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (probe != INVALID_HANDLE_VALUE) {
            LARGE_INTEGER size_bytes {};
            if (::GetFileSizeEx(probe, &size_bytes)) {
                base_size = static_cast<std::uint64_t>(size_bytes.QuadPart);
            }
            ::CloseHandle(probe);
        }
    }
    const auto base_it = g_usage.file_offsets.find("base");
    const std::uint64_t base_offset_prev =
        base_it != g_usage.file_offsets.end() ? base_it->second.offset : 0;
    const bool rotated = base_present && base_size < base_offset_prev;
    if (rotated) {
        const auto one_it = g_usage.file_offsets.find("one");
        const std::uint64_t one_offset_prev =
            one_it != g_usage.file_offsets.end() ? one_it->second.offset : 0;
        g_usage.file_offsets["two"].offset = one_offset_prev;
        g_usage.file_offsets["one"].offset = base_offset_prev;
        g_usage.file_offsets["base"].offset = 0;
    }

    bool any_file_present = false;
    for (std::size_t i = 0; i < paths.size(); ++i) {
        const std::wstring& path = paths[i];
        if (::GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES) {
            continue;  // rotated out and deleted (or never created)
        }
        any_file_present = true;
        HANDLE file = ::CreateFileW(path.c_str(), GENERIC_READ,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            continue;  // present but unreadable this tick: retry next scan; the
                       // saved offset (last_seen untouched) is kept
        }
        LARGE_INTEGER size_bytes {};
        if (!::GetFileSizeEx(file, &size_bytes)) { ::CloseHandle(file); continue; }
        const std::uint64_t size = static_cast<std::uint64_t>(size_bytes.QuadPart);
        // Resume from this file's consumed offset; it always sits on a '\n'
        // boundary, so a trailing partial line (only the base can leave one) is
        // re-read on the next scan.
        std::uint64_t offset = 0;
        if (const auto it = g_usage.file_offsets.find(tags[i]);
            it != g_usage.file_offsets.end()) {
            offset = it->second.offset;
            if (offset > size) { offset = size; }  // shrank: never re-fold
        }
        if (size > offset) {
            // Seek to the consumed offset before reading: CreateFileW opens at
            // byte 0 and ReadFile uses the file pointer — without the seek every
            // tick would re-read the file HEAD instead of the appended tail.
            LARGE_INTEGER seek {};
            seek.QuadPart = static_cast<LONGLONG>(offset);
            LARGE_INTEGER seek_result {};
            if (::SetFilePointerEx(file, seek, &seek_result, FILE_BEGIN) &&
                seek_result.QuadPart == seek.QuadPart) {
                const std::size_t want = static_cast<std::size_t>(size - offset);
                std::string chunk(want, '\0');
                std::size_t got = 0;
                while (got < want) {
                    DWORD n = 0;
                    if (!::ReadFile(file, chunk.data() + got,
                                     static_cast<DWORD>(want - got), &n, nullptr) ||
                        n == 0) {
                        break;
                    }
                    got += n;
                }
                chunk.resize(got);
                // Fold complete lines only; the offset stops at the last '\n'.
                std::size_t line_pos = 0;
                while (line_pos < chunk.size()) {
                    const std::size_t nl = chunk.find('\n', line_pos);
                    if (nl == std::string::npos) { break; }
                    const std::string line = chunk.substr(line_pos, nl - line_pos);
                    line_pos = nl + 1;
                    nlohmann::json record;
                    try {
                        record = nlohmann::json::parse(line);
                    } catch (const nlohmann::json::exception&) {
                        continue;  // torn line at a crash boundary: skip
                    }
                    if (record.value("event", std::string()) != "request_done" ||
                        !record.contains("result") ||
                        !record.contains("timings_seconds")) {
                        continue;
                    }
                    const nlohmann::json& result  = record["result"];
                    const nlohmann::json& timings = record["timings_seconds"];
                    const std::uint64_t computed =
                        result.value("computed_prefill_tokens", 0ULL);
                    const std::uint64_t out = result.value("completion_tokens", 0ULL);
                    const double decode_s = timings.value("decode", 0.0);
                    const std::uint64_t ts_ms = record.value("timestamp_unix_ms", 0ULL);
                    const std::time_t ts = static_cast<std::time_t>(ts_ms / 1000);
                    UsageDay& day = g_usage.by_day[usage_day_key_local(ts)];
                    day.completion_tokens += out;
                    day.requests += 1;
                    if (ts > 0) {
                        g_usage.recent.push_back({ts, computed, out, decode_s});
                    }
                }
                offset += line_pos;
            }
        }
        g_usage.file_offsets[tags[i]] = {offset, now_epoch};
        ::CloseHandle(file);
    }
    if (!any_file_present && g_usage.by_day.empty()) {
        ::SetWindowTextW(GetDlgItem(hwnd, IDC_USAGE_TEXT),
                         L"Usage: no request log (serve --request-log-jsonl off)");
        return;
    }

    // Age out entries outside the rolling 10-minute window. Order-independent:
    // the ring mirrors ledger order, which is chronological in live use but NOT
    // guaranteed on a full rescan (probe ledgers write newer lines first) —
    // popping the front only would strand stale entries behind a young one.
    const std::time_t now = std::time(nullptr);
    auto rit = g_usage.recent.begin();
    while (rit != g_usage.recent.end()) {
        if (now - rit->ts > kUsageRateWindowSec) {
            rit = g_usage.recent.erase(rit);
        } else {
            ++rit;
        }
    }

    const auto today =
        usage_sum_days({usage_day_key_local(std::time(nullptr))}, g_usage);
    const auto week  = usage_sum_days(usage_calendar_day_keys(true), g_usage);
    const auto month = usage_sum_days(usage_calendar_day_keys(false), g_usage);
    UsageTotals total;
    for (const auto& entry : g_usage.by_day) {
        total.completion_tokens += entry.second.completion_tokens;
        total.requests          += entry.second.requests;
    }
    UsageRate rate = usage_rate_window(g_usage);
    // Hold the displayed rate while the serve is idle: with no request_done in
    // the last 10 s the rolling window would fade out, so the last-shown sums
    // stand until new activity recomputes them (user spec 2026-09-05).
    {
        const std::time_t now = std::time(nullptr);
        std::time_t newest = 0;
        for (const UsageRecent& r : g_usage.recent) {
            if (r.ts > newest) { newest = r.ts; }
        }
        const bool active = newest > 0 && now - newest <= kUsageActivityHoldSec;
        if (!active && g_usage.shown_rate_valid) {
            rate.computed_prefill_tokens = g_usage.shown_prompt_tokens;
            rate.completion_tokens       = g_usage.shown_completion_tokens;
            rate.decode_seconds          = g_usage.shown_decode_seconds;
        }
        g_usage.shown_prompt_tokens     = rate.computed_prefill_tokens;
        g_usage.shown_completion_tokens = rate.completion_tokens;
        g_usage.shown_decode_seconds    = rate.decode_seconds;
        g_usage.shown_rate_valid        = true;
    }
    const std::uint64_t prompt_rate = static_cast<std::uint64_t>(std::llround(
        static_cast<double>(rate.computed_prefill_tokens) /
        static_cast<double>(kUsageRateWindowSec)));
    const std::wstring text =
        L"Usage:  today " +
        usage_fmt_tokens(today.completion_tokens) +
        L" tok   week " +
        usage_fmt_tokens(week.completion_tokens) + L"   month " +
        usage_fmt_tokens(month.completion_tokens) + L"   total " +
        usage_fmt_tokens(total.completion_tokens) + L"\r\nRate:   prompt " +
        usage_fmt_thousands(prompt_rate) + L" tok/s   ·   decode " +
        usage_fmt_decode_rate(rate) + L" tok/s   ·   last 10 min   ·   " +
        std::to_wstring(today.requests) + L" requests today";
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_USAGE_TEXT), text.c_str());
    // Persist the buckets + per-file offsets (best-effort; a failed write
    // never blocks the UI). Every 3 s tick: the persisted state is at most
    // one tick behind, so a crash/restart never re-folds or loses data.
    usage_state_save();
}

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
    auto check = [&](int id, std::wstring_view text, int x, int y, bool checked, int w = 130) {
        HWND c = ::CreateWindowExW(0, L"BUTTON", text.data(), WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                                   x, y, w, 20, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
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
    button(IDC_AUTO_BUTTON, L"AutoContext", 480, 34, 90);
    // Default thinking budget (Row 2 free space): token cap for thinking-enabled
    // requests; empty field = no default (the serve leaves thinking uncapped).
    label(0, L"Think budget:", 590, 36);
    edit(IDC_THINKING_BUDGET_EDIT, 690, 34, 60, 22);

    // Row 3: sampling (the 5 Qwen params grouped, then the greedy toggle)
    label(0, L"Temperature:", 8, 64);
    edit(IDC_TEMPERATURE_EDIT, 100, 62, 60, 22);
    label(0, L"Top-p:", 174, 64);
    edit(IDC_TOP_P_EDIT, 222, 62, 60, 22);
    label(0, L"Top-k:", 296, 64);
    edit(IDC_TOP_K_EDIT, 344, 62, 60, 22);
    label(0, L"Min-p:", 418, 64);
    edit(IDC_MIN_P_EDIT, 460, 62, 48, 22);
    label(0, L"Presence:", 520, 64);
    edit(IDC_PRESENCE_PENALTY_EDIT, 592, 62, 48, 22);
    check(IDC_GREEDY_CHECK, L"Greedy", 660, 64, false);

    // Row 4: behavior flags
    check(IDC_THINKING_CHECK, L"Thinking", 8, 92, true);
    label(0, L"Vision:", 150, 94);
    combo(IDC_VISION_COMBO, 200, 92, 64, {L"Off", L"GPU", L"CPU"});
    label(0, L"Spec:", 290, 94);
    combo(IDC_SPEC_COMBO, 330, 92, 90, {L"(none)", L"mtp", L"dflash", L"dflash2", L"dflash2+ngram"});
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
    // 32k context, fp8 KV cache, thinking-mode sampling, greedy decoding, MTP speculation
    // with 5 drafts (measured +38% decode tok/s vs sampling d=3). Users can override any
    // of these per run.
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_MAX_CONTEXT_EDIT), L"32768");
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_MAX_NEW_EDIT), L"8192");
    // Default thinking budget (2026-09-05 request): 4096 tokens; the field stays
    // user-editable and an empty field omits --default-thinking-budget entirely.
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_THINKING_BUDGET_EDIT), L"4096");
    ::SendMessageW(GetDlgItem(hwnd, IDC_KV_DTYPE_COMBO), CB_SETCURSEL, 3, 0);  // fp8
    // Thinking is checked by default (Row 4); seed the sampling fields with the
    // thinking preset. The checkbox handler re-applies the matching preset on toggle.
    apply_sampling_preset(hwnd, true);
    ::SendMessageW(GetDlgItem(hwnd, IDC_SPEC_COMBO), CB_SETCURSEL, 1, 0);      // mtp
    ::SetWindowTextW(GetDlgItem(hwnd, IDC_DRAFT_TOKENS_EDIT), L"5");
    // Vision defaults to CPU (task #19): no GPU VRAM needed and A/B-verified
    // near-identical to the GPU path on this device (2026-09-02).
    ::SendMessageW(GetDlgItem(hwnd, IDC_VISION_COMBO), CB_SETCURSEL, 2, 0);    // cpu

    // Status bar (plain static)
    HWND status = ::CreateWindowExW(0, L"STATIC", L"Idle",
                                    WS_CHILD | WS_VISIBLE | SS_LEFT | SS_NOPREFIX,
                                    8, 150, 884, 18, hwnd, reinterpret_cast<HMENU>(IDC_STATUS_TEXT),
                                    GetModuleHandleW(nullptr), nullptr);
    ::SendMessageW(status, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);

    // Usage block (D11): read-only, refreshed by the WM_TIMER usage scan. A
    // larger face than the default GUI font — the two-line stats are meant to be
    // read at a glance.
    g_usage_font = ::CreateFontW(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                                 DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                 CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
    HWND usage = ::CreateWindowExW(0, L"STATIC", L"Usage: scanning…",
                                   WS_CHILD | WS_VISIBLE | SS_LEFT | SS_NOPREFIX | WS_BORDER,
                                   8, 174, 884, 44, hwnd, reinterpret_cast<HMENU>(IDC_USAGE_TEXT),
                                   GetModuleHandleW(nullptr), nullptr);
    ::SendMessageW(usage, WM_SETFONT, reinterpret_cast<WPARAM>(g_usage_font), TRUE);

    // Tooltips: one tooltip control for all the parameter controls; concise one-line
    // explanations (user request 2026-09-09). The texts are string literals (process
    // lifetime), so the lpszText pointers the control reads stay valid.
    HWND tip = ::CreateWindowExW(0, TOOLTIPS_CLASSW, nullptr, TTS_ALWAYSTIP, 0, 0, 0, 0, hwnd,
                                 nullptr, GetModuleHandleW(nullptr), nullptr);
    ::SendMessageW(tip, TTM_SETMAXTIPWIDTH, 380, 0);
    const struct {
        int id;
        std::wstring_view text;
    } tips[] = {
        {IDC_MODEL_EDIT, L"Path to the .ninfer model file the serve loads."},
        {IDC_MODEL_BROWSE, L"Open a file dialog to choose the model file."},
        {IDC_MAX_CONTEXT_EDIT,
         L"Total context window in tokens (prompt + output). Bigger = more history, more VRAM."},
        {IDC_MAX_NEW_EDIT, L"Maximum new tokens to generate in one response."},
        {IDC_KV_DTYPE_COMBO,
         L"KV cache precision. fp8 fits the most context per byte of VRAM; bf16 is the most "
         L"accurate."},
        {IDC_AUTO_BUTTON, L"Probe VRAM and fill Max context with the largest size that fits."},
        {IDC_THINKING_BUDGET_EDIT, L"Cap on thinking tokens per request. Empty = no cap."},
        {IDC_TEMPERATURE_EDIT, L"Sampling temperature: lower = more deterministic, higher = more varied."},
        {IDC_TOP_P_EDIT, L"Nucleus sampling: draw only from the top p of the probability mass (0-1)."},
        {IDC_TOP_K_EDIT, L"Consider only the top k most likely tokens when sampling. 0 = off."},
        {IDC_MIN_P_EDIT, L"Keep tokens whose probability is at least min-p times the top token. 0 = off."},
        {IDC_PRESENCE_PENALTY_EDIT, L"Presence penalty: nudge away from tokens already used (-2 to 2)."},
        {IDC_GREEDY_CHECK, L"Force temperature 0 (exact argmax). Overrides all sampling settings."},
        {IDC_THINKING_CHECK, L"Enable reasoning/thinking mode. Also sets the temperature/top-p preset."},
        {IDC_VISION_COMBO, L"Vision input: Off, GPU (CUDA encode), or CPU (host-RAM weights, no VRAM)."},
        {IDC_SPEC_COMBO, L"Speculative decoding backend for faster generation. (none) = off; mtp / dflash / dflash2 (dflash2 needs the dflash2 artifact; defaults to it when the model supports it); dflash2+ngram = dflash2 with n-gram self-speculation on."},
        {IDC_DRAFT_TOKENS_EDIT, L"Draft tokens to propose per step (needs a Spec backend)."},
        {IDC_LM_HEAD_DRAFT_CHECK, L"Use the optimized draft head for speculative proposals (faster drafts)."},
        {IDC_SEED_EDIT, L"Fixed sampling seed (reproducible output). Empty = a new random seed per request."},
        {IDC_SYSTEM_PROMPT_BUTTON, L"Edit the default system prompt used when the client sends none."},
        {IDC_TEMPLATE_BUTTON, L"View and manage the chat template baked into the artifact."},
        {IDC_HOST_EDIT, L"Local address the serve listens on."},
        {IDC_PORT_EDIT, L"TCP port the serve listens on."},
        {IDC_SERVE_BUTTON, L"Start the inference server with these settings."},
        {IDC_STOP_BUTTON, L"Stop the running server (the GUI stays open)."},
        {IDC_EXIT_BUTTON, L"Close the GUI and stop all running ninfer servers."},
    };
    for (const auto& t : tips) {
        HWND c = ::GetDlgItem(hwnd, t.id);
        if (c == nullptr) { continue; }
        TOOLINFOW ti {};
        ti.cbSize   = sizeof(TOOLINFOW);
        ti.hwnd     = hwnd;
        ti.uFlags   = TTF_IDISHWND;
        ti.uId      = reinterpret_cast<UINT_PTR>(c);
        ti.lpszText = const_cast<LPWSTR>(t.text.data());
        ::SendMessageW(tip, TTM_ADDTOOLW, 0, reinterpret_cast<LPARAM>(&ti));
    }
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
                // Restore the last used settings (file), before the command-line
                // prefill (CLI) applied in WinMain — CLI beats file beats defaults.
                apply_saved_settings(hwnd);
                // Usage block (D11): 3 s refresh of the serve's request ledger.
                ::SetTimer(hwnd, 1, 3000, nullptr);
            } else if (info->kind == 1) {
                create_sp_controls(hwnd);
            } else {
                create_ct_controls(hwnd, info);
            }
            return 0;
        }

        case WM_TIMER: {
            // Usage block refresh (D11); the timer exists only on the main window.
            if (::GetWindowLongPtrW(hwnd, GWLP_USERDATA) == 0) {
                usage_scan(hwnd);
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
                if (!picked.empty()) {
                    ::SetWindowTextW(GetDlgItem(hwnd, IDC_MODEL_EDIT), picked.c_str());
                    apply_model_sizing_defaults(hwnd, picked);
                }
                return 0;
            }
            if (id == IDC_AUTO_BUTTON) {
                const std::wstring model = get_control_text(GetDlgItem(hwnd, IDC_MODEL_EDIT));
                if (model.empty()) {
                    set_status(hwnd, L"Select a model first");
                    return 0;
                }
                // Re-probe when the model or a sizing control changed (cached otherwise);
                // fill Max context with the probed fit so Serve pins the KV pool to
                // what VRAM holds.
                const std::uint32_t fit = ensure_probe(hwnd, model);
                if (fit > 0) {
                    ::SetWindowTextW(GetDlgItem(hwnd, IDC_MAX_CONTEXT_EDIT),
                                     std::to_wstring(fit).c_str());
                }
                return 0;
            }
            if (id == IDC_THINKING_CHECK) {
                // Repopulate the sampling fields with the Qwen preset for the new mode
                // (checked -> thinking, unchecked -> Instruct). Runs only on the toggle,
                // so a manual edit survives until the next mode change.
                const bool thinking =
                    ::SendMessageW(GetDlgItem(hwnd, IDC_THINKING_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED;
                apply_sampling_preset(hwnd, thinking);
                return 0;
            }
            if (id == IDC_SERVE_BUTTON && !g_child.running.load()) {
                // Persist the last-used form state before launching (on-use save;
                // best-effort — a write failure never blocks the launch).
                save_current_settings(hwnd);
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
            // Main window: persist the current form state (best-effort; a write
            // failure never blocks exit) so settings survive across sessions even
            // if the user never clicks Serve. Then kill every ninfer engine in any
            // session before this GUI exits, so closing one window never leaves
            // orphaned GPU work behind. This also terminates this GUI's own child;
            // the cleanup below still joins its watcher.
            save_current_settings(hwnd);
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
            if (user == 0) {
                ::KillTimer(hwnd, 1);  // usage refresh (D11)
                if (g_usage_font != nullptr) {
                    ::DeleteObject(g_usage_font);
                    g_usage_font = nullptr;
                }
                ::PostQuitMessage(0);   // main window only
            }
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

    // 920x284: the one-shot Answer/Log panes are gone, so the window is just the
    // parameter rows + status bar + the read-only usage block (D11).
    HWND hwnd = ::CreateWindowExW(0, wc.lpszClassName, kAppIdentity,
                                  WS_OVERLAPPEDWINDOW | WS_MINIMIZEBOX, CW_USEDEFAULT, CW_USEDEFAULT,
                                  920, 284, nullptr, nullptr, h_instance, nullptr);
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
                      kAppIdentity, MB_ICONWARNING);
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
