#define UNICODE 1
#define _UNICODE 1
#include <windows.h>
#include <windowsx.h>
#include <commctrl.h>
#include <winhttp.h>
#include <string>
#include <string_view>
#include <vector>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cwchar>
#include <ctime>
#include <deque>
#include <fstream>
#include <map>
#include <optional>
#include <atomic>
#include <thread>
#include <nlohmann/json.hpp>

#pragma comment(lib, "winhttp.lib")
#pragma comment(lib, "comctl32.lib")

namespace {
constexpr wchar_t kWindowClass[]  = L"NinferGui";
constexpr wchar_t kWindowTitle[]  = L"Wallawalla Launcher";

// ---------------------------------------------------------------------------
// Control IDs (stable: later tasks read these to build the serve argv,
// launch the server, and probe it)
// ---------------------------------------------------------------------------
constexpr int IDC_MODEL_EDIT                = 100;
constexpr int IDC_MODEL_BROWSE              = 101;  // (file-picker button; wired in T8)
constexpr int IDC_HOST_EDIT                 = 102;
constexpr int IDC_PORT_EDIT                 = 103;
constexpr int IDC_MAX_CONTEXT_EDIT          = 104;
constexpr int IDC_KV_CAPACITY_EDIT          = 105;  // "auto" or a number
constexpr int IDC_KV_DTYPE_COMBO            = 106;  // bf16 | int8 | fp8 | nvfp4 | k8v4
constexpr int IDC_HEADROOM_EDIT             = 107;  // MiB
constexpr int IDC_MAX_NEW_EDIT              = 108;
constexpr int IDC_TEMPERATURE_EDIT          = 109;
constexpr int IDC_TOPP_EDIT                 = 110;
constexpr int IDC_TOPK_EDIT                 = 111;
constexpr int IDC_MINP_EDIT                 = 112;
constexpr int IDC_PRESENCE_EDIT             = 113;
constexpr int IDC_FREQUENCY_EDIT            = 114;
constexpr int IDC_GREEDY_CHECK              = 115;
constexpr int IDC_THINKING_CHECK            = 116;  // default checked
constexpr int IDC_DEFAULT_THINK_BUDGET_EDIT = 117;
constexpr int IDC_VISION_COMBO              = 118;  // 0=Off 1=On (GPU) 2=Offload
constexpr int IDC_SPEC_COMBO                = 119;  // 0=off 1=mtp 2=dflash 3=dflash2
constexpr int IDC_DRAFT_TOKENS_EDIT         = 120;
constexpr int IDC_LM_HEAD_DRAFT_CHECK       = 121;
constexpr int IDC_SEED_EDIT                 = 122;
constexpr int IDC_PRESERVE_THINKING_CHECK   = 123;  // default checked
constexpr int IDC_REQUEST_LOG_EDIT          = 124;
constexpr int IDC_PROBE_BUTTON              = 125;
constexpr int IDC_LAUNCH_BUTTON             = 126;
constexpr int IDC_STOP_BUTTON               = 127;
constexpr int IDC_STATUS                    = 128;  // STATIC status line (SS_NOTIFY)
constexpr int IDC_USAGE_TEXT                = 129;  // read-only usage block (D11)

// Extended section (below Core): serve tuning flags. Empty edits and
// unchecked boxes are omitted from the serve argv (engine defaults apply).
constexpr int IDC_MODEL_ID_EDIT             = 200;
constexpr int IDC_MAX_CONCURRENCY_EDIT      = 201;
constexpr int IDC_THINK_BUDGET_MSG_EDIT     = 202;
constexpr int IDC_THINK_BUDGET_POLICY_COMBO = 203;  // strict | clamp | ignore
constexpr int IDC_MAX_THINK_BUDGET_EDIT     = 204;
constexpr int IDC_MEDIA_CACHE_EDIT          = 205;
constexpr int IDC_MEDIA_LIVE_EDIT           = 206;
constexpr int IDC_MEDIA_PREPROC_THREADS_EDIT = 207;
constexpr int IDC_PREFILL_CHUNK_EDIT        = 208;
constexpr int IDC_CORS_CHECK                = 209;
constexpr int IDC_API_KEY_EDIT              = 210;
constexpr int IDC_NO_CUDA_GRAPH_CHECK       = 211;
constexpr int IDC_LOG_LEVEL_COMBO           = 212;  // error | warn | info | debug
constexpr int IDC_USAGE_CHUNK_CHOICE_CHECK  = 213;

// Advanced section (collapsed group box below Extended; the "Show advanced"
// toggle shows/hides it): prefix-cache, ngram, cache-tap, and request-store
// tuning flags. Empty edits and unchecked boxes are omitted from the serve
// argv.
constexpr int IDC_ADV_TOGGLE_CHECK            = 300;
constexpr int IDC_PREFIX_CACHE_FILE_EDIT      = 301;
constexpr int IDC_USE_ORIG_PREFIX_CHECK       = 302;
constexpr int IDC_MAX_SHARED_PREFIXES_EDIT    = 303;
constexpr int IDC_MAX_PRIVATE_CONT_EDIT       = 304;
constexpr int IDC_LONG_ANCHOR_SPACING_EDIT    = 305;
constexpr int IDC_MAX_LONG_ANCHORS_EDIT       = 306;
constexpr int IDC_HOST_CACHE_MIB_EDIT         = 307;
constexpr int IDC_HOST_KV_MIB_EDIT            = 308;
constexpr int IDC_HOST_STATE_SLOTS_EDIT       = 309;
constexpr int IDC_DEV_SNAP_SLOTS_EDIT         = 310;
constexpr int IDC_DEV_STATE_SLOTS_EDIT        = 311;
constexpr int IDC_NGRAM_DRAFT_EDIT            = 312;
constexpr int IDC_NGRAM_MIN_MATCH_EDIT        = 313;
constexpr int IDC_NGRAM_NATIVE_CHECK          = 314;
constexpr int IDC_NGRAM_ARCHIVE_MIB_EDIT      = 315;
constexpr int IDC_NGRAM_SESSION_MIB_EDIT      = 316;
constexpr int IDC_CACHE_TAP_LADDER_EDIT       = 317;
constexpr int IDC_CACHE_TAP_MIN_GAP_EDIT      = 318;
constexpr int IDC_CACHE_TAPS_PER_REQ_EDIT     = 319;
constexpr int IDC_RESP_STORE_MAX_RECORDS_EDIT = 320;
constexpr int IDC_RESP_STORE_MAX_MIB_EDIT     = 321;
constexpr int IDC_MAX_REQUEST_MIB_EDIT        = 322;
constexpr int IDC_MAX_PENDING_REQ_EDIT        = 323;
constexpr int IDC_PENDING_TIMEOUT_MS_EDIT     = 324;
constexpr int IDC_CHAT_TEMPLATE_EDIT          = 325;
constexpr int IDC_CONTEXT_COST_PRESETS_EDIT   = 326;
constexpr int IDC_TOLERANT_TOOL_CALLS_CHECK   = 327;
constexpr int IDC_ADV_GROUP                   = 328;  // "Advanced" group box
// The 24 Advanced labels carry dialog IDs so set_advanced_visible can hide
// and show them with the group (statics without an ID would stay visible
// in the compact view). They are handed out sequentially in creation order.
constexpr int IDC_ADV_LABEL_BASE              = 400;  // 24 labels: 400..423

// Sampling preset buttons (sit in the right margin of the sampling rows) and
// the Auto-context button on the max-context row; 130-132 are the first IDs
// free after the Core block (400..423 are the Advanced labels).
constexpr int IDC_PRESET_THINKING             = 130;
constexpr int IDC_PRESET_INSTRUCT             = 131;
constexpr int IDC_AUTO_CONTEXT_BTN            = 132;

// Posted by the serve watcher thread when the child process exits;
// wParam: the exit code.
constexpr UINT WM_APP_DONE = WM_APP + 1;
// WM_TIMER id for the serve /health poll (id 1 is the usage scan).
constexpr UINT_PTR kHealthTimerId = 2;

// gui-settings.ini lives next to the exe; every Core control key sits in
// the [Core] section.
constexpr wchar_t kSettingsSection[] = L"Core";

// The directory containing this executable, with a trailing backslash.
std::wstring module_dir() {
    wchar_t module[MAX_PATH] = {};
    const DWORD got = ::GetModuleFileNameW(nullptr, module, MAX_PATH);
    if (got == 0 || got >= MAX_PATH) { return {}; }
    std::wstring path(module, got);
    const auto slash = path.find_last_of(L"\\/");
    return slash == std::wstring::npos ? std::wstring{} : path.substr(0, slash + 1);
}

std::wstring settings_path() { return module_dir() + L"gui-settings.ini"; }

std::wstring get_control_text(HWND hwnd) {
    const int len = ::GetWindowTextLengthW(hwnd);
    std::wstring out(static_cast<std::size_t>(len), L'\0');
    if (len > 0) { ::GetWindowTextW(hwnd, out.data(), len + 1); }
    return out;
}

void set_status(HWND hwnd, std::wstring_view text) {
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_STATUS), std::wstring(text).c_str());
}

// ---------------------------------------------------------------------------
// Wide/UTF-8 conversion + command-line quoting (ported from the reference GUI).
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

// Locate a sibling executable next to this one.
std::wstring find_sibling(const wchar_t* exe_name) {
    const std::wstring dir = module_dir();
    if (dir.empty()) { return {}; }
    const std::wstring path = dir + exe_name;
    return ::GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES ? path : std::wstring{};
}

// Declared here (defined below): the serve launch's pre-launch port-busy
// check calls it before its definition.
std::string http_get(const std::wstring& host, const std::wstring& port, const std::string& path);

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

std::wstring build_command_line(const std::vector<std::wstring>& args) {
    std::wstring command_line;
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (i != 0) { command_line.push_back(L' '); }
        command_line += quote_arg(args[i]);
    }
    return command_line;
}

// The selected item text of a CBS_DROPDOWNLIST combo (empty if nothing
// selected).
std::wstring combo_text(HWND hwnd, int id) {
    const HWND c = ::GetDlgItem(hwnd, id);
    const int sel = static_cast<int>(::SendMessageW(c, CB_GETCURSEL, 0, 0));
    if (sel < 0) { return {}; }
    const int len = static_cast<int>(::SendMessageW(c, CB_GETLBTEXTLEN, sel, 0));
    std::wstring out(static_cast<std::size_t>(len), L'\0');
    ::SendMessageW(c, CB_GETLBTEXT, sel, reinterpret_cast<LPARAM>(out.data()));
    return out;
}

// ---------------------------------------------------------------------------
// Child process state (serve only). `running` is touched by the UI thread and
// the watcher thread; the atomic keeps them consistent without a mutex.
// ---------------------------------------------------------------------------

struct ChildProcess {
    HANDLE process = nullptr;
    std::atomic<bool> running {false};
    std::thread watcher;  // joinable while running
};
ChildProcess g_child {};
// host/port of the serve this GUI launched (drives the /health poll and the
// already-running check).
std::wstring g_serve_host;
std::wstring g_serve_port;

// The full ninfer-serve.exe argument vector from the current control values
// (element 0 is the serve exe path). Flags follow the canonical launch order;
// blank fields are omitted so the engine default applies.
std::vector<std::wstring> build_serve_argv(HWND h, const std::wstring& model) {
    const auto g = [&](int id) -> std::wstring {
        return get_control_text(GetDlgItem(h, id));
    };
    std::vector<std::wstring> a { find_sibling(L"ninfer-serve.exe") };
    a.push_back(model);                       // positional artifact path (quoted by build_command_line)
    const std::wstring host = g(IDC_HOST_EDIT);
    const std::wstring port = g(IDC_PORT_EDIT);
    if (!host.empty()) { a.push_back(L"--host"); a.push_back(host); }
    if (!port.empty()) { a.push_back(L"--port"); a.push_back(port); }
    const std::wstring max_ctx = g(IDC_MAX_CONTEXT_EDIT);
    if (!max_ctx.empty()) { a.push_back(L"--max-context"); a.push_back(max_ctx); }
    const std::wstring kv_cap = g(IDC_KV_CAPACITY_EDIT);
    if (!kv_cap.empty()) { a.push_back(L"--kv-capacity"); a.push_back(kv_cap); } // "auto" or N
    a.push_back(L"--kv-dtype"); a.push_back(combo_text(h, IDC_KV_DTYPE_COMBO));
    // --vram-headroom-mib requires --kv-capacity auto (the serve rejects it otherwise);
    // with a fixed kv-capacity it is omitted.
    const std::wstring headroom = g(IDC_HEADROOM_EDIT);
    if (!headroom.empty() && kv_cap == L"auto") {
        a.push_back(L"--vram-headroom-mib"); a.push_back(headroom);
    }
    const std::wstring max_new = g(IDC_MAX_NEW_EDIT);
    if (!max_new.empty()) { a.push_back(L"--default-max-tokens"); a.push_back(max_new); }
    // Sampling (always emitted for determinism):
    const std::wstring temperature = g(IDC_TEMPERATURE_EDIT);
    const std::wstring topp = g(IDC_TOPP_EDIT);
    const std::wstring topk  = g(IDC_TOPK_EDIT);
    const std::wstring minp  = g(IDC_MINP_EDIT);
    const std::wstring presence = g(IDC_PRESENCE_EDIT);
    const std::wstring frequency = g(IDC_FREQUENCY_EDIT);
    const bool greedy = ::SendMessageW(GetDlgItem(h, IDC_GREEDY_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED;
    if (greedy) { a.push_back(L"--greedy"); }
    if (!temperature.empty()) { a.push_back(L"--temperature"); a.push_back(temperature); }
    if (!topp.empty()) { a.push_back(L"--top-p"); a.push_back(topp); }
    if (!topk.empty())  { a.push_back(L"--top-k"); a.push_back(topk); }
    if (!minp.empty())  { a.push_back(L"--min-p"); a.push_back(minp); }
    if (!presence.empty()) { a.push_back(L"--presence-penalty"); a.push_back(presence); }
    if (!frequency.empty()) { a.push_back(L"--frequency-penalty"); a.push_back(frequency); }
    const std::wstring seed = g(IDC_SEED_EDIT);
    if (!seed.empty()) { a.push_back(L"--seed"); a.push_back(seed); }
    const bool thinking = ::SendMessageW(GetDlgItem(h, IDC_THINKING_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED;
    if (!thinking) { a.push_back(L"--no-thinking"); }
    const std::wstring think_budget = g(IDC_DEFAULT_THINK_BUDGET_EDIT);
    if (!think_budget.empty()) { a.push_back(L"--default-thinking-budget"); a.push_back(think_budget); }
    const int vision = static_cast<int>(::SendMessageW(GetDlgItem(h, IDC_VISION_COMBO), CB_GETCURSEL, 0, 0));
    if (vision >= 1) { a.push_back(L"--vision"); }
    if (vision == 2) { a.push_back(L"--vision-offload"); a.push_back(L"on"); }
    const int spec = static_cast<int>(::SendMessageW(GetDlgItem(h, IDC_SPEC_COMBO), CB_GETCURSEL, 0, 0));
    static const wchar_t* kSpec[] = {L"", L"mtp", L"dflash", L"dflash2"};
    if (spec > 0) {
        a.push_back(L"--spec"); a.push_back(kSpec[spec]);
        const std::wstring draft = g(IDC_DRAFT_TOKENS_EDIT);
        if (!draft.empty()) { a.push_back(L"--draft-tokens"); a.push_back(draft); }
        if (::SendMessageW(GetDlgItem(h, IDC_LM_HEAD_DRAFT_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED) {
            a.push_back(L"--lm-head-draft");
        }
    }
    if (::SendMessageW(GetDlgItem(h, IDC_PRESERVE_THINKING_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED) {
        a.push_back(L"--preserve-thinking");
    }
    // Request log is a yes/no checkbox with a fixed name (see Global Constraints).
    const bool req_log_on =
        ::SendMessageW(GetDlgItem(h, IDC_REQUEST_LOG_EDIT), BM_GETCHECK, 0, 0) == BST_CHECKED;
    if (req_log_on) { a.push_back(L"--request-log-jsonl"); a.push_back(L"requests.jsonl"); }
    // Extended controls (empty edits / unchecked boxes are omitted).
    const std::wstring model_id = g(IDC_MODEL_ID_EDIT);
    if (!model_id.empty()) { a.push_back(L"--model-id"); a.push_back(model_id); }
    const std::wstring max_conc = g(IDC_MAX_CONCURRENCY_EDIT);
    if (!max_conc.empty()) { a.push_back(L"--max-concurrency"); a.push_back(max_conc); }
    const std::wstring think_msg = g(IDC_THINK_BUDGET_MSG_EDIT);
    if (!think_msg.empty()) { a.push_back(L"--thinking-budget-message"); a.push_back(think_msg); }
    const std::wstring think_policy = combo_text(h, IDC_THINK_BUDGET_POLICY_COMBO);
    if (!think_policy.empty()) { a.push_back(L"--thinking-budget-policy"); a.push_back(think_policy); }
    const std::wstring max_think = g(IDC_MAX_THINK_BUDGET_EDIT);
    if (!max_think.empty()) { a.push_back(L"--max-thinking-budget"); a.push_back(max_think); }
    const std::wstring media_cache = g(IDC_MEDIA_CACHE_EDIT);
    if (!media_cache.empty()) { a.push_back(L"--media-cache-mib"); a.push_back(media_cache); }
    const std::wstring media_live = g(IDC_MEDIA_LIVE_EDIT);
    if (!media_live.empty()) { a.push_back(L"--media-live-mib"); a.push_back(media_live); }
    const std::wstring media_threads = g(IDC_MEDIA_PREPROC_THREADS_EDIT);
    if (!media_threads.empty()) { a.push_back(L"--media-preprocess-threads"); a.push_back(media_threads); }
    const std::wstring prefill = g(IDC_PREFILL_CHUNK_EDIT);
    if (!prefill.empty()) { a.push_back(L"--prefill-chunk"); a.push_back(prefill); }
    if (::SendMessageW(GetDlgItem(h, IDC_CORS_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED) { a.push_back(L"--cors"); }
    const std::wstring api_key = g(IDC_API_KEY_EDIT);
    if (!api_key.empty()) { a.push_back(L"--api-key"); a.push_back(api_key); }
    if (::SendMessageW(GetDlgItem(h, IDC_NO_CUDA_GRAPH_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED) { a.push_back(L"--no-cuda-graph"); }
    const std::wstring log_level = combo_text(h, IDC_LOG_LEVEL_COMBO);
    if (!log_level.empty()) { a.push_back(L"--log-level"); a.push_back(log_level); }
    if (::SendMessageW(GetDlgItem(h, IDC_USAGE_CHUNK_CHOICE_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED) {
        a.push_back(L"--usage-chunk-choice");
    }
    // Advanced controls (the group is hidden by default; Win32 reads work on
    // hidden windows, so the canonical defaults take effect on Launch).
    const std::wstring prefix_file = g(IDC_PREFIX_CACHE_FILE_EDIT);
    if (!prefix_file.empty()) { a.push_back(L"--prefix-cache-file"); a.push_back(prefix_file); }
    if (::SendMessageW(GetDlgItem(h, IDC_USE_ORIG_PREFIX_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED) {
        a.push_back(L"--use-original-prefix-caching");
    }
    const std::wstring shared_prefixes = g(IDC_MAX_SHARED_PREFIXES_EDIT);
    if (!shared_prefixes.empty()) { a.push_back(L"--max-shared-prefixes"); a.push_back(shared_prefixes); }
    const std::wstring private_cont = g(IDC_MAX_PRIVATE_CONT_EDIT);
    if (!private_cont.empty()) { a.push_back(L"--max-private-continuations"); a.push_back(private_cont); }
    const std::wstring anchor_spacing = g(IDC_LONG_ANCHOR_SPACING_EDIT);
    if (!anchor_spacing.empty()) { a.push_back(L"--long-anchor-spacing"); a.push_back(anchor_spacing); }
    const std::wstring max_anchors = g(IDC_MAX_LONG_ANCHORS_EDIT);
    if (!max_anchors.empty()) { a.push_back(L"--max-long-anchors-per-continuation"); a.push_back(max_anchors); }
    const std::wstring host_cache = g(IDC_HOST_CACHE_MIB_EDIT);
    if (!host_cache.empty()) { a.push_back(L"--host-cache-mib"); a.push_back(host_cache); }
    const std::wstring host_kv = g(IDC_HOST_KV_MIB_EDIT);
    if (!host_kv.empty()) { a.push_back(L"--host-kv-mib"); a.push_back(host_kv); }
    const std::wstring host_slots = g(IDC_HOST_STATE_SLOTS_EDIT);
    if (!host_slots.empty()) { a.push_back(L"--host-state-slots"); a.push_back(host_slots); }
    const std::wstring dev_snap = g(IDC_DEV_SNAP_SLOTS_EDIT);
    if (!dev_snap.empty()) { a.push_back(L"--device-snapshot-slots"); a.push_back(dev_snap); }
    const std::wstring dev_state = g(IDC_DEV_STATE_SLOTS_EDIT);
    if (!dev_state.empty()) { a.push_back(L"--device-state-slots"); a.push_back(dev_state); }
    const std::wstring ngram_draft = g(IDC_NGRAM_DRAFT_EDIT);
    if (!ngram_draft.empty()) { a.push_back(L"--ngram-draft-tokens"); a.push_back(ngram_draft); }
    const std::wstring ngram_match = g(IDC_NGRAM_MIN_MATCH_EDIT);
    if (!ngram_match.empty()) { a.push_back(L"--ngram-min-match"); a.push_back(ngram_match); }
    if (::SendMessageW(GetDlgItem(h, IDC_NGRAM_NATIVE_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED) {
        a.push_back(L"--ngram-native-sessions");
    }
    const std::wstring ngram_archive = g(IDC_NGRAM_ARCHIVE_MIB_EDIT);
    if (!ngram_archive.empty()) { a.push_back(L"--ngram-archive-mib"); a.push_back(ngram_archive); }
    const std::wstring ngram_session = g(IDC_NGRAM_SESSION_MIB_EDIT);
    if (!ngram_session.empty()) { a.push_back(L"--ngram-session-mib"); a.push_back(ngram_session); }
    const std::wstring tap_ladder = g(IDC_CACHE_TAP_LADDER_EDIT);
    if (!tap_ladder.empty()) { a.push_back(L"--cache-tap-ladder"); a.push_back(tap_ladder); }
    const std::wstring tap_gap = g(IDC_CACHE_TAP_MIN_GAP_EDIT);
    if (!tap_gap.empty()) { a.push_back(L"--cache-tap-min-gap"); a.push_back(tap_gap); }
    const std::wstring taps_req = g(IDC_CACHE_TAPS_PER_REQ_EDIT);
    if (!taps_req.empty()) { a.push_back(L"--cache-taps-per-request"); a.push_back(taps_req); }
    const std::wstring store_records = g(IDC_RESP_STORE_MAX_RECORDS_EDIT);
    if (!store_records.empty()) { a.push_back(L"--response-store-max-records"); a.push_back(store_records); }
    const std::wstring store_mib = g(IDC_RESP_STORE_MAX_MIB_EDIT);
    if (!store_mib.empty()) { a.push_back(L"--response-store-max-mib"); a.push_back(store_mib); }
    const std::wstring max_req_mib = g(IDC_MAX_REQUEST_MIB_EDIT);
    if (!max_req_mib.empty()) { a.push_back(L"--max-request-mib"); a.push_back(max_req_mib); }
    const std::wstring max_pending = g(IDC_MAX_PENDING_REQ_EDIT);
    if (!max_pending.empty()) { a.push_back(L"--max-pending-requests"); a.push_back(max_pending); }
    const std::wstring pending_to = g(IDC_PENDING_TIMEOUT_MS_EDIT);
    if (!pending_to.empty()) { a.push_back(L"--pending-timeout-ms"); a.push_back(pending_to); }
    const std::wstring chat_template = g(IDC_CHAT_TEMPLATE_EDIT);
    if (!chat_template.empty()) { a.push_back(L"--chat-template"); a.push_back(chat_template); }
    const std::wstring cost_presets = g(IDC_CONTEXT_COST_PRESETS_EDIT);
    if (!cost_presets.empty()) { a.push_back(L"--context-cost-presets"); a.push_back(cost_presets); }
    if (::SendMessageW(GetDlgItem(h, IDC_TOLERANT_TOOL_CALLS_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED) {
        a.push_back(L"--tolerant-tool-calls");
    }
    return a;
}

// A sampling preset fills exactly the five sampling edits; it leaves the
// greedy / thinking checkboxes, the vision + spec combos, and every other
// control untouched. "Thinking" (the on-launch defaults) reverts the fields.
void apply_preset(HWND hwnd, bool thinking) {
    static const wchar_t* const kValues[2][5] = {
        {L"1.0", L"0.95", L"20", L"0.0", L"0.0"},  // Thinking
        {L"0.7", L"0.80", L"20", L"0.0", L"1.5"},  // Instruct
    };
    static const int kIds[5] = {IDC_TEMPERATURE_EDIT, IDC_TOPP_EDIT,
                                IDC_TOPK_EDIT, IDC_MINP_EDIT, IDC_PRESENCE_EDIT};
    const wchar_t* const* v = kValues[thinking ? 0 : 1];
    for (int i = 0; i < 5; ++i) {
        ::SetWindowTextW(::GetDlgItem(hwnd, kIds[i]), v[i]);
    }
}

// ---------------------------------------------------------------------------
// VRAM probe. Runs the sibling CLI in --probe mode, which loads the model
// weights (~32 GiB) and reports the KV fit ceiling: the largest KV capacity
// that fits current free VRAM. Because of the weight load the probe is
// expensive and only runs on the explicit Probe / Auto buttons -- never on
// Launch (which would otherwise load the model a second time before serving).
// The Probe button is informational: it never mutates a control.
// ---------------------------------------------------------------------------

// Read an edit as a non-negative integer; std::nullopt for a blank or
// non-numeric field (kv-capacity "auto" is not a number).
std::optional<std::uint64_t> parse_uint_field(HWND hwnd, int id) {
    const std::wstring text = get_control_text(::GetDlgItem(hwnd, id));
    if (text.empty()) { return std::nullopt; }
    for (wchar_t ch : text) {
        if (ch < L'0' || ch > L'9') { return std::nullopt; }
    }
    try { return std::stoull(text); }
    catch (const std::out_of_range&) { return std::nullopt; }
}

// Cached VRAM probe: the last probe key and its KV fit ceiling in tokens, plus
// the free VRAM after the weight load for the status line. The key combines
// the model with every control the probe forwards, so re-probing only happens
// when one of them changes (a ~32 GiB load), which is what keeps a second
// Probe click instant.
std::wstring g_probe_key;
std::uint32_t g_probe_fit = 0;
std::uint64_t g_probe_free_after_bytes = 0;

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

// Synchronously launch the sibling CLI in probe mode and return the KV fit
// ceiling (tokens) for the current free VRAM. The probe always runs in
// auto-KV mode, so the fit is a VRAM ceiling, not capped by --max-context.
// The forwarded sizing flags mirror build_serve_argv so the ceiling matches
// the launch; --max-context and --kv-capacity are deliberately NOT forwarded.
// Returns 0 on failure -- the status line carries the reason.
std::uint32_t run_probe(HWND hwnd, const std::wstring& model, const std::wstring& key) {
    const std::wstring cli = find_sibling(L"ninfer.exe");
    if (cli.empty()) {
        set_status(hwnd, L"Error: ninfer.exe not found next to ninfer-gui.exe");
        return 0;
    }

    const int vision   = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_VISION_COMBO), CB_GETCURSEL, 0, 0));
    const int kv_idx   = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_KV_DTYPE_COMBO), CB_GETCURSEL, 0, 0));
    const int spec_idx = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_SPEC_COMBO), CB_GETCURSEL, 0, 0));
    const std::wstring draft = get_control_text(GetDlgItem(hwnd, IDC_DRAFT_TOKENS_EDIT));
    const bool lm_head_draft =
        ::SendMessageW(GetDlgItem(hwnd, IDC_LM_HEAD_DRAFT_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED;
    std::vector<std::wstring> args { cli, model, L"--probe" };
    if (vision >= 1) { args.push_back(L"--vision"); }
    if (vision == 2) { args.push_back(L"--vision-offload"); args.push_back(L"on"); }
    static const wchar_t* kKvDtypes[] = {L"bf16", L"int8", L"fp8", L"nvfp4", L"k8v4"};
    if (kv_idx >= 0) { args.push_back(L"--kv-dtype"); args.push_back(kKvDtypes[kv_idx]); }
    static const wchar_t* kSpecs[] = {L"", L"mtp", L"dflash", L"dflash2"};
    if (spec_idx > 0) {
        args.push_back(L"--spec"); args.push_back(kSpecs[spec_idx]);
        if (!draft.empty()) { args.push_back(L"--draft-tokens"); args.push_back(draft); }
        if (lm_head_draft) { args.push_back(L"--lm-head-draft"); }
    }
    const std::wstring prefill = get_control_text(GetDlgItem(hwnd, IDC_PREFILL_CHUNK_EDIT));
    if (!prefill.empty()) { args.push_back(L"--prefill-chunk"); args.push_back(prefill); }
    const std::wstring ngram_draft = get_control_text(GetDlgItem(hwnd, IDC_NGRAM_DRAFT_EDIT));
    if (!ngram_draft.empty()) { args.push_back(L"--ngram-draft-tokens"); args.push_back(ngram_draft); }
    const std::wstring ngram_match = get_control_text(GetDlgItem(hwnd, IDC_NGRAM_MIN_MATCH_EDIT));
    if (!ngram_match.empty()) { args.push_back(L"--ngram-min-match"); args.push_back(ngram_match); }
    const std::wstring headroom = get_control_text(GetDlgItem(hwnd, IDC_HEADROOM_EDIT));
    if (!headroom.empty()) {
        const std::optional<std::uint64_t> headroom_mib = parse_uint_field(hwnd, IDC_HEADROOM_EDIT);
        if (headroom_mib && *headroom_mib > 0) {
            args.push_back(L"--vram-headroom-mib"); args.push_back(headroom);
        }
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
    g_probe_free_after_bytes = free_after;
    set_status(hwnd, L"~" + std::to_wstring(fit) + L" tokens fit (free after weights: " +
                           format_gib(free_after) + L")");
    return g_probe_fit;
}

// The probe cache key: the model plus every control the probe forwards, so a
// changed key means a different probe command line (and a re-probe).
std::wstring build_probe_key(HWND hwnd, const std::wstring& model) {
    const int vision   = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_VISION_COMBO), CB_GETCURSEL, 0, 0));
    const int kv_idx   = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_KV_DTYPE_COMBO), CB_GETCURSEL, 0, 0));
    const int spec_idx = static_cast<int>(::SendMessageW(GetDlgItem(hwnd, IDC_SPEC_COMBO), CB_GETCURSEL, 0, 0));
    const bool lm_head_draft =
        ::SendMessageW(GetDlgItem(hwnd, IDC_LM_HEAD_DRAFT_CHECK), BM_GETCHECK, 0, 0) == BST_CHECKED;
    std::wstring key = model;
    key.push_back(L'\x01'); key.append(std::to_wstring(vision));
    key.push_back(L'\x01'); key.append(std::to_wstring(kv_idx));
    key.push_back(L'\x01'); key.append(std::to_wstring(spec_idx));
    key.push_back(L'\x01'); key.append(get_control_text(GetDlgItem(hwnd, IDC_DRAFT_TOKENS_EDIT)));
    key.push_back(L'\x01'); key.push_back(lm_head_draft ? L'1' : L'0');
    key.push_back(L'\x01'); key.append(get_control_text(GetDlgItem(hwnd, IDC_PREFILL_CHUNK_EDIT)));
    key.push_back(L'\x01'); key.append(get_control_text(GetDlgItem(hwnd, IDC_NGRAM_DRAFT_EDIT)));
    key.push_back(L'\x01'); key.append(get_control_text(GetDlgItem(hwnd, IDC_NGRAM_MIN_MATCH_EDIT)));
    key.push_back(L'\x01'); key.append(get_control_text(GetDlgItem(hwnd, IDC_HEADROOM_EDIT)));
    return key;
}

// Return the cached probe fit for the current (model, sizing controls),
// re-probing when any part of the key changed.
std::uint32_t ensure_probe(HWND hwnd, const std::wstring& model) {
    const std::wstring key = build_probe_key(hwnd, model);
    if (key == g_probe_key && g_probe_fit > 0) { return g_probe_fit; }
    set_status(hwnd, L"Measuring VRAM...");
    ::UpdateWindow(hwnd);
    return run_probe(hwnd, model, key);
}

// ---------------------------------------------------------------------------
// Usage / consumption tracking (D11). Tail the serve's request-log JSONL,
// fold `request_done` records into per-day buckets + a rolling 10-minute rate
// window, and refresh a read-only usage block on a 3 s timer. State persists
// to gui-usage-state.json next to the exe so a restart never re-folds history.
// ---------------------------------------------------------------------------

struct UsageDay {
    std::uint64_t completion_tokens = 0;
    std::uint64_t requests          = 0;
};
struct UsageTotals {
    std::uint64_t completion_tokens = 0;
    std::uint64_t requests          = 0;
};
struct UsageRecent {
    std::time_t   ts;
    std::uint64_t computed_prefill_tokens;
    std::uint64_t completion_tokens;
    double        decode_seconds;
};
struct UsageState {
    std::uint64_t consumed_offset  = 0;   // bytes read from the request log
    std::time_t   offset_last_seen = 0;   // epoch sec of last successful read
    std::map<std::string, UsageDay> by_day;  // "YYYY-MM-DD" local time
    std::deque<UsageRecent> recent;             // last-10-min rate window
    std::uint64_t shown_prompt_tokens   = 0;   // rate-hold (idle: keep last rate)
    std::uint64_t shown_completion_tokens = 0;
    double        shown_decode_seconds  = 0.0;
    bool          shown_rate_valid      = false;
};
UsageState g_usage {};
HFONT g_usage_font = nullptr;  // larger face for the usage block
constexpr std::int64_t kUsageRateWindowSec   = 600;  // "senaste 10 min"
constexpr std::int64_t kUsageActivityHoldSec = 10;   // hold rate when idle

std::optional<std::string> read_text_file(const std::wstring& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { return std::nullopt; }
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}
bool write_text_file_atomic(const std::wstring& path, const std::string& bytes) {
    const std::wstring tmp = path + L".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) { return false; }
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!out) { ::DeleteFileW(tmp.c_str()); return false; }
    }
    if (!::MoveFileExW(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        ::DeleteFileW(tmp.c_str());
        return false;
    }
    return true;
}

std::wstring usage_state_path() { return module_dir() + L"gui-usage-state.json"; }

// Missing/corrupt state leaves everything fresh; the first scan folds the
// existing log tail and rebuilds the buckets.
void usage_state_load() {
    const std::optional<std::string> raw = read_text_file(usage_state_path());
    if (!raw || raw->empty()) { return; }
    nlohmann::json doc;
    try { doc = nlohmann::json::parse(raw->data()); }
    catch (const nlohmann::json::exception&) { return; }
    if (!doc.is_object()) { return; }
    if (doc.contains("offset") && doc["offset"].is_number_unsigned()) {
        g_usage.consumed_offset = doc["offset"].get<std::uint64_t>();
    }
    if (doc.contains("days") && doc["days"].is_object()) {
        for (auto it = doc["days"].begin(); it != doc["days"].end(); ++it) {
            const nlohmann::json& v = it.value();
            if (!v.is_array() || v.size() != 2 || !v[0].is_number_unsigned() ||
                !v[1].is_number_unsigned()) { continue; }
            g_usage.by_day[it.key()] =
                {v[0].get<std::uint64_t>(), v[1].get<std::uint64_t>()};
        }
    }
}

// Best-effort, every scan: atomic replace so a crash never leaves a torn file.
// Day buckets are kept forever (a year is a few KB) so "total" stays a true
// all-time sum.
void usage_state_save() {
    nlohmann::json days = nlohmann::json::object();
    for (const auto& entry : g_usage.by_day) {
        days[entry.first] = nlohmann::json::array(
            {entry.second.completion_tokens, entry.second.requests});
    }
    nlohmann::json doc = nlohmann::json::object();
    doc["offset"]           = g_usage.consumed_offset;
    doc["offset_last_seen"] = static_cast<std::uint64_t>(g_usage.offset_last_seen);
    doc["days"]             = std::move(days);
    (void)write_text_file_atomic(usage_state_path(), doc.dump());
}

std::string usage_day_key_local(std::time_t seconds) {
    std::tm tm {};
    localtime_s(&tm, &seconds);
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d", tm.tm_year + 1900,
                  tm.tm_mon + 1, tm.tm_mday);
    return buf;
}

// Day keys for the running calendar window, oldest first: week = this Monday
// through today; month = the 1st through today.
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
        std::swprintf(buf, std::size(buf), L"%llu",
                      static_cast<unsigned long long>(v));
    } else if (v < 1000000) {
        std::swprintf(buf, std::size(buf), L"%lluK",
                      static_cast<unsigned long long>(
                          std::llround(static_cast<double>(v) / 1000.0)));
    } else {
        std::swprintf(buf, std::size(buf), L"%.2fM",
                      static_cast<double>(v) / 1000000.0);
    }
    return buf;
}

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
    if (rate.decode_seconds <= 0.0) { return L"\u2014"; }
    const long long tok_s = std::llround(
        static_cast<double>(rate.completion_tokens) / rate.decode_seconds);
    return std::to_wstring(tok_s);
}
// Full number with space thousands separators: 12480 -> "12 480".
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
UsageTotals usage_sum_days(const std::vector<std::string>& keys,
                           const UsageState& state) {
    UsageTotals sum;
    for (const std::string& key : keys) {
        const auto it = state.by_day.find(key);
        if (it == state.by_day.end()) { continue; }
        sum.completion_tokens += it->second.completion_tokens;
        sum.requests          += it->second.requests;
    }
    return sum;
}

// Path of the serve's request log: the "Request log (requests.jsonl)"
// checkbox selects on/off for both the --request-log-jsonl flag (see
// build_serve_argv) and this read path; off means there is no log. The file
// always sits next to the exe.
std::wstring request_log_path(HWND hwnd) {
    if (::SendMessageW(::GetDlgItem(hwnd, IDC_REQUEST_LOG_EDIT), BM_GETCHECK, 0, 0)
            != BST_CHECKED) {
        return {};
    }
    const std::wstring dir = module_dir();
    return dir.empty() ? std::wstring{} : dir + L"requests.jsonl";
}

void usage_scan(HWND hwnd) {
    static bool state_loaded = false;
    if (!state_loaded) { usage_state_load(); state_loaded = true; }
    const std::wstring path = request_log_path(hwnd);
    const bool present = !path.empty() &&
        ::GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES;

    std::uint64_t offset = g_usage.consumed_offset;
    if (present) {
        HANDLE file = ::CreateFileW(path.c_str(), GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file != INVALID_HANDLE_VALUE) {
            LARGE_INTEGER size_bytes {};
            if (::GetFileSizeEx(file, &size_bytes)) {
                const std::uint64_t size =
                    static_cast<std::uint64_t>(size_bytes.QuadPart);
                if (offset > size) { offset = size; }  // shrank: never re-fold
                if (size > offset) {
                    // Seek to the consumed offset: CreateFileW opens at byte 0,
                    // so without the seek every tick re-reads the HEAD.
                    LARGE_INTEGER seek {}; seek.QuadPart = static_cast<LONGLONG>(offset);
                    LARGE_INTEGER seek_result {};
                    if (::SetFilePointerEx(file, seek, &seek_result, FILE_BEGIN) &&
                        seek_result.QuadPart == seek.QuadPart) {
                        const std::size_t want =
                            static_cast<std::size_t>(size - offset);
                        std::string chunk(want, '\0');
                        std::size_t got = 0;
                        while (got < want) {
                            DWORD n = 0;
                            if (!::ReadFile(file, chunk.data() + got,
                                             static_cast<DWORD>(want - got), &n,
                                             nullptr) ||
                                n == 0) { break; }
                            got += n;
                        }
                        chunk.resize(got);
                        // Fold complete lines only; the offset stops at the last
                        // '\n', so a trailing partial line re-reads next scan.
                        std::size_t line_pos = 0;
                        while (line_pos < chunk.size()) {
                            const std::size_t nl = chunk.find('\n', line_pos);
                            if (nl == std::string::npos) { break; }
                            const std::string line =
                                chunk.substr(line_pos, nl - line_pos);
                            line_pos = nl + 1;
                            try {
                                nlohmann::json record =
                                    nlohmann::json::parse(line);
                                if (record.value("event", std::string()) !=
                                        "request_done" ||
                                    !record.contains("result") ||
                                    !record.contains("timings_seconds")) {
                                    continue;
                                }
                                const nlohmann::json& result  =
                                    record["result"];
                                const nlohmann::json& timings =
                                    record["timings_seconds"];
                                const std::uint64_t computed =
                                    result.value("computed_prefill_tokens",
                                                 0ULL);
                                const std::uint64_t out =
                                    result.value("completion_tokens", 0ULL);
                                const double decode_s =
                                    timings.value("decode", 0.0);
                                const std::uint64_t ts_ms =
                                    record.value("timestamp_unix_ms", 0ULL);
                                const std::time_t ts =
                                    static_cast<std::time_t>(ts_ms / 1000);
                                UsageDay& day =
                                    g_usage.by_day[usage_day_key_local(ts)];
                                day.completion_tokens += out;
                                day.requests += 1;
                                if (ts > 0) {
                                    g_usage.recent.push_back(
                                        {ts, computed, out, decode_s});
                                }
                            } catch (const nlohmann::json::exception&) {
                                // A malformed or mistyped line must not take
                                // down the timer loop: skip it, advance on.
                                continue;
                            }
                        }
                        offset += line_pos;
                    }
                }
                g_usage.offset_last_seen =
                    static_cast<std::time_t>(std::time(nullptr));
            }
            ::CloseHandle(file);
        }
    }
    g_usage.consumed_offset = offset;

    if (!present && g_usage.by_day.empty()) {
        if (HWND u = ::GetDlgItem(hwnd, IDC_USAGE_TEXT)) {
            ::SetWindowTextW(u,
                L"Usage: no request log (serve --request-log-jsonl)");
        }
        return;
    }

    // Age out entries outside the rolling 10-minute window (scan the whole
    // ring; pop-front-only would strand stale entries behind a young one).
    const std::time_t now = std::time(nullptr);
    auto rit = g_usage.recent.begin();
    while (rit != g_usage.recent.end()) {
        if (now - rit->ts > kUsageRateWindowSec) { rit = g_usage.recent.erase(rit); }
        else { ++rit; }
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
    // Hold the displayed rate while idle: with no request_done in the last
    // 10 s the rolling window would fade out, so the last-shown sums stand
    // until new activity recomputes them.
    {
        std::time_t newest = 0;
        for (const UsageRecent& r : g_usage.recent) {
            if (r.ts > newest) { newest = r.ts; }
        }
        const bool active = newest > 0 &&
                            now - newest <= kUsageActivityHoldSec;
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
    // "\u00b7" is the middle dot as a universal char name: source stays
    // ASCII while the literal still renders as a middle dot at runtime.
    const std::wstring text =
        L"Usage:  today " + usage_fmt_tokens(today.completion_tokens) +
        L" tok   week " + usage_fmt_tokens(week.completion_tokens) +
        L"   month " + usage_fmt_tokens(month.completion_tokens) +
        L"   total " + usage_fmt_tokens(total.completion_tokens) +
        L"\r\nRate:   prompt " + usage_fmt_thousands(prompt_rate) +
        L" tok/s   \u00b7   decode " + usage_fmt_decode_rate(rate) +
        L" tok/s   \u00b7   last 10 min   \u00b7   " +
        std::to_wstring(today.requests) + L" requests today";
    if (HWND u = ::GetDlgItem(hwnd, IDC_USAGE_TEXT)) { ::SetWindowTextW(u, text.c_str()); }
    // Persist buckets + offset (best-effort; at most one 3 s tick stale).
    usage_state_save();
}

void create_usage_block(HWND hwnd) {
    g_usage_font = ::CreateFontW(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Segoe UI");
    HWND usage = ::CreateWindowExW(0, L"STATIC", L"Usage: scanning\u2026",
        WS_CHILD | WS_VISIBLE | SS_LEFT | SS_NOPREFIX | WS_BORDER,
        8, 414, 744, 44, hwnd,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(IDC_USAGE_TEXT)),
        ::GetModuleHandleW(nullptr), nullptr);
    ::SendMessageW(usage, WM_SETFONT, reinterpret_cast<WPARAM>(g_usage_font), TRUE);
}

// ---------------------------------------------------------------------------
// Settings persistence (gui-settings.ini next to the exe). load_settings
// overrides a default only when its INI key is present; save_settings writes
// the current value of every Core control.
// ---------------------------------------------------------------------------
void load_settings(HWND hwnd) {
    const std::wstring path = settings_path();
    if (hwnd == nullptr || path.empty()) { return; }

    auto load_edit = [&](int id, const wchar_t* key) {
        wchar_t buffer[1024] = {};
        // Return value > 0 only when the key exists (and is non-empty); a
        // missing key leaves the default untouched.
        if (::GetPrivateProfileStringW(kSettingsSection, key, L"", buffer,
                                       static_cast<DWORD>(std::size(buffer)), path.c_str()) > 0) {
            ::SetWindowTextW(::GetDlgItem(hwnd, id), buffer);
        }
    };
    auto load_combo = [&](int id, const wchar_t* key, int default_index) {
        HWND c = ::GetDlgItem(hwnd, id);
        // A missing key yields default_index (the current selection), so the
        // saved index only overrides the default when present; out-of-range
        // indices are skipped (forward-compatible).
        const int saved = ::GetPrivateProfileIntW(kSettingsSection, key, default_index,
                                                  path.c_str());
        const int count = static_cast<int>(::SendMessageW(c, CB_GETCOUNT, 0, 0));
        if (saved >= 0 && saved < count) { ::SendMessageW(c, CB_SETCURSEL, saved, 0); }
    };
    auto load_check = [&](int id, const wchar_t* key, int default_state) {
        const int saved = ::GetPrivateProfileIntW(kSettingsSection, key, default_state,
                                                  path.c_str());
        ::SendMessageW(::GetDlgItem(hwnd, id), BM_SETCHECK,
                       saved ? BST_CHECKED : BST_UNCHECKED, 0);
    };

    load_edit(IDC_MODEL_EDIT, L"model");
    load_edit(IDC_HOST_EDIT, L"host");
    load_edit(IDC_PORT_EDIT, L"port");
    load_edit(IDC_MAX_CONTEXT_EDIT, L"max_context");
    load_edit(IDC_KV_CAPACITY_EDIT, L"kv_capacity");
    load_combo(IDC_KV_DTYPE_COMBO, L"kv_dtype", 2);
    load_edit(IDC_HEADROOM_EDIT, L"headroom");
    load_edit(IDC_MAX_NEW_EDIT, L"max_new");
    load_edit(IDC_TEMPERATURE_EDIT, L"temperature");
    load_edit(IDC_TOPP_EDIT, L"topp");
    load_edit(IDC_TOPK_EDIT, L"topk");
    load_edit(IDC_MINP_EDIT, L"minp");
    load_edit(IDC_PRESENCE_EDIT, L"presence");
    load_edit(IDC_FREQUENCY_EDIT, L"frequency");
    load_check(IDC_GREEDY_CHECK, L"greedy", 0);
    load_check(IDC_THINKING_CHECK, L"thinking", 1);
    load_edit(IDC_DEFAULT_THINK_BUDGET_EDIT, L"think_budget");
    load_combo(IDC_VISION_COMBO, L"vision", 2);  // default Offload (governing request)
    load_combo(IDC_SPEC_COMBO, L"spec", 3);
    load_edit(IDC_DRAFT_TOKENS_EDIT, L"draft_tokens");
    load_check(IDC_LM_HEAD_DRAFT_CHECK, L"lm_head_draft", 0);
    load_edit(IDC_SEED_EDIT, L"seed");
    load_check(IDC_PRESERVE_THINKING_CHECK, L"preserve_thinking", 1);
    load_check(IDC_REQUEST_LOG_EDIT, L"request_log", 1);
    // Extended controls.
    load_edit(IDC_MODEL_ID_EDIT, L"model_id");
    load_edit(IDC_MAX_CONCURRENCY_EDIT, L"max_concurrency");
    load_edit(IDC_THINK_BUDGET_MSG_EDIT, L"think_budget_message");
    load_combo(IDC_THINK_BUDGET_POLICY_COMBO, L"think_budget_policy", 1);  // clamp
    load_edit(IDC_MAX_THINK_BUDGET_EDIT, L"max_thinking_budget");
    load_edit(IDC_MEDIA_CACHE_EDIT, L"media_cache");
    load_edit(IDC_MEDIA_LIVE_EDIT, L"media_live");
    load_edit(IDC_MEDIA_PREPROC_THREADS_EDIT, L"media_preprocess_threads");
    load_edit(IDC_PREFILL_CHUNK_EDIT, L"prefill_chunk");
    load_check(IDC_CORS_CHECK, L"cors", 0);
    load_edit(IDC_API_KEY_EDIT, L"api_key");
    load_check(IDC_NO_CUDA_GRAPH_CHECK, L"no_cuda_graph", 0);
    load_combo(IDC_LOG_LEVEL_COMBO, L"log_level", 2);  // info
    load_check(IDC_USAGE_CHUNK_CHOICE_CHECK, L"usage_chunk_choice", 0);
    // Advanced controls.
    load_check(IDC_ADV_TOGGLE_CHECK, L"adv_toggle", 0);
    load_edit(IDC_PREFIX_CACHE_FILE_EDIT, L"prefix_cache_file");
    load_check(IDC_USE_ORIG_PREFIX_CHECK, L"use_orig_prefix", 0);
    load_edit(IDC_MAX_SHARED_PREFIXES_EDIT, L"max_shared_prefixes");
    load_edit(IDC_MAX_PRIVATE_CONT_EDIT, L"max_private_contin");
    load_edit(IDC_LONG_ANCHOR_SPACING_EDIT, L"long_anchor_spacing");
    load_edit(IDC_MAX_LONG_ANCHORS_EDIT, L"max_long_anchors");
    load_edit(IDC_HOST_CACHE_MIB_EDIT, L"host_cache_mib");
    load_edit(IDC_HOST_KV_MIB_EDIT, L"host_kv_mib");
    load_edit(IDC_HOST_STATE_SLOTS_EDIT, L"host_state_slots");
    load_edit(IDC_DEV_SNAP_SLOTS_EDIT, L"dev_snap_slots");
    load_edit(IDC_DEV_STATE_SLOTS_EDIT, L"dev_state_slots");
    load_edit(IDC_NGRAM_DRAFT_EDIT, L"ngram_draft");
    load_edit(IDC_NGRAM_MIN_MATCH_EDIT, L"ngram_min_match");
    load_check(IDC_NGRAM_NATIVE_CHECK, L"ngram_native", 0);
    load_edit(IDC_NGRAM_ARCHIVE_MIB_EDIT, L"ngram_archive_mib");
    load_edit(IDC_NGRAM_SESSION_MIB_EDIT, L"ngram_session_mib");
    load_edit(IDC_CACHE_TAP_LADDER_EDIT, L"cache_tap_ladder");
    load_edit(IDC_CACHE_TAP_MIN_GAP_EDIT, L"cache_tap_min_gap");
    load_edit(IDC_CACHE_TAPS_PER_REQ_EDIT, L"cache_taps_per_req");
    load_edit(IDC_RESP_STORE_MAX_RECORDS_EDIT, L"resp_store_max_records");
    load_edit(IDC_RESP_STORE_MAX_MIB_EDIT, L"resp_store_max_mib");
    load_edit(IDC_MAX_REQUEST_MIB_EDIT, L"max_request_mib");
    load_edit(IDC_MAX_PENDING_REQ_EDIT, L"max_pending_req");
    load_edit(IDC_PENDING_TIMEOUT_MS_EDIT, L"pending_timeout_ms");
    load_edit(IDC_CHAT_TEMPLATE_EDIT, L"chat_template");
    load_edit(IDC_CONTEXT_COST_PRESETS_EDIT, L"context_cost_presets");
    load_check(IDC_TOLERANT_TOOL_CALLS_CHECK, L"tolerant_tool_calls", 0);
}

void save_settings(HWND hwnd) {
    const std::wstring path = settings_path();
    if (hwnd == nullptr || path.empty()) { return; }

    auto save_edit = [&](int id, const wchar_t* key) {
        ::WritePrivateProfileStringW(kSettingsSection, key,
                                     get_control_text(::GetDlgItem(hwnd, id)).c_str(),
                                     path.c_str());
    };
    auto save_combo = [&](int id, const wchar_t* key) {
        const LRESULT sel = ::SendMessageW(::GetDlgItem(hwnd, id), CB_GETCURSEL, 0, 0);
        const int value = sel == CB_ERR ? 0 : static_cast<int>(sel);
        wchar_t buffer[16] = {};
        ::swprintf(buffer, std::size(buffer), L"%d", value);
        ::WritePrivateProfileStringW(kSettingsSection, key, buffer, path.c_str());
    };
    auto save_check = [&](int id, const wchar_t* key) {
        ::WritePrivateProfileStringW(kSettingsSection, key,
                                     ::CheckDlgButton(hwnd, id, BST_CHECKED) ? L"1" : L"0",
                                     path.c_str());
    };

    save_edit(IDC_MODEL_EDIT, L"model");
    save_edit(IDC_HOST_EDIT, L"host");
    save_edit(IDC_PORT_EDIT, L"port");
    save_edit(IDC_MAX_CONTEXT_EDIT, L"max_context");
    save_edit(IDC_KV_CAPACITY_EDIT, L"kv_capacity");
    save_combo(IDC_KV_DTYPE_COMBO, L"kv_dtype");
    save_edit(IDC_HEADROOM_EDIT, L"headroom");
    save_edit(IDC_MAX_NEW_EDIT, L"max_new");
    save_edit(IDC_TEMPERATURE_EDIT, L"temperature");
    save_edit(IDC_TOPP_EDIT, L"topp");
    save_edit(IDC_TOPK_EDIT, L"topk");
    save_edit(IDC_MINP_EDIT, L"minp");
    save_edit(IDC_PRESENCE_EDIT, L"presence");
    save_edit(IDC_FREQUENCY_EDIT, L"frequency");
    save_check(IDC_GREEDY_CHECK, L"greedy");
    save_check(IDC_THINKING_CHECK, L"thinking");
    save_edit(IDC_DEFAULT_THINK_BUDGET_EDIT, L"think_budget");
    save_combo(IDC_VISION_COMBO, L"vision");
    save_combo(IDC_SPEC_COMBO, L"spec");
    save_edit(IDC_DRAFT_TOKENS_EDIT, L"draft_tokens");
    save_check(IDC_LM_HEAD_DRAFT_CHECK, L"lm_head_draft");
    save_edit(IDC_SEED_EDIT, L"seed");
    save_check(IDC_PRESERVE_THINKING_CHECK, L"preserve_thinking");
    save_check(IDC_REQUEST_LOG_EDIT, L"request_log");
    // Extended controls.
    save_edit(IDC_MODEL_ID_EDIT, L"model_id");
    save_edit(IDC_MAX_CONCURRENCY_EDIT, L"max_concurrency");
    save_edit(IDC_THINK_BUDGET_MSG_EDIT, L"think_budget_message");
    save_combo(IDC_THINK_BUDGET_POLICY_COMBO, L"think_budget_policy");
    save_edit(IDC_MAX_THINK_BUDGET_EDIT, L"max_thinking_budget");
    save_edit(IDC_MEDIA_CACHE_EDIT, L"media_cache");
    save_edit(IDC_MEDIA_LIVE_EDIT, L"media_live");
    save_edit(IDC_MEDIA_PREPROC_THREADS_EDIT, L"media_preprocess_threads");
    save_edit(IDC_PREFILL_CHUNK_EDIT, L"prefill_chunk");
    save_check(IDC_CORS_CHECK, L"cors");
    save_edit(IDC_API_KEY_EDIT, L"api_key");
    save_check(IDC_NO_CUDA_GRAPH_CHECK, L"no_cuda_graph");
    save_combo(IDC_LOG_LEVEL_COMBO, L"log_level");
    save_check(IDC_USAGE_CHUNK_CHOICE_CHECK, L"usage_chunk_choice");
    // Advanced controls.
    save_check(IDC_ADV_TOGGLE_CHECK, L"adv_toggle");
    save_edit(IDC_PREFIX_CACHE_FILE_EDIT, L"prefix_cache_file");
    save_check(IDC_USE_ORIG_PREFIX_CHECK, L"use_orig_prefix");
    save_edit(IDC_MAX_SHARED_PREFIXES_EDIT, L"max_shared_prefixes");
    save_edit(IDC_MAX_PRIVATE_CONT_EDIT, L"max_private_contin");
    save_edit(IDC_LONG_ANCHOR_SPACING_EDIT, L"long_anchor_spacing");
    save_edit(IDC_MAX_LONG_ANCHORS_EDIT, L"max_long_anchors");
    save_edit(IDC_HOST_CACHE_MIB_EDIT, L"host_cache_mib");
    save_edit(IDC_HOST_KV_MIB_EDIT, L"host_kv_mib");
    save_edit(IDC_HOST_STATE_SLOTS_EDIT, L"host_state_slots");
    save_edit(IDC_DEV_SNAP_SLOTS_EDIT, L"dev_snap_slots");
    save_edit(IDC_DEV_STATE_SLOTS_EDIT, L"dev_state_slots");
    save_edit(IDC_NGRAM_DRAFT_EDIT, L"ngram_draft");
    save_edit(IDC_NGRAM_MIN_MATCH_EDIT, L"ngram_min_match");
    save_check(IDC_NGRAM_NATIVE_CHECK, L"ngram_native");
    save_edit(IDC_NGRAM_ARCHIVE_MIB_EDIT, L"ngram_archive_mib");
    save_edit(IDC_NGRAM_SESSION_MIB_EDIT, L"ngram_session_mib");
    save_edit(IDC_CACHE_TAP_LADDER_EDIT, L"cache_tap_ladder");
    save_edit(IDC_CACHE_TAP_MIN_GAP_EDIT, L"cache_tap_min_gap");
    save_edit(IDC_CACHE_TAPS_PER_REQ_EDIT, L"cache_taps_per_req");
    save_edit(IDC_RESP_STORE_MAX_RECORDS_EDIT, L"resp_store_max_records");
    save_edit(IDC_RESP_STORE_MAX_MIB_EDIT, L"resp_store_max_mib");
    save_edit(IDC_MAX_REQUEST_MIB_EDIT, L"max_request_mib");
    save_edit(IDC_MAX_PENDING_REQ_EDIT, L"max_pending_req");
    save_edit(IDC_PENDING_TIMEOUT_MS_EDIT, L"pending_timeout_ms");
    save_edit(IDC_CHAT_TEMPLATE_EDIT, L"chat_template");
    save_edit(IDC_CONTEXT_COST_PRESETS_EDIT, L"context_cost_presets");
    save_check(IDC_TOLERANT_TOOL_CALLS_CHECK, L"tolerant_tool_calls");
}

// ---------------------------------------------------------------------------
// Controls (two-column form: label column at x=8 / x=380, controls at
// x=132 / x=510; rows every 30 px from y=10)
// ---------------------------------------------------------------------------

void create_status(HWND hwnd) {
    const HFONT font = static_cast<HFONT>(::GetStockObject(DEFAULT_GUI_FONT));
    RECT rc = {};
    ::GetClientRect(hwnd, &rc);
    HWND status = ::CreateWindowExW(0, L"STATIC", L"Idle",
                                    WS_CHILD | WS_VISIBLE | SS_NOTIFY,
                                    8, rc.bottom - 28, rc.right - 16, 24, hwnd,
                                    reinterpret_cast<HMENU>(static_cast<INT_PTR>(IDC_STATUS)),
                                    ::GetModuleHandleW(nullptr), nullptr);
    ::SendMessageW(status, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
}

void create_core_controls(HWND hwnd) {
    const HFONT font = static_cast<HFONT>(::GetStockObject(DEFAULT_GUI_FONT));

    auto label = [&](const wchar_t* text, int x, int y) {
        HWND h = ::CreateWindowExW(0, L"STATIC", text, WS_CHILD | WS_VISIBLE | SS_LEFT,
                                   x, y, 120, 18, hwnd, nullptr,
                                   ::GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(h, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
    };
    auto edit = [&](int id, const wchar_t* text, int x, int y, int w) {
        HWND e = ::CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", text,
                                   WS_CHILD | WS_VISIBLE | WS_TABSTOP | ES_AUTOHSCROLL,
                                   x, y, w, 22, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                                   ::GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(e, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
        return e;
    };
    auto combo = [&](int id, int x, int y, int w, const wchar_t* const* items, int count) {
        HWND c = ::CreateWindowExW(0, L"COMBOBOX", L"",
                                   WS_CHILD | WS_VISIBLE | WS_TABSTOP | CBS_DROPDOWNLIST,
                                   x, y, w, 200, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                                   ::GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(c, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
        for (int i = 0; i < count; ++i) {
            ::SendMessageW(c, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(items[i]));
        }
        ::SendMessageW(c, CB_SETCURSEL, 0, 0);
        return c;
    };
    auto check = [&](int id, const wchar_t* text, int x, int y, int w, bool checked) {
        HWND c = ::CreateWindowExW(0, L"BUTTON", text,
                                   WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_AUTOCHECKBOX,
                                   x, y, w, 20, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                                   ::GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(c, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
        if (checked) { ::SendMessageW(c, BM_SETCHECK, BST_CHECKED, 0); }
        return c;
    };
    auto button = [&](int id, const wchar_t* text, int x, int y, int w) {
        HWND b = ::CreateWindowExW(0, L"BUTTON", text,
                                   WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_PUSHBUTTON,
                                   x, y, w, 28, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                                   ::GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(b, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
        return b;
    };
    const HINSTANCE hinst = ::GetModuleHandleW(nullptr);

    // Row 1: model artifact (full-width edit + file-picker button)
    label(L"Model (.ninfer):", 8, 12);
    edit(IDC_MODEL_EDIT, L"", 132, 10, 444);
    button(IDC_MODEL_BROWSE, L"Browse...", 584, 9, 90);
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_MODEL_EDIT),
                     L"qwen3_8_27b_nvfp4 23,7gb.ninfer");

    // Row 2: context sizing. The Auto button fills the max-context field
    // with the probed VRAM fit ceiling (beside the edit: 232..292, clear of
    // the "Max new:" label at x=380).
    label(L"Max context:", 8, 42);
    edit(IDC_MAX_CONTEXT_EDIT, L"", 132, 40, 90);
    button(IDC_AUTO_CONTEXT_BTN, L"Auto", 232, 40, 60);
    label(L"Max new:", 380, 42);
    edit(IDC_MAX_NEW_EDIT, L"", 510, 40, 80);
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_MAX_CONTEXT_EDIT), L"200000");
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_MAX_NEW_EDIT), L"8196");

    // Row 3: KV cache
    label(L"KV capacity:", 8, 72);
    edit(IDC_KV_CAPACITY_EDIT, L"", 132, 70, 90);
    label(L"KV dtype:", 380, 72);
    static const wchar_t* const kKvDtypes[] = {L"bf16", L"int8", L"fp8", L"nvfp4", L"k8v4"};
    combo(IDC_KV_DTYPE_COMBO, 510, 70, 100, kKvDtypes, 5);
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_KV_CAPACITY_EDIT), L"200000");
    ::SendMessageW(::GetDlgItem(hwnd, IDC_KV_DTYPE_COMBO), CB_SETCURSEL, 2, 0);  // fp8

    // Row 4: VRAM headroom + request log (yes/no; the file name is fixed to
    // requests.jsonl next to the exe when on)
    label(L"VRAM headroom (MiB):", 8, 102);
    edit(IDC_HEADROOM_EDIT, L"", 132, 100, 80);  // left empty by default
    check(IDC_REQUEST_LOG_EDIT, L"Request log (requests.jsonl)", 380, 102, 220, true);

    // Row 5: endpoint
    label(L"Host:", 8, 132);
    edit(IDC_HOST_EDIT, L"", 132, 130, 120);
    label(L"Port:", 380, 132);
    edit(IDC_PORT_EDIT, L"", 510, 130, 80);
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_HOST_EDIT), L"127.0.0.1");
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_PORT_EDIT), L"8888");

    // Row 6: sampling
    label(L"Temperature:", 8, 162);
    edit(IDC_TEMPERATURE_EDIT, L"", 132, 160, 70);
    label(L"Top-p:", 380, 162);
    edit(IDC_TOPP_EDIT, L"", 510, 160, 70);
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_TEMPERATURE_EDIT), L"1.0");
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_TOPP_EDIT), L"0.95");

    // Row 7: sampling (continued)
    label(L"Top-k:", 8, 192);
    edit(IDC_TOPK_EDIT, L"", 132, 190, 70);
    label(L"Min-p:", 380, 192);
    edit(IDC_MINP_EDIT, L"", 510, 190, 70);
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_TOPK_EDIT), L"20");
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_MINP_EDIT), L"0.0");

    // Row 8: penalties
    label(L"Presence:", 8, 222);
    edit(IDC_PRESENCE_EDIT, L"", 132, 220, 70);
    label(L"Frequency:", 380, 222);
    edit(IDC_FREQUENCY_EDIT, L"", 510, 220, 70);
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_PRESENCE_EDIT), L"0.0");
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_FREQUENCY_EDIT), L"0.0");

    // Sampling presets: the right margin (x=600) of the sampling rows is
    // free (the right-column edits end at x=580), so the pair stacks beside
    // the fields they set.
    button(IDC_PRESET_THINKING, L"Thinking", 600, 160, 80);
    button(IDC_PRESET_INSTRUCT, L"Instruct", 600, 194, 80);

    // Row 9: behavior toggles (the check text is its own label)
    check(IDC_GREEDY_CHECK, L"Greedy", 132, 250, 120, false);
    check(IDC_THINKING_CHECK, L"Thinking", 270, 250, 120, true);
    check(IDC_PRESERVE_THINKING_CHECK, L"Preserve thinking", 510, 250, 180, true);

    // Row 10: thinking budget + draft tokens
    label(L"Think budget:", 8, 282);
    edit(IDC_DEFAULT_THINK_BUDGET_EDIT, L"", 132, 280, 80);
    label(L"Draft tokens:", 380, 282);
    edit(IDC_DRAFT_TOKENS_EDIT, L"", 510, 280, 70);
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_DEFAULT_THINK_BUDGET_EDIT), L"2048");
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_DRAFT_TOKENS_EDIT), L"7");

    // Row 11: vision + speculative decoding
    label(L"Vision:", 8, 312);
    static const wchar_t* const kVision[] = {L"Off", L"On (GPU)", L"Offload"};
    combo(IDC_VISION_COMBO, 132, 310, 120, kVision, 3);
    ::SendMessageW(::GetDlgItem(hwnd, IDC_VISION_COMBO), CB_SETCURSEL, 2, 0);  // default Offload
    label(L"Spec:", 380, 312);
    static const wchar_t* const kSpecs[] = {L"off", L"mtp", L"dflash", L"dflash2"};
    combo(IDC_SPEC_COMBO, 510, 310, 120, kSpecs, 4);
    ::SendMessageW(::GetDlgItem(hwnd, IDC_SPEC_COMBO), CB_SETCURSEL, 3, 0);  // dflash2

    // Row 12: draft head + seed
    check(IDC_LM_HEAD_DRAFT_CHECK, L"LM head draft", 132, 340, 160, true);
    label(L"Seed:", 380, 342);
    edit(IDC_SEED_EDIT, L"", 510, 340, 100);  // left empty by default (engine seed)

    // Row 13: actions
    button(IDC_PROBE_BUTTON, L"Probe VRAM", 132, 372, 110);
    button(IDC_LAUNCH_BUTTON, L"Launch", 252, 372, 90);
    button(IDC_STOP_BUTTON, L"Stop", 352, 372, 80);
    ::EnableWindow(::GetDlgItem(hwnd, IDC_STOP_BUTTON), FALSE);
}

// Extended section (below the Core block + usage): serve tuning flags. Two
// columns (label x=8 / x=380, controls x=132 / x=510), rows every 30 px
// starting at y=496. The think-budget-message edit is full-width so its
// long default is visible. Defaults come from the canonical config; empty
// edits and unchecked boxes are omitted from the serve argv.
void create_extended_controls(HWND hwnd) {
    const HFONT font = static_cast<HFONT>(::GetStockObject(DEFAULT_GUI_FONT));

    auto label = [&](const wchar_t* text, int x, int y) {
        HWND h = ::CreateWindowExW(0, L"STATIC", text, WS_CHILD | WS_VISIBLE | SS_LEFT,
                                   x, y, 120, 18, hwnd, nullptr,
                                   ::GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(h, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
    };
    auto edit = [&](int id, const wchar_t* text, int x, int y, int w) {
        HWND e = ::CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", text,
                                   WS_CHILD | WS_VISIBLE | WS_TABSTOP | ES_AUTOHSCROLL,
                                   x, y, w, 22, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                                   ::GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(e, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
        return e;
    };
    auto combo = [&](int id, int x, int y, int w, const wchar_t* const* items, int count) {
        HWND c = ::CreateWindowExW(0, L"COMBOBOX", L"",
                                   WS_CHILD | WS_VISIBLE | WS_TABSTOP | CBS_DROPDOWNLIST,
                                   x, y, w, 200, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                                   ::GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(c, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
        for (int i = 0; i < count; ++i) {
            ::SendMessageW(c, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(items[i]));
        }
        ::SendMessageW(c, CB_SETCURSEL, 0, 0);
        return c;
    };
    auto check = [&](int id, const wchar_t* text, int x, int y, int w, bool checked) {
        HWND c = ::CreateWindowExW(0, L"BUTTON", text,
                                   WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_AUTOCHECKBOX,
                                   x, y, w, 20, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                                   ::GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(c, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
        if (checked) { ::SendMessageW(c, BM_SETCHECK, BST_CHECKED, 0); }
        return c;
    };

    // Section header.
    label(L"Extended:", 8, 466);

    // E-row 1: model id + prefill chunk
    label(L"Model ID:", 8, 498);
    edit(IDC_MODEL_ID_EDIT, L"", 132, 496, 120);
    label(L"Prefill chunk:", 380, 498);
    edit(IDC_PREFILL_CHUNK_EDIT, L"", 510, 496, 80);
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_PREFILL_CHUNK_EDIT), L"4096");

    // E-row 2: max concurrency + api key
    label(L"Max concurrency:", 8, 528);
    edit(IDC_MAX_CONCURRENCY_EDIT, L"", 132, 526, 80);
    label(L"API key:", 380, 528);
    edit(IDC_API_KEY_EDIT, L"", 510, 526, 180);
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_MAX_CONCURRENCY_EDIT), L"2");

    // E-row 3: think budget message (full-width for the long default)
    label(L"Think budget msg:", 8, 558);
    edit(IDC_THINK_BUDGET_MSG_EDIT, L"", 132, 556, 558);
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_THINK_BUDGET_MSG_EDIT),
                     L"Considering the limited time available to the user, I must stop thinking now. Time to act:");

    // E-row 4: think budget policy + log level
    label(L"Think policy:", 8, 588);
    static const wchar_t* const kThinkPolicies[] = {L"strict", L"clamp", L"ignore"};
    combo(IDC_THINK_BUDGET_POLICY_COMBO, 132, 586, 100, kThinkPolicies, 3);
    ::SendMessageW(::GetDlgItem(hwnd, IDC_THINK_BUDGET_POLICY_COMBO), CB_SETCURSEL, 1, 0);  // clamp
    label(L"Log level:", 380, 588);
    static const wchar_t* const kLogLevels[] = {L"error", L"warn", L"info", L"debug"};
    combo(IDC_LOG_LEVEL_COMBO, 510, 586, 100, kLogLevels, 4);
    ::SendMessageW(::GetDlgItem(hwnd, IDC_LOG_LEVEL_COMBO), CB_SETCURSEL, 2, 0);  // info

    // E-row 5: max thinking budget + cors
    label(L"Max think budget:", 8, 618);
    edit(IDC_MAX_THINK_BUDGET_EDIT, L"", 132, 616, 80);
    check(IDC_CORS_CHECK, L"CORS", 510, 616, 120, false);
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_MAX_THINK_BUDGET_EDIT), L"4096");

    // E-row 6: media cache + no cuda graph
    label(L"Media cache (MiB):", 8, 648);
    edit(IDC_MEDIA_CACHE_EDIT, L"", 132, 646, 80);  // empty (flag omitted)
    check(IDC_NO_CUDA_GRAPH_CHECK, L"No CUDA graph", 510, 646, 150, false);

    // E-row 7: media live + usage chunk choice
    label(L"Media live (MiB):", 8, 678);
    edit(IDC_MEDIA_LIVE_EDIT, L"", 132, 676, 80);  // empty (flag omitted)
    check(IDC_USAGE_CHUNK_CHOICE_CHECK, L"Usage chunk choice", 510, 676, 160, false);

    // E-row 8: media preprocess threads
    label(L"Media preproc thr:", 8, 708);
    edit(IDC_MEDIA_PREPROC_THREADS_EDIT, L"", 132, 706, 80);  // empty (flag omitted)
}

// The Win32 SDK headers do not define SS_GROUPBOX (it is an MFC constant);
// 0x0003 is the documented group-box static style.
constexpr int kSSGroupBox      = 0x0003;

// Advanced (collapsible) section geometry. The group box sits below the
// Extended section; its 27 controls fill 14 two-column rows of 30 px.
constexpr int kAdvGroupX       = 8;
constexpr int kAdvGroupY       = 768;
constexpr int kAdvGroupW       = 744;
constexpr int kAdvGroupH       = 448;   // group bottom at 768 + 448 = 1216
constexpr int kAdvRow0Y        = 788;   // first row's control y
constexpr int kAdvRowStep      = 30;
constexpr int kCompactWinH     = 940;   // the height CreateWindowExW sets
constexpr int kExpandedClientH = kAdvGroupY + kAdvGroupH + 40;  // group + status strip

// Show/hide the Advanced group box, all 27 controls, and all 24 labels, and
// grow/shrink the window so the group and the auto-anchored status line stay
// fully visible. WM_SIZE re-anchors the status line on the resize. The
// "Show advanced" checkbox itself always stays visible.
void set_advanced_visible(HWND hwnd, bool show) {
    const int cmd = show ? SW_SHOWNOACTIVATE : SW_HIDE;
    HWND group = ::GetDlgItem(hwnd, IDC_ADV_GROUP);
    if (group != nullptr) { ::ShowWindow(group, cmd); }
    auto toggle = [&](int from, int to) {
        for (int id = from; id <= to; ++id) {
            HWND c = ::GetDlgItem(hwnd, id);
            if (c != nullptr) { ::ShowWindow(c, cmd); }
        }
    };
    toggle(IDC_PREFIX_CACHE_FILE_EDIT, IDC_TOLERANT_TOOL_CALLS_CHECK);  // controls
    toggle(IDC_ADV_LABEL_BASE, IDC_ADV_LABEL_BASE + 23);                // labels
    RECT wr = {};
    RECT cr = {};
    if (!::GetWindowRect(hwnd, &wr) || !::GetClientRect(hwnd, &cr)) { return; }
    const int nonclient = (wr.bottom - wr.top) - cr.bottom;
    const int new_h = show ? (kExpandedClientH + nonclient) : kCompactWinH;
    ::MoveWindow(hwnd, wr.left, wr.top, wr.right - wr.left, new_h, TRUE);
}

// Advanced section: one collapsible "Advanced" group box below the Extended
// block holding 27 tuning controls (same two-column scheme as Core/Extended:
// labels x=8 / x=380, controls x=132 / x=510, 30 px rows; 14 rows). Hidden
// by default; the "Show advanced" toggle shows it and grows the window.
// Canonical defaults are set here; the serve argv is built from these values
// whether or not the group is visible.
void create_advanced_controls(HWND hwnd) {
    const HFONT font = static_cast<HFONT>(::GetStockObject(DEFAULT_GUI_FONT));

    // Every label created below is an Advanced label: hand out dialog IDs
    // (IDC_ADV_LABEL_BASE..+23) so the collapse toggle can hide them.
    int label_id = IDC_ADV_LABEL_BASE;
    auto label = [&](const wchar_t* text, int x, int y) {
        HWND h = ::CreateWindowExW(0, L"STATIC", text, WS_CHILD | WS_VISIBLE | SS_LEFT,
                                   x, y, 120, 18, hwnd,
                                   reinterpret_cast<HMENU>(static_cast<INT_PTR>(label_id++)),
                                   ::GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(h, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
    };
    auto edit = [&](int id, const wchar_t* text, int x, int y, int w) {
        HWND e = ::CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", text,
                                   WS_CHILD | WS_VISIBLE | WS_TABSTOP | ES_AUTOHSCROLL,
                                   x, y, w, 22, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                                   ::GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(e, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
        return e;
    };
    auto check = [&](int id, const wchar_t* text, int x, int y, int w, bool checked) {
        HWND c = ::CreateWindowExW(0, L"BUTTON", text,
                                   WS_CHILD | WS_VISIBLE | WS_TABSTOP | BS_AUTOCHECKBOX,
                                   x, y, w, 20, hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                                   ::GetModuleHandleW(nullptr), nullptr);
        ::SendMessageW(c, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
        if (checked) { ::SendMessageW(c, BM_SETCHECK, BST_CHECKED, 0); }
        return c;
    };

    // Compact toggle just above the group (the Extended block ends at 728).
    check(IDC_ADV_TOGGLE_CHECK, L"Show advanced", 8, 740, 150, false);

    HWND group = ::CreateWindowExW(0, L"STATIC", L"Advanced",
        WS_CHILD | WS_VISIBLE | WS_GROUP | kSSGroupBox | WS_TABSTOP,
        kAdvGroupX, kAdvGroupY, kAdvGroupW, kAdvGroupH, hwnd,
        reinterpret_cast<HMENU>(static_cast<INT_PTR>(IDC_ADV_GROUP)),
        ::GetModuleHandleW(nullptr), nullptr);
    ::SendMessageW(group, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);

    const int y0 = kAdvRow0Y;
    const int dy = kAdvRowStep;

    // A-row 1: prefix cache file (full-width: holds a long absolute path).
    // Default resolves to <deploy-dir>\prefix-cache.bin at runtime: the deploy
    // folder name carries non-ASCII bytes the source cannot spell, and module
    // dir() carries its real bytes.
    label(L"Prefix cache file:", 8, y0 + 2);
    edit(IDC_PREFIX_CACHE_FILE_EDIT, L"", 132, y0, 558);
    const std::wstring pcache_dir = module_dir();
    if (!pcache_dir.empty()) {
        ::SetWindowTextW(::GetDlgItem(hwnd, IDC_PREFIX_CACHE_FILE_EDIT),
                         (pcache_dir + L"prefix-cache.bin").c_str());
    }

    // A-row 2: original prefix caching + shared prefixes
    check(IDC_USE_ORIG_PREFIX_CHECK, L"Use orig prefix caching", 132, y0 + dy, 220, false);
    label(L"Shared prefixes:", 380, y0 + dy + 2);
    edit(IDC_MAX_SHARED_PREFIXES_EDIT, L"", 510, y0 + dy, 80);  // empty (flag omitted)

    // A-row 3: private continuations + anchor spacing
    label(L"Priv. continuations:", 8, y0 + 2*dy + 2);
    edit(IDC_MAX_PRIVATE_CONT_EDIT, L"", 132, y0 + 2*dy, 80);  // empty (flag omitted)
    label(L"Anchor spacing:", 380, y0 + 2*dy + 2);
    edit(IDC_LONG_ANCHOR_SPACING_EDIT, L"", 510, y0 + 2*dy, 80);  // empty (flag omitted)

    // A-row 4: long anchors + host cache
    label(L"Max long anchors:", 8, y0 + 3*dy + 2);
    edit(IDC_MAX_LONG_ANCHORS_EDIT, L"", 132, y0 + 3*dy, 80);  // empty (flag omitted)
    label(L"Host cache (MiB):", 380, y0 + 3*dy + 2);
    edit(IDC_HOST_CACHE_MIB_EDIT, L"", 510, y0 + 3*dy, 80);
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_HOST_CACHE_MIB_EDIT), L"32000");

    // A-row 5: host KV + host state slots
    label(L"Host KV (MiB):", 8, y0 + 4*dy + 2);
    edit(IDC_HOST_KV_MIB_EDIT, L"", 132, y0 + 4*dy, 80);  // empty (flag omitted)
    label(L"Host state slots:", 380, y0 + 4*dy + 2);
    edit(IDC_HOST_STATE_SLOTS_EDIT, L"", 510, y0 + 4*dy, 80);  // empty (flag omitted)

    // A-row 6: device slots
    label(L"Dev snap slots:", 8, y0 + 5*dy + 2);
    edit(IDC_DEV_SNAP_SLOTS_EDIT, L"", 132, y0 + 5*dy, 80);  // empty (flag omitted)
    label(L"Dev state slots:", 380, y0 + 5*dy + 2);
    edit(IDC_DEV_STATE_SLOTS_EDIT, L"", 510, y0 + 5*dy, 80);  // empty (flag omitted)

    // A-row 7: ngram draft + min match
    label(L"Ngram draft:", 8, y0 + 6*dy + 2);
    edit(IDC_NGRAM_DRAFT_EDIT, L"", 132, y0 + 6*dy, 80);
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_NGRAM_DRAFT_EDIT), L"15");
    label(L"Ngram min match:", 380, y0 + 6*dy + 2);
    edit(IDC_NGRAM_MIN_MATCH_EDIT, L"", 510, y0 + 6*dy, 80);
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_NGRAM_MIN_MATCH_EDIT), L"8");

    // A-row 8: ngram native sessions + ngram archive
    check(IDC_NGRAM_NATIVE_CHECK, L"Ngram native sessions", 132, y0 + 7*dy, 200, false);
    label(L"Ngram archive MiB:", 380, y0 + 7*dy + 2);
    edit(IDC_NGRAM_ARCHIVE_MIB_EDIT, L"", 510, y0 + 7*dy, 80);  // empty (flag omitted)

    // A-row 9: ngram session + cache tap ladder
    label(L"Ngram session MiB:", 8, y0 + 8*dy + 2);
    edit(IDC_NGRAM_SESSION_MIB_EDIT, L"", 132, y0 + 8*dy, 80);  // empty (flag omitted)
    label(L"Cache tap ladder:", 380, y0 + 8*dy + 2);
    edit(IDC_CACHE_TAP_LADDER_EDIT, L"", 510, y0 + 8*dy, 80);  // empty (flag omitted)

    // A-row 10: cache tap gap + taps per request
    label(L"Cache tap min gap:", 8, y0 + 9*dy + 2);
    edit(IDC_CACHE_TAP_MIN_GAP_EDIT, L"", 132, y0 + 9*dy, 80);  // empty (flag omitted)
    label(L"Cache taps/req:", 380, y0 + 9*dy + 2);
    edit(IDC_CACHE_TAPS_PER_REQ_EDIT, L"", 510, y0 + 9*dy, 80);  // empty (flag omitted)

    // A-row 11: response store limits
    label(L"Resp store rec max:", 8, y0 + 10*dy + 2);
    edit(IDC_RESP_STORE_MAX_RECORDS_EDIT, L"", 132, y0 + 10*dy, 80);  // empty (flag omitted)
    label(L"Resp store MiB max:", 380, y0 + 10*dy + 2);
    edit(IDC_RESP_STORE_MAX_MIB_EDIT, L"", 510, y0 + 10*dy, 80);  // empty (flag omitted)

    // A-row 12: request limits
    label(L"Max request (MiB):", 8, y0 + 11*dy + 2);
    edit(IDC_MAX_REQUEST_MIB_EDIT, L"", 132, y0 + 11*dy, 80);  // empty (flag omitted)
    label(L"Max pending reqs:", 380, y0 + 11*dy + 2);
    edit(IDC_MAX_PENDING_REQ_EDIT, L"", 510, y0 + 11*dy, 80);  // empty (flag omitted)

    // A-row 13: pending timeout + chat template
    label(L"Pending timeout ms:", 8, y0 + 12*dy + 2);
    edit(IDC_PENDING_TIMEOUT_MS_EDIT, L"", 132, y0 + 12*dy, 80);  // empty (flag omitted)
    label(L"Chat template:", 380, y0 + 12*dy + 2);
    edit(IDC_CHAT_TEMPLATE_EDIT, L"", 510, y0 + 12*dy, 100);  // empty (flag omitted)

    // A-row 14: context cost presets + tolerant tool calls
    label(L"Ctx cost presets:", 8, y0 + 13*dy + 2);
    edit(IDC_CONTEXT_COST_PRESETS_EDIT, L"", 132, y0 + 13*dy, 80);  // empty (flag omitted)
    check(IDC_TOLERANT_TOOL_CALLS_CHECK, L"Tolerant tool calls", 510, y0 + 13*dy, 180, false);

    // Collapsed by default (no-op window resize: already the compact height).
    set_advanced_visible(hwnd, false);
}

// ---------------------------------------------------------------------------
// Tooltips: one TOOLTIPS_CLASS3 child on the main window, one TTM_ADDTOOL per
// control. Each tool is keyed by its child HWND (TTF_SUBCLASS sublasses the
// child so the tooltip manager sees its WM_MOUSEMOVE), so hidden controls can
// still carry a tooltip. The tooltip window itself (ID 1600) is never given
// one.
// ---------------------------------------------------------------------------
struct ToolTipRow {
    int id;
    const wchar_t* text;
};

void create_tooltips(HWND hMain) {
    static const ToolTipRow kRows[] = {
        { IDC_MODEL_EDIT,                L"the `.ninfer` model to load (filename next to the launcher)." },
        { IDC_HOST_EDIT,                 L"address the server listens on (default `127.0.0.1`, local only)." },
        { IDC_PORT_EDIT,                 L"TCP port the server listens on (default `8888`)." },
        { IDC_MAX_NEW_EDIT,              L"maximum output tokens per request." },
        { IDC_MAX_CONTEXT_EDIT,          L"total context length in tokens (prompt + output)." },
        { IDC_KV_CAPACITY_EDIT,          L"KV-cache token budget, or `auto` to size it to VRAM." },
        { IDC_KV_DTYPE_COMBO,            L"precision of the KV cache (lower precision = more fits)." },
        { IDC_TEMPERATURE_EDIT,          L"sampling temperature (`0` = greedy)." },
        { IDC_TOPP_EDIT,                 L"nucleus sampling threshold." },
        { IDC_TOPK_EDIT,                 L"keep only the top-k candidates." },
        { IDC_MINP_EDIT,                 L"drop tokens whose probability is below a fraction of the top one." },
        { IDC_PRESENCE_EDIT,             L"bias against repeating tokens already present." },
        { IDC_FREQUENCY_EDIT,            L"bias against repeating tokens in proportion to how often they appear." },
        { IDC_DEFAULT_THINK_BUDGET_EDIT, L"tokens the model may spend reasoning per request." },
        { IDC_REQUEST_LOG_EDIT,          L"log each request to `requests.jsonl` (also powers the usage readout below)." },
        { IDC_DRAFT_TOKENS_EDIT,         L"speculative-decoding tokens per step (higher = faster, needs more VRAM)." },
        { IDC_LM_HEAD_DRAFT_CHECK,       L"use the main LM head for drafting (better drafts, slight cost)." },
        { IDC_THINKING_CHECK,            L"enable the model's reasoning mode." },
        { IDC_PRESERVE_THINKING_CHECK,   L"keep reasoning tokens in the output." },
        { IDC_VISION_COMBO,              L"how images are processed: Off (no vision), On-GPU, or Offload." },
        { IDC_SPEC_COMBO,                L"speculative-decoding method (e.g. `dflash2`)." },
        { IDC_GREEDY_CHECK,              L"force deterministic greedy decoding (overrides temperature)." },
        { IDC_SEED_EDIT,                 L"RNG seed for reproducible sampling (empty = random)." },
        { IDC_HEADROOM_EDIT,             L"safety reserve the auto/probe path keeps free (only used when KV capacity is `auto`)." },
        { IDC_LAUNCH_BUTTON,             L"start `ninfer-serve.exe` with these settings." },
        { IDC_STOP_BUTTON,               L"stop the running server." },
        { IDC_PROBE_BUTTON,              L"check how much context fits (does not change any setting)." },
        { IDC_AUTO_CONTEXT_BTN,          L"probe and fill Max context (and raise KV capacity if it is below the fit)." },
        { IDC_MAX_CONCURRENCY_EDIT,      L"how many requests to serve at once." },
        { IDC_PREFILL_CHUNK_EDIT,        L"tokens of prompt processed per prefill step." },
        { IDC_MAX_THINK_BUDGET_EDIT,     L"hard ceiling on reasoning tokens per request." },
        { IDC_THINK_BUDGET_MSG_EDIT,     L"text appended when the thinking budget is reached." },
        { IDC_THINK_BUDGET_POLICY_COMBO, L"how an over-budget reasoning request is handled: `strict` / `clamp` / `ignore`." },
        { IDC_HOST_CACHE_MIB_EDIT,       L"host (CPU) memory for the KV host tier (offload)." },
        { IDC_PREFIX_CACHE_FILE_EDIT,    L"file to persist the host-tier prefix cache." },
        { IDC_NGRAM_DRAFT_EDIT,          L"n-gram speculative tokens per step (needs concurrency 1 above 15)." },
        { IDC_NGRAM_MIN_MATCH_EDIT,      L"minimum matching prefix length for n-gram drafting." },
    };

    HWND hTip = ::CreateWindowExW(WS_EX_TOOLWINDOW, TOOLTIPS_CLASSW, L"",
                                   WS_CHILD | WS_POPUP, 0, 0, 0, 0, hMain,
                                   (HMENU)1600, ::GetModuleHandleW(nullptr), nullptr);
    if (hTip == nullptr) { return; }
    for (const ToolTipRow& row : kRows) {
        const HWND hCtl = ::GetDlgItem(hMain, row.id);
        if (hCtl == nullptr) { continue; }
        TOOLINFO ti {};
        ti.cbSize   = sizeof(ti);
        ti.uFlags   = TTF_SUBCLASS;
        ti.hwnd     = hCtl;
        ti.uId      = static_cast<UINT_PTR>(row.id);
        ti.lpszText = const_cast<LPWSTR>(row.text);  // control copies the string; never writes through it
        ::SendMessageW(hTip, TTM_ADDTOOL, 0, reinterpret_cast<LPARAM>(&ti));
    }
}

void create_scaffold(HWND hwnd) {
    create_core_controls(hwnd);
    create_extended_controls(hwnd);
    create_advanced_controls(hwnd);
    load_settings(hwnd);  // override the defaults above with any saved values
    // Re-sync the Advanced visibility after load_settings: a persisted
    // "Show advanced"=1 must actually show the group and grow the window on
    // startup (create_advanced_controls always starts collapsed).
    set_advanced_visible(
        hwnd,
        ::SendMessageW(::GetDlgItem(hwnd, IDC_ADV_TOGGLE_CHECK), BM_GETCHECK, 0, 0)
            == BST_CHECKED);
    create_status(hwnd);
    create_usage_block(hwnd);
    create_tooltips(hwnd);  // after every control (core/extended/advanced) exists
}

// ---------------------------------------------------------------------------
// Serve lifecycle: build the argv, launch ninfer-serve.exe in its own
// console, poll /health until it reports up, and Stop / exit terminate the
// child so no orphaned GPU work lingers.
// ---------------------------------------------------------------------------

// Launch the serve: CreateProcessW in its own console (CREATE_NEW_CONSOLE) so
// the user watches "listening on http://host:port" and the live tok/s stats
// directly -- unsloth-style. Disable Launch / enable Stop, start the ~1 s
// /health poll, and reap the child on a watcher thread that posts
// WM_APP_DONE.
void launch_serve(HWND hwnd, const std::vector<std::wstring>& argv,
                  std::wstring_view extra_status = {}) {
    // Reap a previous run's watcher thread: it stays joinable after posting
    // WM_APP_DONE, and assigning a new joinable thread to g_child.watcher would
    // std::terminate.
    if (g_child.watcher.joinable()) { g_child.watcher.join(); }

    STARTUPINFOW si {};
    si.cb = sizeof(si);

    // CreateProcessW takes an LPWSTR and may modify the buffer; work on a copy.
    std::wstring mutable_command = build_command_line(argv);
    // Run the serve from the exe dir so a bare --request-log-jsonl name lands
    // where the usage tracker reads it (module_dir() + "requests.jsonl").
    // CreateProcessW may modify lpCurrentDirectory, so work on a copy.
    std::wstring current_dir = module_dir();
    PROCESS_INFORMATION pi {};
    if (!::CreateProcessW(argv[0].c_str(), mutable_command.data(), nullptr, nullptr, TRUE,
                          CREATE_NEW_CONSOLE,
                          current_dir.empty() ? nullptr : current_dir.data(),
                          nullptr, &si, &pi)) {
        const DWORD err = ::GetLastError();
        set_status(hwnd, L"Error: failed to start ninfer-serve.exe (Win32 error " +
                            std::to_wstring(err) + L")");
        return;
    }
    ::CloseHandle(pi.hThread);

    g_child.process   = pi.hProcess;
    g_child.running.store(true);

    ::EnableWindow(::GetDlgItem(hwnd, IDC_LAUNCH_BUTTON), FALSE);
    ::EnableWindow(::GetDlgItem(hwnd, IDC_STOP_BUTTON), TRUE);
    std::wstring status = L"Serving \u2014 waiting for /health\u2026";
    if (!extra_status.empty()) { status += L' '; status += extra_status; }
    set_status(hwnd, status);
    ::SetTimer(hwnd, kHealthTimerId, 1000, nullptr);

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

// Validate the form, guard against a serve already bound to the target port,
// then build and launch the serve command line.
void serve_child(HWND hwnd) {
    const std::wstring model = get_control_text(::GetDlgItem(hwnd, IDC_MODEL_EDIT));
    const std::wstring host  = get_control_text(::GetDlgItem(hwnd, IDC_HOST_EDIT));
    const std::wstring port  = get_control_text(::GetDlgItem(hwnd, IDC_PORT_EDIT));

    if (model.empty()) {
        set_status(hwnd, L"Error: choose a model artifact (.ninfer)");
        return;
    }
    if (port.empty()) { set_status(hwnd, L"Error: enter a port number"); return; }
    for (wchar_t ch : port) {
        if (ch < L'0' || ch > L'9') {
            set_status(hwnd, L"Error: port must be numeric");
            return;
        }
    }

    if (find_sibling(L"ninfer-serve.exe").empty()) {
        set_status(hwnd, L"Error: ninfer-serve.exe not found next to ninfer-gui.exe");
        return;
    }

    // A serve already answering on the target port would make the new child die
    // on bind with no useful status; detect it up front via /health (the same
    // helper the health poll uses).
    if (!http_get(host, port, "/health").empty()) {
        set_status(hwnd, L"A serve is already running on " +
                            (host.empty() ? std::wstring(L"127.0.0.1") : host) + L":" + port +
                            L"; Stop it first or pick another port.");
        return;
    }

    g_serve_host = host.empty() ? std::wstring(L"127.0.0.1") : host;
    g_serve_port = port;

    // The probe never runs here (it would load the ~32 GiB of weights before
    // serving). The manual values launch as-is, with one hard invariant:
    // a sequence context cannot exceed the KV pool, so clamp the max-context
    // edit down to a numeric kv-capacity when it exceeds it.
    std::wstring extra_status;
    const std::optional<std::uint64_t> max_ctx = parse_uint_field(hwnd, IDC_MAX_CONTEXT_EDIT);
    const std::optional<std::uint64_t> kv_cap  = parse_uint_field(hwnd, IDC_KV_CAPACITY_EDIT);
    if (max_ctx && kv_cap && *max_ctx > *kv_cap) {
        ::SetWindowTextW(::GetDlgItem(hwnd, IDC_MAX_CONTEXT_EDIT), std::to_wstring(*kv_cap).c_str());
        extra_status += L"max-context clamped to kv-capacity " + std::to_wstring(*kv_cap) + L"; ";
    }
    // Read-only probe-cache check (never a re-probe): warn when the manual
    // context exceeds the last probed ceiling; the launch still honors the
    // manual value -- the user takes responsibility for it.
    if (build_probe_key(hwnd, model) == g_probe_key && g_probe_fit > 0 &&
        max_ctx && *max_ctx > g_probe_fit) {
        extra_status = L"manual max-context " + std::to_wstring(*max_ctx) +
                       L" > probed fit " + std::to_wstring(g_probe_fit) + L" \u2014 launching anyway; " +
                       extra_status;
    }

    const std::vector<std::wstring> argv = build_serve_argv(hwnd, model);
    launch_serve(hwnd, argv, extra_status);
}

// Stop the serve this GUI launched (its terminal window dies); the GUI stays
// open. The watcher posts WM_APP_DONE, which re-enables Launch and reports
// the exit code.
void stop_serve(HWND hwnd) {
    if (g_child.running.load() && g_child.process != nullptr) {
        set_status(hwnd, L"Stopping serve\u2026");
        ::EnableWindow(::GetDlgItem(hwnd, IDC_STOP_BUTTON), FALSE);
        ::TerminateProcess(g_child.process, 1);
    } else {
        set_status(hwnd, L"No serve started from this GUI.");
    }
}

// WinHttp GET: short timeout, local HTTP only. Any failure collapses to an
// empty body, so a closed port never hangs the UI.
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

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_COMMAND:
        if (LOWORD(wParam) == IDCANCEL) { ::PostQuitMessage(0); }
        if (HIWORD(wParam) == EN_KILLFOCUS) { save_settings(hwnd); }
        if (LOWORD(wParam) == IDC_LAUNCH_BUTTON && HIWORD(wParam) == BN_CLICKED &&
            !g_child.running.load()) {
            serve_child(hwnd);
            return 0;
        }
        if (LOWORD(wParam) == IDC_PROBE_BUTTON && HIWORD(wParam) == BN_CLICKED) {
            const std::wstring model = get_control_text(::GetDlgItem(hwnd, IDC_MODEL_EDIT));
            if (model.empty()) {
                set_status(hwnd, L"Error: choose a model artifact (.ninfer)");
                return 0;
            }
            // Informational only: show the VRAM ceiling; never change a
            // control. On a cache hit re-assert the ceiling (run_probe only
            // sets the status on a fresh probe); on failure run_probe has
            // already reported the reason.
            const std::uint32_t fit = ensure_probe(hwnd, model);
            if (fit > 0) {
                set_status(hwnd, L"~" + std::to_wstring(fit) + L" tokens fit (free after weights: " +
                               format_gib(g_probe_free_after_bytes) + L")");
            }
            return 0;
        }
        if (LOWORD(wParam) == IDC_AUTO_CONTEXT_BTN && HIWORD(wParam) == BN_CLICKED) {
            const std::wstring model = get_control_text(::GetDlgItem(hwnd, IDC_MODEL_EDIT));
            if (model.empty()) {
                set_status(hwnd, L"Error: choose a model artifact (.ninfer)");
                return 0;
            }
            // Fill the max-context field with the probed ceiling. On failure
            // (fit == 0) run_probe has already reported the reason, so leave
            // both fields untouched.
            const std::uint32_t fit = ensure_probe(hwnd, model);
            if (fit > 0) {
                ::SetWindowTextW(::GetDlgItem(hwnd, IDC_MAX_CONTEXT_EDIT),
                                 std::to_wstring(fit).c_str());
                // Keep max-context <= kv-capacity: a fixed pool below the
                // ceiling is bumped up to the fit.
                const std::optional<std::uint64_t> kv_cap = parse_uint_field(hwnd, IDC_KV_CAPACITY_EDIT);
                if (kv_cap && *kv_cap < fit) {
                    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_KV_CAPACITY_EDIT),
                                     std::to_wstring(fit).c_str());
                }
            }
            return 0;
        }
        if (LOWORD(wParam) == IDC_STOP_BUTTON && HIWORD(wParam) == BN_CLICKED) {
            stop_serve(hwnd);
            return 0;
        }
        if (LOWORD(wParam) == IDC_PRESET_THINKING && HIWORD(wParam) == BN_CLICKED) {
            apply_preset(hwnd, true);
            return 0;
        }
        if (LOWORD(wParam) == IDC_PRESET_INSTRUCT && HIWORD(wParam) == BN_CLICKED) {
            apply_preset(hwnd, false);
            return 0;
        }
        if (LOWORD(wParam) == IDC_ADV_TOGGLE_CHECK && HIWORD(wParam) == BN_CLICKED) {
            set_advanced_visible(
                hwnd,
                ::SendMessageW(::GetDlgItem(hwnd, IDC_ADV_TOGGLE_CHECK), BM_GETCHECK, 0, 0)
                    == BST_CHECKED);
            return 0;
        }
        return 0;
    case WM_TIMER:
        if (wParam == kHealthTimerId) {
            // Poll /health until it answers: model + KV-pool load takes
            // several seconds, so keep ticking while the body is empty.
            // Only stop when the serve reports up; a dead child ends the
            // poll via WM_APP_DONE (and WM_CLOSE kills the timer too).
            if (!http_get(g_serve_host, g_serve_port, "/health").empty()) {
                ::KillTimer(hwnd, kHealthTimerId);
                set_status(hwnd, L"Serve up on " + g_serve_host + L":" + g_serve_port);
            }
            return 0;
        }
        usage_scan(hwnd);
        return 0;
    case WM_SIZE: {
        HWND status = ::GetDlgItem(hwnd, IDC_STATUS);
        if (status != nullptr) {
            const int w = static_cast<int>(LOWORD(lParam));
            const int h = static_cast<int>(HIWORD(lParam));
            ::MoveWindow(status, 8, h - 28, w - 16, 24, TRUE);
        }
        return 0;
    }
    case WM_APP_DONE:
        ::KillTimer(hwnd, kHealthTimerId);
        ::EnableWindow(::GetDlgItem(hwnd, IDC_LAUNCH_BUTTON), TRUE);
        ::EnableWindow(::GetDlgItem(hwnd, IDC_STOP_BUTTON), FALSE);
        set_status(hwnd, wParam == 0 ? L"Serve stopped (exit 0)"
                                     : L"Serve exited with code " + std::to_wstring(wParam));
        return 0;
    case WM_CLOSE:
        save_settings(hwnd);
        ::KillTimer(hwnd, 1);
        ::KillTimer(hwnd, kHealthTimerId);
        // Terminate the serve this GUI launched so closing the window never
        // leaves orphaned GPU work behind.
        if (g_child.running.load()) {
            if (g_child.process != nullptr) { ::TerminateProcess(g_child.process, 1); }
            if (g_child.watcher.joinable()) { g_child.watcher.join(); }
        }
        ::DestroyWindow(hwnd);
        return 0;
    case WM_DESTROY:
        if (g_usage_font != nullptr) { ::DeleteObject(g_usage_font); g_usage_font = nullptr; }
        ::PostQuitMessage(0);
        return 0;
    }
    return ::DefWindowProcW(hwnd, msg, wParam, lParam);
}
} // namespace

int WINAPI wWinMain(HINSTANCE hinst, HINSTANCE, LPWSTR, int nCmdShow) {
    INITCOMMONCONTROLSEX icc { sizeof(icc), ICC_STANDARD_CLASSES | ICC_BAR_CLASSES };
    ::InitCommonControlsEx(&icc);

    WNDCLASSEXW wc { sizeof(wc) };
    wc.lpfnWndProc   = WndProc;
    wc.hInstance      = hinst;
    wc.hCursor        = ::LoadCursorW(nullptr, IDC_ARROW);
    wc.hbrBackground  = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName  = kWindowClass;
    ::RegisterClassExW(&wc);

    HWND hwnd = ::CreateWindowExW(0, kWindowClass, kWindowTitle, WS_OVERLAPPEDWINDOW,
                                  CW_USEDEFAULT, CW_USEDEFAULT, 760, 940,
                                  nullptr, nullptr, hinst, nullptr);
    if (!hwnd) { return 1; }
    create_scaffold(hwnd);
    ::SetTimer(hwnd, 1, 3000, nullptr);
    ::ShowWindow(hwnd, nCmdShow);
    ::UpdateWindow(hwnd);

    MSG msg;
    for (;;) {
        if (::GetMessageW(&msg, nullptr, 0, 0) > 0) {
            ::TranslateMessage(&msg);
            ::DispatchMessageW(&msg);
        } else {
            break;
        }
    }
    return static_cast<int>(msg.wParam);
}
