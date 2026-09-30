#define UNICODE 1
#define _UNICODE 1
#include <windows.h>
#include <windowsx.h>
#include <commctrl.h>
#include <winhttp.h>
#include <string>
#include <vector>
#include <cstdint>
#include <ctime>
#include <deque>
#include <fstream>
#include <map>
#include <optional>
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

void set_status(HWND hwnd, const wchar_t* text) {
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_STATUS), text);
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

// Path of the serve's request log: the "Request log:" control value resolved
// against the exe dir (the serve is launched with --request-log-jsonl at this
// same file). A drive-relative or rooted path is used as-is.
std::wstring request_log_path(HWND hwnd) {
    const std::wstring dir = module_dir();
    if (dir.empty()) { return {}; }
    std::wstring name = L"requests.jsonl";
    if (HWND e = ::GetDlgItem(hwnd, IDC_REQUEST_LOG_EDIT)) {
        const std::wstring text = get_control_text(e);
        if (!text.empty()) { name = text; }
    }
    if (name.size() >= 2 && name[1] == L':') { return name; }  // "C:\..."
    if (!name.empty() && (name[0] == L'\\' || name[0] == L'/')) { return name; }
    return dir + name;
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
                            nlohmann::json record;
                            try { record = nlohmann::json::parse(line); }
                            catch (const nlohmann::json::exception&) { continue; }
                            if (record.value("event", std::string()) !=
                                    "request_done" ||
                                !record.contains("result") ||
                                !record.contains("timings_seconds")) { continue; }
                            const nlohmann::json& result  = record["result"];
                            const nlohmann::json& timings = record["timings_seconds"];
                            const std::uint64_t computed =
                                result.value("computed_prefill_tokens", 0ULL);
                            const std::uint64_t out =
                                result.value("completion_tokens", 0ULL);
                            const double decode_s = timings.value("decode", 0.0);
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
    // ASCII while the literal renders as "·" regardless of file encoding.
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
    load_combo(IDC_VISION_COMBO, L"vision", 0);
    load_combo(IDC_SPEC_COMBO, L"spec", 3);
    load_edit(IDC_DRAFT_TOKENS_EDIT, L"draft_tokens");
    load_check(IDC_LM_HEAD_DRAFT_CHECK, L"lm_head_draft", 0);
    load_edit(IDC_SEED_EDIT, L"seed");
    load_check(IDC_PRESERVE_THINKING_CHECK, L"preserve_thinking", 1);
    load_edit(IDC_REQUEST_LOG_EDIT, L"request_log");
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
    save_edit(IDC_REQUEST_LOG_EDIT, L"request_log");
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

    // Row 2: context sizing
    label(L"Max context:", 8, 42);
    edit(IDC_MAX_CONTEXT_EDIT, L"", 132, 40, 90);
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

    // Row 4: VRAM headroom + request log
    label(L"VRAM headroom (MiB):", 8, 102);
    edit(IDC_HEADROOM_EDIT, L"", 132, 100, 80);  // left empty by default
    label(L"Request log:", 380, 102);
    edit(IDC_REQUEST_LOG_EDIT, L"", 510, 100, 160);
    ::SetWindowTextW(::GetDlgItem(hwnd, IDC_REQUEST_LOG_EDIT), L"requests.jsonl");

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
    combo(IDC_VISION_COMBO, 132, 310, 120, kVision, 3);  // Off = index 0
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

void create_scaffold(HWND hwnd) {
    create_core_controls(hwnd);
    load_settings(hwnd);  // override the defaults above with any saved values
    create_status(hwnd);
    create_usage_block(hwnd);
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_COMMAND:
        if (LOWORD(wParam) == IDCANCEL) { ::PostQuitMessage(0); }
        if (HIWORD(wParam) == EN_KILLFOCUS) { save_settings(hwnd); }
        return 0;
    case WM_TIMER:
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
    case WM_CLOSE:
        save_settings(hwnd);
        ::KillTimer(hwnd, 1);
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
    INITCOMMONCONTROLSEX icc { sizeof(icc), ICC_STANDARD_CLASSES };
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
