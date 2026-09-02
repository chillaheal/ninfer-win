// Serve-side execution of the emulated Anthropic server tools (web search,
// web fetch). libcurl is used the same way as in media_acquire: lazy global
// init, NOSIGNAL (httplib worker threads), SSL peer/host verification.
//
// Search backend is keyless DuckDuckGo Lite (https://lite.duckduckgo.com),
// whose simple table markup (result-link / result-snippet) is parsed with
// plain string scans. Results come back as DDG redirect links; the real URL
// is the percent-encoded `uddg` query parameter.

#include "serve/server_tools.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <curl/curl.h>

namespace ninfer::serve {
namespace {

constexpr std::size_t kMaxResponseBodyBytes = 4u << 20; // 4 MiB (real home pages exceed 1 MiB)
constexpr std::size_t kMaxTextResultChars   = 24000;
constexpr int kSearchTimeoutMs              = 15000;
constexpr int kFetchTimeoutMs               = 20000;
constexpr int kMaxSearchResults             = 8;
constexpr const char* kUserAgent =
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36";

struct HttpResponse {
    bool ok             = false;
    long status         = 0;
    std::string content_type;
    std::vector<std::uint8_t> body;
    std::string error;
};

struct CurlBuffer {
    std::vector<std::uint8_t> bytes;
    std::size_t limit;
};

std::size_t curl_write(char* data, std::size_t size, std::size_t count, void* opaque) {
    auto& out = *static_cast<CurlBuffer*>(opaque);
    if (size != 0 && count > std::numeric_limits<std::size_t>::max() / size) { return 0; }
    const std::size_t amount = size * count;
    if (out.bytes.size() + amount > out.limit) { return 0; }
    out.bytes.insert(out.bytes.end(), reinterpret_cast<std::uint8_t*>(data),
                     reinterpret_cast<std::uint8_t*>(data) + amount);
    return amount;
}

HttpResponse fetch_http(const std::string& url, int timeout_ms) {
    static std::once_flag init;
    std::call_once(init, [] {
        if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
            throw std::runtime_error("failed to initialize libcurl");
        }
    });
    HttpResponse out;
    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(),
                                                             curl_easy_cleanup);
    if (!curl) {
        out.error = "failed to create libcurl handle";
        return out;
    }
    CurlBuffer buffer{.bytes = {}, .limit = kMaxResponseBodyBytes};
    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_HTTPGET, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS_STR, "http,https");
    curl_easy_setopt(curl.get(), CURLOPT_REDIR_PROTOCOLS_STR, "http,https");
    curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_MAXREDIRS, 3L);
    curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT_MS, 8000L);
    curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT_MS, static_cast<long>(timeout_ms));
    curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_SSL_VERIFYHOST, 2L);
    curl_easy_setopt(curl.get(), CURLOPT_PROXY, "");
    curl_easy_setopt(curl.get(), CURLOPT_MAXFILESIZE_LARGE,
                     static_cast<curl_off_t>(kMaxResponseBodyBytes));
    curl_easy_setopt(curl.get(), CURLOPT_USERAGENT, kUserAgent);
    // A bare curl request (User-Agent only, no Accept / Accept-Language) is 403'd by the
    // WAFs of many real sites — the full browser header set is what passes (verified:
    // netonnet.se 403 -> 200 with UA + Accept + Accept-Language).
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(
        headers, "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8");
    headers = curl_slist_append(headers, "Accept-Language: en-US,en;q=0.9");
    curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, curl_write);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &buffer);
    const CURLcode code = curl_easy_perform(curl.get());
    curl_slist_free_all(headers); // consumed by perform; free once it has returned
    if (code == CURLE_WRITE_ERROR || code == CURLE_FILESIZE_EXCEEDED) {
        out.error = "response exceeds the byte limit";
        return out;
    }
    if (code != CURLE_OK) {
        out.error = curl_easy_strerror(code);
        return out;
    }
    long status = 0;
    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &status);
    char* content_type = nullptr;
    curl_easy_getinfo(curl.get(), CURLINFO_CONTENT_TYPE, &content_type);
    if (content_type != nullptr) { out.content_type = content_type; }
    out.status = status;
    out.body   = std::move(buffer.bytes);
    if (status >= 200 && status < 300) {
        out.ok = true;
    } else {
        out.error = "HTTP " + std::to_string(status);
    }
    return out;
}

// --- URL / text helpers ------------------------------------------------------

std::string url_encode(std::string_view value) {
    static const char* kHex = "0123456789ABCDEF";
    std::string out;
    out.reserve(value.size() * 3);
    for (const char ch : value) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out.push_back(ch);
        } else {
            out.push_back('%');
            out.push_back(kHex[c >> 4]);
            out.push_back(kHex[c & 0xF]);
        }
    }
    return out;
}

std::optional<int> hex_value(char c) {
    if (c >= '0' && c <= '9') { return c - '0'; }
    if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
    if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
    return std::nullopt;
}

std::string url_decode(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
        const char c = value[i];
        if (c == '%' && i + 2 < value.size()) {
            const std::optional<int> hi = hex_value(value[i + 1]);
            const std::optional<int> lo = hex_value(value[i + 2]);
            if (hi && lo) {
                out.push_back(static_cast<char>(*hi * 16 + *lo));
                i += 2;
                continue;
            }
        }
        out.push_back(c == '+' ? ' ' : c);
    }
    return out;
}

void append_codepoint(std::string& out, unsigned long cp) {
    if (cp == 0 || cp > 0x10FFFF) { cp = 0xFFFD; }
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::optional<char> named_entity_char(const std::string& name) {
    static const std::pair<const char*, char> kNamed[] = {
        {"amp", '&'},  {"lt", '<'},   {"gt", '>'},  {"quot", '"'},
        {"apos", '\''}, {"nbsp", ' '},
    };
    for (const auto& entry : kNamed) {
        if (name == entry.first) { return entry.second; }
    }
    return std::nullopt;
}

void decode_entities(const std::string& in, std::string& out) {
    std::size_t i = 0;
    while (i < in.size()) {
        const char c = in[i];
        if (c != '&') {
            out.push_back(c);
            ++i;
            continue;
        }
        const std::size_t semi = in.find(';', i + 1);
        if (semi == std::string::npos || semi - i > 12) {
            out.push_back(c);
            ++i;
            continue;
        }
        const std::string_view entity(in.data() + i + 1, semi - i - 1);
        unsigned long cp    = 0;
        bool decoded        = false;
        if (entity.size() >= 2 && entity[0] == '#') {
            const bool hex             = entity.size() >= 3 && (entity[1] == 'x' || entity[1] == 'X');
            const std::size_t digits   = hex ? 2 : 1;
            cp                         = 0;
            bool all_digits            = entity.size() > digits;
            for (std::size_t d = digits; d < entity.size() && all_digits; ++d) {
                const char dc = entity[d];
                if (hex) {
                    const std::optional<int> v = hex_value(dc);
                    if (!v) { all_digits = false; break; }
                    cp = cp * 16UL + static_cast<unsigned long>(*v);
                } else {
                    if (dc < '0' || dc > '9') { all_digits = false; break; }
                    cp = cp * 10UL + static_cast<unsigned long>(dc - '0');
                }
            }
            if (all_digits && cp <= 0x10FFFF) { decoded = true; }
        } else {
            const std::optional<char> single = named_entity_char(std::string(entity));
            if (single) {
                decoded = true;
                cp      = static_cast<unsigned long>(static_cast<unsigned char>(*single));
            }
        }
        if (decoded) {
            append_codepoint(out, cp);
            i = semi + 1;
        } else {
            out.push_back(c);
            ++i;
        }
    }
}

std::string normalize_whitespace(std::string_view text, std::size_t max_chars) {
    std::string out;
    out.reserve(std::min(text.size(), max_chars) + 16);
    int consecutive_blank = 0;
    std::string line;
    auto flush_line = [&] {
        std::size_t b = 0;
        std::size_t e = line.size();
        while (b < e && std::isspace(static_cast<unsigned char>(line[b]))) { ++b; }
        while (e > b && std::isspace(static_cast<unsigned char>(line[e - 1]))) { --e; }
        line = line.substr(b, e - b);
        if (line.empty()) {
            ++consecutive_blank;
            return;
        }
        if (consecutive_blank > 0) {
            out += "\n";
            if (consecutive_blank > 1) { out += "\n"; }
            consecutive_blank = 0;
        }
        if (!out.empty()) { out += "\n"; }
        out += line;
        line.clear();
    };
    for (const char c : text) {
        if (c == '\n') {
            flush_line();
        } else if (c == '\r') {
            continue;
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            line.push_back(' ');
        } else {
            line.push_back(c);
        }
    }
    flush_line();
    if (out.size() > max_chars) {
        out.resize(max_chars);
        out += " [truncated]";
    }
    return out;
}

std::string inline_text(std::string_view html) {
    std::string flat;
    flat.reserve(html.size());
    std::size_t i = 0;
    while (i < html.size()) {
        if (html[i] == '<') {
            const std::size_t close = html.find('>', i);
            if (close == std::string_view::npos) { break; }
            i = close + 1;
            continue;
        }
        flat.push_back(html[i]);
        ++i;
    }
    std::string decoded;
    decoded.reserve(flat.size());
    decode_entities(flat, decoded);
    return normalize_whitespace(decoded, std::numeric_limits<std::size_t>::max());
}

std::string html_to_text(std::string_view html, std::size_t max_chars) {
    std::string text(html);
    static const std::regex comment_re(R"(<!--[\s\S]*?-->)",
                                       std::regex::ECMAScript | std::regex::icase |
                                           std::regex::optimize);
    static const std::regex stripped_block_re(
        R"(<(script|style|noscript)[\s\S]*?</\1>)",
        std::regex::ECMAScript | std::regex::icase | std::regex::optimize);
    static const std::regex block_boundary_re(
        R"(</(p|div|li|tr|h[1-6]|ul|ol|table|section|article|header|footer|blockquote|pre|dl|dd|dt|form|main|nav|aside)>|<br\s*/?>)",
        std::regex::ECMAScript | std::regex::icase | std::regex::optimize);
    static const std::regex tag_re(R"(<[^>\n]*>)",
                                   std::regex::ECMAScript | std::regex::icase |
                                       std::regex::optimize);
    text = std::regex_replace(text, comment_re, "");
    text = std::regex_replace(text, stripped_block_re, "");
    text = std::regex_replace(text, block_boundary_re, "\n");
    text = std::regex_replace(text, tag_re, "");
    std::string decoded;
    decoded.reserve(text.size());
    decode_entities(text, decoded);
    return normalize_whitespace(decoded, max_chars);
}

std::string extract_attribute(const std::string& tag, const char* attribute) {
    const std::string marker_prefix = std::string(attribute) + "=";
    for (const char quote : {'"', '\''}) {
        const std::string marker = marker_prefix + quote;
        const std::size_t pos = tag.find(marker);
        if (pos == std::string::npos) { continue; }
        const std::size_t start = pos + marker.size();
        const std::size_t end   = tag.find(quote, start);
        if (end != std::string::npos) { return tag.substr(start, end - start); }
    }
    return {};
}

// --- DuckDuckGo Lite parsing --------------------------------------------------

struct SearchResult {
    std::string title;
    std::string url;
    std::string snippet;
};

std::string resolve_ddg_href(const std::string& href) {
    const std::size_t uddg = href.find("uddg=");
    if (uddg != std::string::npos) {
        const std::size_t start = uddg + 5;
        const std::size_t end   = href.find('&', start);
        const std::string_view encoded =
            end == std::string::npos ? std::string_view(href).substr(start)
                                     : std::string_view(href).substr(start, end - start);
        const std::string decoded = url_decode(encoded);
        if (decoded.rfind("http", 0) == 0) { return decoded; }
    }
    if (href.rfind("http://", 0) == 0 || href.rfind("https://", 0) == 0) { return href; }
    if (href.rfind("//", 0) == 0) { return "https:" + href; }
    return {};
}

std::vector<SearchResult> parse_ddg_lite(const std::string& html, int max_results) {
    std::vector<SearchResult> results;
    std::size_t pos = 0;
    while (results.size() < static_cast<std::size_t>(max_results) &&
           (pos = html.find("result-link", pos)) != std::string::npos) {
        const std::size_t tag_start = html.rfind("<a", pos);
        const std::size_t tag_end   = html.find("</a>", pos);
        if (tag_start == std::string::npos || tag_end == std::string::npos ||
            tag_start > tag_end) {
            pos = tag_end == std::string::npos ? html.size() : tag_end;
            continue;
        }
        const std::string tag = html.substr(tag_start, tag_end - tag_start);
        const std::string url = resolve_ddg_href(extract_attribute(tag, "href"));
        const std::size_t text_start = html.find('>', pos);
        if (!url.empty() && text_start != std::string::npos && text_start < tag_end) {
            const std::string title =
                inline_text(html.substr(text_start + 1, tag_end - text_start - 1));
            std::string snippet;
            const std::size_t snippet_pos = html.find("result-snippet", tag_end);
            if (snippet_pos != std::string::npos) {
                const std::size_t snippet_open = html.find('>', snippet_pos);
                const std::size_t td_end       = html.find("</td>", snippet_pos);
                if (snippet_open != std::string::npos && td_end != std::string::npos &&
                    snippet_open < td_end) {
                    snippet =
                        inline_text(html.substr(snippet_open + 1, td_end - snippet_open - 1));
                }
            }
            results.push_back(SearchResult{std::move(title), std::move(url),
                                           std::move(snippet)});
        }
        pos = tag_end;
    }
    return results;
}

// DDG-lite rate-limits by IP with a transient "anomaly challenge" (HTTP 202, or a
// 200 page full of "anomaly" markers but no result rows). Detected so the model
// gets an actionable message instead of challenge-page garbage.
bool is_ddg_anomaly_challenge(const HttpResponse& response) {
    if (response.status == 202) { return true; }
    std::string lower;
    lower.reserve(response.body.size());
    for (const unsigned char c : response.body) {
        lower.push_back(static_cast<char>(std::tolower(c)));
    }
    return lower.find("anomaly") != std::string::npos &&
           lower.find("result-link") == std::string::npos;
}

// Short-lived result cache: repeated identical queries (model retries, several
// parallel conversations) would each hit DDG-lite and re-trigger the rate limit.
// Successful results only — errors are never cached, so a retry after the cooldown
// actually re-fetches.
struct SearchCacheEntry {
    std::chrono::steady_clock::time_point expires;
    std::string text;
};
constexpr std::chrono::milliseconds kSearchCacheTtl{600000}; // 10 min
constexpr std::size_t kSearchCacheMaxEntries = 64;
std::mutex g_search_cache_mutex;
std::map<std::string, SearchCacheEntry> g_search_cache;

} // namespace

std::string run_web_search(std::string_view query) {
    const std::string query_string(query);
    if (query_string.empty()) { return "Web search failed: empty query"; }

    {
        std::lock_guard<std::mutex> lock(g_search_cache_mutex);
        const auto it = g_search_cache.find(query_string);
        if (it != g_search_cache.end() && it->second.expires > std::chrono::steady_clock::now()) {
            return it->second.text;
        }
    }

    const HttpResponse response =
        fetch_http("https://lite.duckduckgo.com/lite/?q=" + url_encode(query_string),
                   kSearchTimeoutMs);
    if (!response.ok) { return "Web search failed: " + response.error; }
    if (is_ddg_anomaly_challenge(response)) {
        return "Web search failed: the search provider is rate-limiting this machine "
               "(anomaly challenge). Wait a few minutes and try the search again.";
    }
    const std::string html(response.body.begin(), response.body.end());
    const std::vector<SearchResult> results = parse_ddg_lite(html, kMaxSearchResults);
    if (results.empty()) {
        return "Web search returned no results for: " + query_string;
    }
    std::string out;
    out += "Web search results for: ";
    out += query_string;
    out += "\n\n";
    int index = 1;
    for (const SearchResult& result : results) {
        out += std::to_string(index);
        out += ". ";
        out += result.title;
        out += "\nURL: ";
        out += result.url;
        out += "\n";
        if (!result.snippet.empty()) {
            out += result.snippet;
            out += "\n";
        }
        out += "\n";
        ++index;
    }
    {
        std::lock_guard<std::mutex> lock(g_search_cache_mutex);
        const auto now = std::chrono::steady_clock::now();
        for (auto it = g_search_cache.begin(); it != g_search_cache.end();) {
            it = (it->second.expires <= now) ? g_search_cache.erase(it) : std::next(it);
        }
        if (g_search_cache.size() >= kSearchCacheMaxEntries) { g_search_cache.clear(); }
        g_search_cache.emplace(query_string, SearchCacheEntry{now + kSearchCacheTtl, out});
    }
    return out;
}

std::string run_web_fetch(std::string_view url) {
    const std::string url_string(url);
    if (url_string.rfind("http://", 0) != 0 && url_string.rfind("https://", 0) != 0) {
        return "Web fetch failed: URL must start with http:// or https://";
    }
    const HttpResponse response = fetch_http(url_string, kFetchTimeoutMs);
    if (!response.ok) {
        return "Web fetch failed for " + url_string + ": " + response.error;
    }
    const std::string raw_body(
        reinterpret_cast<const char*>(response.body.data()), response.body.size());
    std::string_view body(raw_body);
    while (!body.empty() && (body.front() == ' ' || body.front() == '\t' ||
                             body.front() == '\r' || body.front() == '\n')) {
        body.remove_prefix(1);
    }
    const bool html_page =
        body.rfind("<!doctype html", 0) == 0 || body.rfind("<html", 0) == 0 ||
        response.content_type.find("html") != std::string::npos;
    std::string content;
    if (html_page) {
        content = html_to_text(body, kMaxTextResultChars);
    } else {
        std::string raw(body);
        std::string decoded;
        decoded.reserve(raw.size());
        decode_entities(raw, decoded);
        content = normalize_whitespace(decoded, kMaxTextResultChars);
    }
    std::string out;
    out += "Web page content for ";
    out += url_string;
    out += ":\n\n";
    out += content;
    return out;
}

} // namespace ninfer::serve
