// Flash-Next P9 — frontend + output session (v1).
//
// Stub tokenizer: UTF-8 text-part bytes -> token ids = byte values (vocab 256). No
// thinking, no stop tokens, no media. preview_model accepts min(count, remaining)
// and reports finished with the budget limit reason once the budget is exhausted.

#include "targets/qwen3_8_flash_next/impl/runtime/family.h"
// M2 real mode reuses the proven qwen3_6 frontend leaf utilities: the real 248k-vocab BPE
// tokenizer (encode/decode) + the allowlist-compiled chat template (render). The real
// flash_next chat_template.jinja resolves to the existing ReasoningEffort digest, so the
// hardcoded render() is shared verbatim; only encode (input) and decode (output) replace
// the 256-vocab stub.
#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next {

namespace fqi = ninfer::targets::qwen3_6::frontend_internal;

// ---------------------------------------------------------------------------
// OutputSession
// ---------------------------------------------------------------------------

class OutputSession::Impl {
public:
    std::uint32_t budget = 0;
    std::vector<TokenId> previewed;
    // M2 real mode: the real tokenizer decodes sampled ids to text. Null in the mini stub,
    // which keeps its id-byte -> UTF-8-byte mapping byte-identical.
    std::shared_ptr<const fqi::Tokenizer> tokenizer;
};

OutputSession::~OutputSession() = default;

OutputSession::OutputSession(OutputSession&& other) noexcept : impl_(std::move(other.impl_)) {}

OutputSession& OutputSession::operator=(OutputSession&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
    }
    return *this;
}

runtime::OutputDecision
OutputSession::preview_model(std::span<const TokenId> tokens,
                             std::uint32_t total_budget_remaining, FinishReason limit_reason) {
    if (!impl_) {
        throw std::logic_error("flash_next OutputSession is empty");
    }
    const std::uint32_t count = static_cast<std::uint32_t>(tokens.size());
    const std::uint32_t accepted = std::min(count, total_budget_remaining);
    impl_->previewed.assign(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(accepted));
    const bool finished = accepted == total_budget_remaining && total_budget_remaining != 0;
    runtime::OutputDecision decision;
    decision.accepted_tokens = accepted;
    decision.continuation    = runtime::ContinuationAction::Decode;
    if (finished) {
        decision.finish_reason =
            limit_reason == FinishReason::None ? FinishReason::OutputLimit : limit_reason;
    }
    return decision;
}

std::uint32_t
OutputSession::model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept {
    return total_budget_remaining;
}

std::span<const TokenId> OutputSession::pending_control_tokens() const noexcept {
    return {};
}

runtime::OutputDecision
OutputSession::preview_control(std::span<const TokenId> tokens,
                               std::uint32_t) {
    runtime::OutputDecision decision;
    decision.accepted_tokens = static_cast<std::uint32_t>(tokens.size());
    decision.continuation    = runtime::ContinuationAction::ApplyTargetControl;
    return decision;
}

void OutputSession::validate_generation_capacity(std::uint32_t effective_output_tokens) const {
    if (!impl_) {
        throw std::logic_error("flash_next OutputSession is empty");
    }
    impl_->budget = effective_output_tokens;
}

runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    runtime::OutputDecision decision;
    decision.accepted_tokens = 0;
    decision.finish_reason   = reason;
    decision.continuation    = runtime::ContinuationAction::Decode;
    return decision;
}

PublishedOutput OutputSession::commit_preview() noexcept {
    PublishedOutput output;
    if (impl_) {
        std::string text;
        if (impl_->tokenizer) {
            // M2 real mode: decode the sampled ids with the real tokenizer. skip_special_tokens
            // drops control tokens (EOS etc.); the thinking/content split is a follow-up, so the
            // whole preview lands on the Content channel for now.
            fqi::DecodeOptions decode;
            decode.skip_special_tokens = true;
            const std::vector<TokenId>& ids = impl_->previewed;
            text = impl_->tokenizer->decode(std::span<const int>(ids.data(), ids.size()), decode);
        } else {
            // Mini stub: map each token id to the UTF-8 encoding of codepoint U+0000..U+00FF. A
            // raw id byte 0x80-0xFF is not valid standalone UTF-8, so pushing it verbatim makes
            // the response JSON non-serializable (nlohmann type_error.316); the 2-byte encoding
            // keeps the content a valid UTF-8 string for any token while leaving the ASCII path
            // (ids 0x00-0x7F) byte-identical.
            text.reserve(impl_->previewed.size());
            for (const TokenId id : impl_->previewed) {
                const std::uint32_t cp = static_cast<std::uint32_t>(static_cast<std::uint8_t>(id));
                if (cp < 0x80u) {
                    text.push_back(static_cast<char>(cp));
                } else {
                    text.push_back(static_cast<char>(0xC0u | (cp >> 6)));
                    text.push_back(static_cast<char>(0x80u | (cp & 0x3Fu)));
                }
            }
        }
        if (!text.empty()) {
            output.push_back({OutputChannel::Content, std::move(text)});
        }
        impl_->previewed.clear();
    }
    return output;
}

std::uint32_t OutputSession::reasoning_tokens() const noexcept {
    return 0;
}

ThinkingBudgetStats OutputSession::thinking_stats() const noexcept {
    return ThinkingBudgetStats{};
}

// ---------------------------------------------------------------------------
// Frontend
// ---------------------------------------------------------------------------

class Frontend::Impl {
public:
    FrontendOptions options;
    StopPolicy stop;
    // M2 real mode: non-null when built by make_real_frontend; the mini stub leaves both null.
    std::shared_ptr<const fqi::Tokenizer> tokenizer;
    std::shared_ptr<const fqi::CompiledChatTemplate> chat;

    std::vector<TokenId> tokenize(const PromptInput& input) const {
        if (!tokenizer || !chat) {
            // Mini stub: UTF-8 text-part bytes -> token ids = byte values (vocab 256).
            std::vector<TokenId> ids;
            for (const auto& message : input.messages) {
                for (const auto& part : message.parts) {
                    if (part.kind != MessagePartKind::Text) {
                        continue;
                    }
                    ids.reserve(ids.size() + part.text.size());
                    for (const char c : part.text) {
                        ids.push_back(static_cast<TokenId>(static_cast<std::uint8_t>(c)));
                    }
                }
            }
            return ids;
        }
        // M2 real mode: lower the wire ChatMessages to the frontend-internal form, render the
        // allowlist chat template, and encode the rendered text with the real tokenizer.
        std::vector<fqi::ChatMessage> messages;
        messages.reserve(input.messages.size());
        for (const auto& source : input.messages) {
            fqi::ChatMessage target;
            target.role              = source.role;
            target.reasoning_content = source.reasoning_content;
            target.tool_call_id      = source.tool_call_id;
            target.tool_calls.reserve(source.tool_calls.size());
            for (const auto& call : source.tool_calls) {
                target.tool_calls.push_back(
                    fqi::ToolCall{.id             = call.id,
                                  .name           = call.name,
                                  .arguments_json = call.arguments_json});
            }
            target.parts.reserve(source.parts.size());
            for (const auto& part : source.parts) {
                if (part.kind != MessagePartKind::Text) {
                    throw std::invalid_argument(
                        "real flash_next frontend is text-only; media is not supported yet");
                }
                target.parts.push_back(fqi::ChatPart::text_part(part.text));
            }
            messages.push_back(std::move(target));
        }
        fqi::ChatRenderOptions render{.add_generation_prompt = input.options.add_generation_prompt,
                                      .enable_thinking       = input.options.enable_thinking,
                                      .reasoning_effort      = input.options.reasoning_effort,
                                      .preserve_thinking     = input.options.preserve_thinking,
                                      .add_vision_id         = input.options.add_vision_id,
                                      .tool_jsons            = input.options.tool_jsons};
        const fqi::RenderedChat rendered = chat->render(messages, render);
        const std::vector<int> encoded   = tokenizer->encode(rendered.text, fqi::EncodeOptions{});
        std::vector<TokenId> ids;
        ids.reserve(encoded.size());
        for (const int id : encoded) {
            ids.push_back(static_cast<TokenId>(id));
        }
        return ids;
    }
};

Frontend::~Frontend() = default;

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl&) const {
    if (!impl_) {
        throw std::logic_error("flash_next Frontend is empty");
    }
    auto ids = impl_->tokenize(input);
    PreparedPrompt prompt;
    prompt.summary_.prompt_tokens = static_cast<std::uint32_t>(ids.size());
    prompt.token_ids_             = std::move(ids);
    return prompt;
}

std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl&) const {
    if (!impl_) {
        throw std::logic_error("flash_next Frontend is empty");
    }
    return static_cast<std::uint32_t>(impl_->tokenize(input).size());
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> token_ids, bool) const {
    if (!impl_) {
        throw std::logic_error("flash_next Frontend is empty");
    }
    PreparedPrompt prompt;
    prompt.summary_.prompt_tokens = static_cast<std::uint32_t>(token_ids.size());
    prompt.token_ids_             = std::move(token_ids);
    return prompt;
}

PromptCapabilities Frontend::prompt_capabilities() const {
    return PromptCapabilities{};
}

MediaCacheSummary Frontend::media_cache_summary() const {
    return MediaCacheSummary{};
}

OutputSession
Frontend::make_output_session(const PreparedPrompt&, const StopPolicy&, const OutputOptions&,
                              const ThinkingControlOptions&) const {
    OutputSession session;
    auto impl = std::make_unique<OutputSession::Impl>();
    impl->tokenizer = impl_ ? impl_->tokenizer : nullptr;
    session.impl_   = std::move(impl);
    return session;
}

const StopPolicy& Frontend::default_stop_policy() const noexcept {
    static const StopPolicy kEmptyStop;
    if (!impl_) {
        return kEmptyStop;
    }
    return impl_->stop;
}

Frontend make_frontend(FrontendOptions options) {
    Frontend frontend;
    auto impl = std::make_shared<Frontend::Impl>();
    impl->options = options;
    frontend.impl_ = std::static_pointer_cast<const Frontend::Impl>(std::move(impl));
    return frontend;
}

// M2 real-mode factory: build a real flash_next Frontend from the artifact's embedded frontend
// resources (tokenizer + chat template). The jinja resolves to the existing ReasoningEffort
// allowlist digest, so the hardcoded render() is shared with qwen3_6. Text-only (no vision);
// the default stop tokens come from the tokenizer.
Frontend make_real_frontend(std::string tokenizer_json, std::string tokenizer_config_json,
                            std::string generation_config_json, std::string chat_template_jinja,
                            FrontendOptions options) {
    fqi::TokenizerResources resources{std::string_view(tokenizer_json),
                                      std::string_view(tokenizer_config_json),
                                      std::string_view(generation_config_json)};
    auto tokenizer = std::make_shared<const fqi::Tokenizer>(resources);
    const fqi::CompiledChatTemplate template_ =
        fqi::CompiledChatTemplate::resolve(std::string_view(chat_template_jinja));
    auto chat = std::make_shared<const fqi::CompiledChatTemplate>(template_);

    Frontend frontend;
    auto impl = std::make_shared<Frontend::Impl>();
    impl->options   = options;
    impl->tokenizer = tokenizer;
    impl->chat      = chat;
    for (const int token : tokenizer->default_stop_token_ids()) {
        impl->stop.token_ids.push_back(static_cast<TokenId>(token));
    }
    frontend.impl_ = std::static_pointer_cast<const Frontend::Impl>(std::move(impl));
    return frontend;
}

}  // namespace ninfer::targets::qwen3_8_flash_next
