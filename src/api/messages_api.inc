// Adapted from official 91b111aeb4bf85723433a821f200398ca8d91cc2; local generation/XML policy retained.
// ------------------------------------------------- Anthropic Messages API --
// POST /v1/messages and /v1/messages/count_tokens. The Anthropic request is
// translated into the chat-completions body shape and then runs through the
// same override / template / tool-parser path as /v1/chat/completions; only
// the wire format differs: content blocks in, content blocks and named SSE
// events out, errors in Anthropic's {"type":"error",...} envelope.
// x-api-key / anthropic-version / anthropic-beta headers are accepted and
// ignored (the inference endpoints carry no auth, same as the OpenAI ones).

const char* anthropic_error_type(int status) {
    switch (status) {
        case 400: return "invalid_request_error";
        case 401: return "authentication_error";
        case 403: return "permission_error";
        case 404: return "not_found_error";
        case 413: return "request_too_large";
        case 429: return "rate_limit_error";
        case 503: return "overloaded_error";
        default: return status >= 500 ? "api_error" : "invalid_request_error";
    }
}

json anthropic_error_json(int status, const std::string& message) {
    return json{{"type", "error"},
                {"error", {{"type", anthropic_error_type(status)}, {"message", message}}}};
}

// One named SSE event: `event: <type>\ndata: <json>\n\n`.
bool send_anthropic_event(http::Stream* st, const char* type, const json& data) {
    return st->send_raw(std::string("event: ") + type + "\ndata: " +
                        json_py::dumps(data, /*spaced=*/false) + "\n\n");
}

std::string block_type(const json& block) {
    if (!block.is_object()) http::fail(400, "content blocks must be objects");
    if (!block.contains("type") || !block["type"].is_string())
        http::fail(400, "content blocks require a string type");
    return block["type"].get<std::string>();
}

const std::string& block_string(const json& block, const char* key, const char* type) {
    if (!block.contains(key) || !block[key].is_string())
        http::fail(400, std::string(type) + " blocks require a string " + key);
    return block[key].get_ref<const std::string&>();
}

// Text of a string or an array of text blocks (system prompt), joined by `sep`.
std::string anthropic_text(const json& content, const char* field, const char* sep) {
    if (content.is_null()) return {};
    if (content.is_string()) return content.get<std::string>();
    if (!content.is_array())
        http::fail(400, std::string(field) + " must be a string or an array of text blocks");
    std::string text;
    for (const auto& block : content) {
        if (block_type(block) != "text")
            http::fail(400, std::string(field) + " must contain only text blocks");
        if (!text.empty()) text += sep;
        text += block_string(block, "text", "text");
    }
    return text;
}

// Anthropic image source -> the image_url part the vision path decodes. URL
// sources are passed through so vision:: rejects them with its usual message.
json anthropic_image(const json& block) {
    if (!block.contains("source") || !block["source"].is_object())
        http::fail(400, "image blocks require a source object");
    const json& source = block["source"];
    const std::string type =
        source.contains("type") && source["type"].is_string() ? source["type"].get<std::string>()
                                                              : std::string();
    std::string url;
    if (type == "base64") {
        if (!source.contains("media_type") || !source["media_type"].is_string() ||
            !source.contains("data") || !source["data"].is_string())
            http::fail(400, "base64 image sources require string media_type and data");
        url = "data:" + source["media_type"].get<std::string>() + ";base64," +
              source["data"].get<std::string>();
    } else if (type == "url") {
        url = block_string(source, "url", "url image source");
    } else {
        http::fail(400, "image source type '" + type +
                            "' is not supported; send the image as base64");
    }
    return json{{"type", "image_url"}, {"image_url", {{"url", std::move(url)}}}};
}

// Appends a text part, merging it into a preceding text part: the template
// concatenates content parts without a separator, which would glue Claude
// Code's separate reminder / prompt blocks into one line.
void push_text_part(json* parts, const std::string& text) {
    if (!parts->empty() && parts->back().contains("text")) {
        std::string merged = parts->back()["text"].get<std::string>();
        if (!merged.empty() && !text.empty()) merged += "\n\n";
        parts->back()["text"] = merged + text;
        return;
    }
    parts->push_back(json{{"type", "text"}, {"text", text}});
}

// A single text part collapses to a plain string.
json collapse_parts(json parts) {
    if (parts.size() == 1 && parts[0].contains("text")) return parts[0]["text"];
    return parts;
}

json anthropic_tool_result(const json& block) {
    const std::string& call_id = block_string(block, "tool_use_id", "tool_result");
    json content = "";
    if (block.contains("content") && !block["content"].is_null()) {
        const json& raw = block["content"];
        if (raw.is_string()) {
            content = raw;
        } else if (raw.is_array()) {
            json parts = json::array();
            for (const auto& part : raw) {
                const std::string type = block_type(part);
                if (type == "text") push_text_part(&parts, block_string(part, "text", "text"));
                else if (type == "image") parts.push_back(anthropic_image(part));
                else http::fail(400, "unsupported block type in tool_result content: " + type);
            }
            content = parts.empty() ? json("") : collapse_parts(std::move(parts));
        } else {
            http::fail(400, "tool_result content must be a string or an array of blocks");
        }
    }
    return json{{"role", "tool"}, {"tool_call_id", call_id}, {"content", std::move(content)}};
}

// Anthropic system + messages -> chat messages. tool_result blocks become
// role "tool" messages placed before the rest of their user turn: the
// template needs them right after the assistant turn that made the calls.
json anthropic_messages(const json& in) {
    json out = json::array();
    if (in.contains("system") && !in["system"].is_null())
        out.push_back(json{{"role", "system"},
                           {"content", anthropic_text(in["system"], "system", "\n\n")}});
    if (!in.contains("messages") || !in["messages"].is_array() || in["messages"].empty())
        http::fail(400, "messages: at least one message is required");
    for (const auto& m : in["messages"]) {
        if (!m.is_object() || !m.contains("role") || !m["role"].is_string())
            http::fail(400, "messages entries require a string role");
        const std::string role = m["role"].get<std::string>();
        if (role != "user" && role != "assistant")
            http::fail(400, "messages: role must be 'user' or 'assistant'");
        if (!m.contains("content")) http::fail(400, "messages entries require content");
        const json& content = m["content"];
        if (content.is_string()) {
            out.push_back(json{{"role", role}, {"content", content}});
            continue;
        }
        if (!content.is_array())
            http::fail(400, "message content must be a string or an array of blocks");

        if (role == "user") {
            json parts = json::array();
            for (const auto& block : content) {
                const std::string type = block_type(block);
                if (type == "text") {
                    push_text_part(&parts, block_string(block, "text", "text"));
                } else if (type == "image") {
                    parts.push_back(anthropic_image(block));
                } else if (type == "tool_result") {
                    out.push_back(anthropic_tool_result(block));
                } else if (type == "document") {
                    const json* source = block.contains("source") ? &block["source"] : nullptr;
                    if (source == nullptr || !source->is_object() || !source->contains("type") ||
                        (*source)["type"] != "text")
                        http::fail(400, "only plain-text document sources are supported");
                    push_text_part(&parts, block_string(*source, "data", "document source"));
                } else {
                    http::fail(400, "unsupported content block type in a user message: " + type);
                }
            }
            if (!parts.empty())
                out.push_back(json{{"role", "user"}, {"content", collapse_parts(std::move(parts))}});
            continue;
        }

        std::string text, reasoning;
        json calls = json::array();
        for (const auto& block : content) {
            const std::string type = block_type(block);
            if (type == "text") {
                text += block_string(block, "text", "text");
            } else if (type == "thinking") {
                reasoning += block_string(block, "thinking", "thinking");
            } else if (type == "redacted_thinking") {
                continue;  // opaque to this server
            } else if (type == "tool_use") {
                json input = block.contains("input") && !block["input"].is_null()
                                 ? block["input"]
                                 : json::object();
                if (!input.is_object()) http::fail(400, "tool_use input must be an object");
                calls.push_back(json{{"id", block_string(block, "id", "tool_use")},
                                     {"type", "function"},
                                     {"function",
                                      {{"name", block_string(block, "name", "tool_use")},
                                       {"arguments", std::move(input)}}}});
            } else {
                http::fail(400, "unsupported content block type in an assistant message: " + type);
            }
        }
        json message{{"role", "assistant"}, {"content", text}};
        if (!reasoning.empty()) message["reasoning_content"] = reasoning;
        if (!calls.empty()) message["tool_calls"] = std::move(calls);
        out.push_back(std::move(message));
    }
    return out;
}

// Anthropic body -> chat-completions body. A trailing assistant message is a
// prefill: it is removed from `messages` and returned in *prefill, and the
// reply continues from it (the prefill itself is not echoed back).
json anthropic_to_chat(const json& in, std::string* prefill) {
    json out = json::object();
    out["messages"] = anthropic_messages(in);
    if (in.contains("tool_call_format")) out["tool_call_format"] = in["tool_call_format"];
    for (const char* key : {"preserve_reasoning", "auto_disable_thinking_with_tools",
                            "max_tool_arg_chars", "max_tool_response_chars", "think_guard_test_cut"})
        if (in.contains(key)) http::fail(400, std::string("unsupported extension: ") + key);
    if (in["messages"].back()["role"] == "assistant") {
        json& last = out["messages"].back();
        if (last.contains("tool_calls"))
            http::fail(400, "the final assistant message cannot contain tool_use blocks");
        *prefill = last["content"].get<std::string>();
        out["messages"].erase(out["messages"].size() - 1);
    }

    // Anthropic sampling fields share the chat names; the rest are this
    // server's chat extensions, accepted here too.
    for (const char* k : {"max_tokens", "temperature", "top_p", "top_k", "stream", "seed",
                          "min_p", "presence_penalty", "frequency_penalty", "logit_bias",
                          "enable_thinking", "preserve_thinking", "reasoning_effort",
                          "add_vision_id"})
        if (in.contains(k)) out[k] = in[k];
    if (in.contains("stop_sequences") && !in["stop_sequences"].is_null()) {
        if (!in["stop_sequences"].is_array())
            http::fail(400, "stop_sequences must be an array of strings");
        out["stop"] = in["stop_sequences"];
    }

    // thinking -> enable_thinking. budget_tokens has no engine equivalent and
    // is ignored. Omitted => left unset, so the server override table (or the
    // template default) decides.
    if (in.contains("thinking") && !in["thinking"].is_null() && !out.contains("enable_thinking")) {
        const json& t = in["thinking"];
        if (!t.is_object() || !t.contains("type") || !t["type"].is_string())
            http::fail(400, "thinking must be an object with a string type");
        const std::string type = t["type"].get<std::string>();
        if (type == "enabled" || type == "adaptive") out["enable_thinking"] = true;
        else if (type == "disabled") out["enable_thinking"] = false;
        else http::fail(400, "thinking.type must be 'enabled', 'adaptive' or 'disabled'");
    }
    if (in.contains("output_config") && in["output_config"].is_object() &&
        in["output_config"].contains("effort") && !out.contains("reasoning_effort"))
        out["reasoning_effort"] = in["output_config"]["effort"];

    if (in.contains("tools") && !in["tools"].is_null()) {
        if (!in["tools"].is_array()) http::fail(400, "tools must be an array");
        json tools = json::array();
        for (const auto& t : in["tools"]) {
            if (!t.is_object() || !t.contains("name") || !t["name"].is_string())
                http::fail(400, "tools entries require a string name");
            // Server tools (web_search, code_execution, ...) have a versioned
            // type and no input_schema: Anthropic runs those, this server can't.
            if (t.contains("type") && t["type"] != "custom" && !t.contains("input_schema"))
                continue;
            json function{{"name", t["name"]}};
            if (t.contains("description")) function["description"] = t["description"];
            function["parameters"] =
                t.contains("input_schema")
                    ? t["input_schema"]
                    : json{{"type", "object"}, {"properties", json::object()}};
            tools.push_back(json{{"type", "function"}, {"function", std::move(function)}});
        }
        out["tools"] = std::move(tools);
    }
    if (in.contains("tool_choice") && !in["tool_choice"].is_null()) {
        const json& c = in["tool_choice"];
        if (!c.is_object() || !c.contains("type") || !c["type"].is_string())
            http::fail(400, "tool_choice must be an object with a string type");
        const std::string type = c["type"].get<std::string>();
        if (type == "auto") out["tool_choice"] = "auto";
        else if (type == "any") out["tool_choice"] = "required";
        else if (type == "none") out["tool_choice"] = "none";
        else if (type == "tool")
            out["tool_choice"] = json{{"type", "function"},
                                      {"function", {{"name", block_string(c, "name", "tool_choice")}}}};
        else http::fail(400, "tool_choice.type must be 'auto', 'any', 'tool' or 'none'");
    }
    return out;
}

// Request state shared by /v1/messages and /v1/messages/count_tokens. Not
// copyable: opts points into body and tool_setup.
struct MessagesPrep {
    json body;
    json messages;
    std::string prefill;
    ToolSetup tool_setup;
    chat_template::Options opts;
    json normalized_effort;
    std::string tool_format = "xml";
    bool thinking = false;  // the reply opens inside <think>
    GenSpec spec;

    MessagesPrep() = default;
    MessagesPrep(const MessagesPrep&) = delete;
    MessagesPrep& operator=(const MessagesPrep&) = delete;
};

void prepare_messages_request(const http::Request& q, http::Response* r, MessagesPrep* p,
                              bool count_only) {
    if (g_tok.vocab_size() == 0)
        http::fail(503, "tokenizer not loaded (" + g_cfg.tokenizer_dir +
                        "); check TOKENIZER_DIR in service.conf");
    p->body = anthropic_to_chat(parse_body(q), &p->prefill);
    apply_overrides(&p->body, Api::Chat, r);
    const json& body = p->body;
    p->messages = body["messages"];
    check_model_modalities(p->messages);
    std::string error;
    if (!toolparse::normalize_messages(&p->messages, &error)) http::fail(400, error);
    std::vector<vision::Frame> frames;
    if (!vision::prepare_messages(p->messages, &frames, &error)) http::fail(400, error);

    p->tool_setup = parse_tool_setup(body);
    if (p->tool_setup.choice.forced() && !p->prefill.empty())
        http::fail(400, "an assistant prefill cannot be combined with a forced tool_choice");
    chat_template::Options& opts = p->opts;
    if (p->tool_setup.enabled()) opts.tools = &p->tool_setup.tools;
    static const json kTrue = true;
    opts.add_generation_prompt = &kTrue;
    if (body.contains("reasoning_effort"))
        opts.reasoning_effort =
            normalize_reasoning_effort(&body["reasoning_effort"], &p->normalized_effort);
    require_optional_boolean(body, "enable_thinking");
    require_optional_boolean(body, "preserve_thinking");
    require_optional_boolean(body, "add_vision_id");
    if (body.contains("enable_thinking") && !body["enable_thinking"].is_null())
        opts.enable_thinking = &body["enable_thinking"];
    if (body.contains("preserve_thinking") && !body["preserve_thinking"].is_null())
        opts.preserve_thinking = &body["preserve_thinking"];
    // Retain local XML tooling; do not silently accept upstream-only formats.
    if (body.contains("tool_call_format") && body["tool_call_format"] != "xml")
        http::fail(400, "this deployment supports XML tool_call_format only");
    if (body.contains("add_vision_id")) opts.add_vision_id = &body["add_vision_id"];

    chat_template::RenderResult rr = chat_template::render_chat_template(&p->messages, opts);
    if (!rr.ok) http::fail(400, rr.error);
    // Forced tool calls and prefills both close the think block in the prompt
    // (ToolChoice::prompt_suffix does it for the former), so the reply carries
    // no reasoning to split off.
    p->thinking = rr.thinking_enabled && !p->tool_setup.choice.forced() && p->prefill.empty();
    rr.text += p->tool_setup.choice.prompt_suffix(rr.thinking_enabled);
    if (!p->prefill.empty()) {
        if (rr.thinking_enabled) rr.text += "</think>\n\n";
        const size_t at = rr.text.size();
        rr.text += p->prefill;
        rr.literal.push_back({at, rr.text.size(), true});
    }

    GenSpec& spec = p->spec;
    std::vector<size_t> starts;
    if (!encode_spliced(rr.text, &frames, &spec.ids, &starts, &error, &rr.literal)) http::fail(400, error);
    if (count_only) return;
    for (auto& frame : frames) {
        spec.mrope_grids.push_back(frame.grid);
        spec.patches.push_back(std::move(frame.patches));
    }
    if (spec.ids.empty()) http::fail(400, kEngineRejected);
    spec.snaps = compute_snap_cuts(p->messages, opts, rr.text, spec.ids, starts);
    apply_sampling(body, &spec);
}

// Complete tool calls carry valid JSON objects; anything else (cannot happen
// for a finished call) degrades to an empty input rather than a 500.
json tool_use_input(const std::string& arguments) {
    if (arguments.empty()) return json::object();
    try {
        json value = json::parse(arguments);
        if (value.is_object()) return value;
    } catch (...) {
    }
    return json::object();
}

bool has_visible_text(const std::string& s) {
    return s.find_first_not_of(" \t\r\n") != std::string::npos;
}

void messages_impl(const http::Request& q, http::Response* r, http::Stream* st) {
    MessagesPrep p;
    prepare_messages_request(q, r, &p, /*count_only=*/false);
    GenSpec& spec = p.spec;
    const ToolSetup& tool_setup = p.tool_setup;
    const bool stream = bool_field(p.body, "stream", false) && st != nullptr;
    const std::string id = make_id("msg_");

    // Anthropic counts cache reads apart from input_tokens.
    auto usage = [&](const GenOutcome* o) {
        const long long prompt = o != nullptr && o->n_prompt > 0
                                     ? o->n_prompt
                                     : static_cast<long long>(spec.ids.size());
        const long long cached = o != nullptr ? o->n_cached : 0;
        json u{{"input_tokens", prompt - cached},
               {"cache_creation_input_tokens", 0},
               {"cache_read_input_tokens", cached},
               {"output_tokens", o != nullptr ? o->n_gen : 0}};
        return u;
    };
    auto stop_reason = [](const GenOutcome& o, bool partial_call, bool has_calls) {
        if (o.reason == "length" || partial_call) return "max_tokens";
        if (has_calls) return "tool_use";
        if (o.hit_stop) return "stop_sequence";
        return "end_turn";
    };
    auto stop_sequence = [](const GenOutcome& o) {
        return o.hit_stop ? json(o.stop_hit) : json(nullptr);
    };
    auto message_object = [&](json content, json reason, json sequence, json u) {
        json m;
        m["id"] = id;
        m["type"] = "message";
        m["role"] = "assistant";
        m["model"] = g_cfg.model;
        m["content"] = std::move(content);
        m["stop_reason"] = std::move(reason);
        m["stop_sequence"] = std::move(sequence);
        m["usage"] = std::move(u);
        return m;
    };
    auto make_parser = [&]() {
        return toolparse::StreamParser(
            tool_setup.tools, []() { return make_id("toolu_"); }, tool_setup.enabled());
    };

    if (!stream) {
        GenOutcome o = run_generation(
            spec, nullptr, id,
            [st]() { return st == nullptr || !st->disconnected(); });
        std::string reasoning, answer;
        split_reasoning(o.text, p.thinking, &reasoning, &answer);
        toolparse::StreamParser parser = make_parser();
        parser.feed(tool_setup.choice.parser_prefix());
        parser.feed(answer);
        parser.finish();
        if (parser.has_partial_call() && o.reason != "length")
            http::fail(502, parser.error(), "model_output_error", "invalid_tool_call",
                       parser.error_details());

        json content = json::array();
        if (!reasoning.empty())
            content.push_back(json{{"type", "thinking"}, {"thinking", reasoning}, {"signature", ""}});
        if (has_visible_text(parser.content()))
            content.push_back(json{{"type", "text"}, {"text", parser.content()}});
        for (const auto& call : parser.calls())
            content.push_back(json{{"type", "tool_use"},
                                   {"id", call.id},
                                   {"name", call.name},
                                   {"input", tool_use_input(call.arguments)}});
        json m = message_object(
            std::move(content),
            stop_reason(o, parser.has_partial_call(), !parser.calls().empty()),
            stop_sequence(o), usage(&o));
        r->body = json_py::dumps(m, /*spaced=*/false);
        return;
    }

    // message_start is sent lazily (first ping or first block) so that errors
    // raised before the slot is granted still get a real HTTP status.
    r->sse = true;
    bool started = false;
    auto ensure_started = [&]() {
        if (started) return true;
        started = true;
        return send_anthropic_event(
            st, "message_start",
            json{{"type", "message_start"},
                 {"message", message_object(json::array(), nullptr, nullptr, usage(nullptr))}});
    };

    enum class Block { None, Thinking, Text, Tool };
    Block open_kind = Block::None;
    int open_index = -1;
    int next_index = 0;
    auto close_block = [&]() {
        if (open_kind == Block::None) return true;
        open_kind = Block::None;
        return send_anthropic_event(st, "content_block_stop",
                                    json{{"type", "content_block_stop"}, {"index", open_index}});
    };
    auto open_block = [&](Block kind, json block) {
        if (!ensure_started() || !close_block()) return false;
        open_kind = kind;
        open_index = next_index++;
        return send_anthropic_event(st, "content_block_start",
                                    json{{"type", "content_block_start"},
                                         {"index", open_index},
                                         {"content_block", std::move(block)}});
    };
    auto block_delta = [&](json delta) {
        return send_anthropic_event(st, "content_block_delta",
                                    json{{"type", "content_block_delta"},
                                         {"index", open_index},
                                         {"delta", std::move(delta)}});
    };
    auto emit_thinking = [&](const std::string& text) {
        if (text.empty()) return true;
        if (open_kind != Block::Thinking &&
            !open_block(Block::Thinking,
                        json{{"type", "thinking"}, {"thinking", ""}, {"signature", ""}}))
            return false;
        return block_delta(json{{"type", "thinking_delta"}, {"thinking", text}});
    };
    // Whitespace-only text (the newlines Qwen puts before <tool_call>) is held
    // back and only opens a text block once real text follows, matching the
    // non-streaming path, which drops whitespace-only text.
    std::string held_space;
    auto emit_text = [&](const std::string& text) {
        if (text.empty()) return true;
        if (open_kind != Block::Text) {
            if (!has_visible_text(text)) {
                held_space += text;
                return true;
            }
            if (!open_block(Block::Text, json{{"type", "text"}, {"text", ""}})) return false;
            std::string first = std::move(held_space);
            held_space.clear();
            return block_delta(json{{"type", "text_delta"}, {"text", first + text}});
        }
        return block_delta(json{{"type", "text_delta"}, {"text", text}});
    };
    std::string tool_arguments;
    auto dispatch_tool_events = [&](const std::vector<toolparse::Event>& events) {
        for (const auto& event : events) {
            if (event.type == toolparse::EventType::Content) {
                if (!emit_text(event.data)) return false;
            } else if (event.type == toolparse::EventType::CallStart) {
                held_space.clear();
                tool_arguments.clear();
                if (!open_block(Block::Tool, json{{"type", "tool_use"},
                                                  {"id", event.id},
                                                  {"name", event.name},
                                                  {"input", json::object()}}))
                    return false;
            } else if (event.type == toolparse::EventType::ArgumentsDelta) {
                if (event.data.empty()) continue;
                tool_arguments += event.data;
                if (!block_delta(json{{"type", "input_json_delta"}, {"partial_json", event.data}}))
                    return false;
            } else if (event.type == toolparse::EventType::CallEnd) {
                if (tool_arguments.empty() && !event.data.empty() &&
                    !block_delta(json{{"type", "input_json_delta"}, {"partial_json", event.data}}))
                    return false;
                if (!close_block()) return false;
            }
        }
        return true;
    };

    ThinkSplitter splitter(p.thinking);
    toolparse::StreamParser parser = make_parser();
    if (!dispatch_tool_events(parser.feed(tool_setup.choice.parser_prefix()))) return;
    auto on_wait = [&]() {
        if (st->disconnected()) return false;
        if (!ensure_started()) return false;
        return send_anthropic_event(st, "ping", json{{"type", "ping"}});
    };
    GenOutcome o = run_generation(
        spec,
        [&](const std::string& delta) {
            if (st->disconnected()) return false;
            if (!p.thinking) return dispatch_tool_events(parser.feed(delta));
            std::string reasoning_delta, content_delta;
            splitter.feed(delta, &reasoning_delta, &content_delta);
            if (!emit_thinking(reasoning_delta)) return false;
            return dispatch_tool_events(parser.feed(content_delta));
        },
        id, on_wait);
    if (o.client_gone) return;
    if (p.thinking) {
        std::string reasoning_delta, content_delta;
        splitter.finish(&reasoning_delta, &content_delta);
        if (!emit_thinking(reasoning_delta)) return;
        if (!dispatch_tool_events(parser.feed(content_delta))) return;
    }
    if (!dispatch_tool_events(parser.finish())) return;
        if (parser.has_partial_call() && o.reason != "length")
            http::fail(502, parser.error(), "model_output_error", "invalid_tool_call",
                       parser.error_details());

    // A tool block still open_kind here was cut off mid-call.
    const bool partial_call = parser.has_partial_call() || open_kind == Block::Tool;
    if (!ensure_started() || !close_block()) return;
    if (!send_anthropic_event(
            st, "message_delta",
            json{{"type", "message_delta"},
                 {"delta",
                  {{"stop_reason", stop_reason(o, partial_call, !parser.calls().empty())},
                   {"stop_sequence", stop_sequence(o)}}},
                 {"usage", usage(&o)}}))
        return;
    send_anthropic_event(st, "message_stop", json{{"type", "message_stop"}});
}

void messages_count_impl(const http::Request& q, http::Response* r, http::Stream*) {
    MessagesPrep p;
    prepare_messages_request(q, r, &p, /*count_only=*/true);
    r->body = json_py::dumps(
        json{{"input_tokens", static_cast<long long>(p.spec.ids.size())}}, /*spaced=*/false);
}

// Re-wraps errors in Anthropic's envelope. Once the SSE stream has started
// the status line is gone, so the error goes out as an `error` event.
void with_anthropic_errors(void (*impl)(const http::Request&, http::Response*, http::Stream*),
                           const http::Request& q, http::Response* r, http::Stream* st) {
    int status = 0;
    std::string message;
    try {
        impl(q, r, st);
        return;
    } catch (const http::Error& e) {
        status = e.status;
        message = e.message;
    } catch (const std::exception& e) {
        status = 500;
        message = std::string("internal error: ") + e.what();
    }
    if (st != nullptr && st->started()) {
        send_anthropic_event(st, "error", anthropic_error_json(status, message));
        return;  // r->sse is still set: the server closes the chunked body
    }
    r->sse = false;
    r->status = status;
    r->content_type = "application/json";
    r->body = json_py::dumps(anthropic_error_json(status, message), /*spaced=*/false);
}

// POST /v1/messages
void handle_messages(const http::Request& q, http::Response* r, http::Stream* st) {
    with_anthropic_errors(messages_impl, q, r, st);
}

// POST /v1/messages/count_tokens
void handle_messages_count(const http::Request& q, http::Response* r, http::Stream* st) {
    with_anthropic_errors(messages_count_impl, q, r, st);
}
