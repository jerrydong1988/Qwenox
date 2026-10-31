// From official 91b111aeb4bf85723433a821f200398ca8d91cc2.
// POST /v1/tokenize — text -> token ids. `text` may also be spelled `prompt`
// (vLLM's /tokenize convention). Offsets, when requested, are UTF-8 byte
// ranges into the input text, not character indexes.
void handle_tokenize(const http::Request& q, http::Response* r, http::Stream*) {
    if (g_tok.vocab_size() == 0)
        http::fail(503, "tokenizer not loaded (" + g_cfg.tokenizer_dir +
                        "); check TOKENIZER_DIR in service.conf");
    const json body = parse_body(q);
    const char* key = body.contains("text")      ? "text"
                      : body.contains("prompt")  ? "prompt"
                                                 : nullptr;
    if (key == nullptr) http::fail(400, "text is required");
    const json& tv = body[key];
    if (!tv.is_string()) http::fail(400, std::string(key) + " must be a string");
    const bool want_offsets = bool_field(body, "return_offsets", false);

    const auto spans = g_tok.encode_with_offsets(tv.get_ref<const std::string&>());
    json ids = json::array();
    ids.get_ref<json::array_t&>().reserve(spans.size());
    json offsets = json::array();
    if (want_offsets) offsets.get_ref<json::array_t&>().reserve(spans.size());
    for (const auto& t : spans) {
        ids.push_back(t.id);
        if (want_offsets) offsets.push_back(json::array({t.start, t.end}));
    }
    json j;
    j["tokens"] = std::move(ids);
    j["count"] = spans.size();
    if (want_offsets) j["offsets"] = std::move(offsets);
    r->set("Cache-Control", "no-store");
    r->body = json_py::dumps(j, /*spaced=*/false);
}

// POST /v1/detokenize — token ids -> text. `tokens` may also be spelled `ids`.
void handle_detokenize(const http::Request& q, http::Response* r,
                       http::Stream*) {
    if (g_tok.vocab_size() == 0)
        http::fail(503, "tokenizer not loaded (" + g_cfg.tokenizer_dir +
                        "); check TOKENIZER_DIR in service.conf");
    const json body = parse_body(q);
    const json* arr = body.contains("tokens") ? &body["tokens"]
                      : body.contains("ids")  ? &body["ids"]
                                              : nullptr;
    if (arr == nullptr) http::fail(400, "tokens is required");
    if (!arr->is_array()) http::fail(400, "tokens must be an array of integers");
    const bool skip_special = bool_field(body, "skip_special_tokens", true);

    std::vector<int> ids;
    ids.reserve(arr->size());
    for (const auto& v : *arr) {
        if (!v.is_number_integer() && !v.is_number_unsigned())
            http::fail(400, "tokens must be an array of integers");
        const long long id = v.is_number_unsigned()
                                 ? (long long)v.get<unsigned long long>()
                                 : v.get<long long>();
        if (id < 0 || id >= g_tok.vocab_size())
            http::fail(400, "token id out of range: " + std::to_string(id));
        ids.push_back((int)id);
    }
    json j;
    j["text"] = g_tok.decode(ids, skip_special);
    r->set("Cache-Control", "no-store");
    r->body = json_py::dumps(j, /*spaced=*/false);
}
