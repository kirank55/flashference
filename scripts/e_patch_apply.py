"""Apply the E2/E4 edits (adaptive draft length + context-adaptive draft vocabulary) on top of the B4-patched llama-src (b11884)."""
import sys
R = r"C:\Users\kiran\code\p\infer\slmspeed\llama-src"
def edit(path, pairs, region=None):
    p = R + "\\" + path; t = open(p, encoding="utf-8", newline="").read(); nl = "\r\n" if "\r\n" in t else "\n"; t = t.replace("\r\n", "\n")
    a, b = 0, len(t)
    if region:
        a = t.index(region[0]); b = t.index(region[1])
    mid = t[a:b]
    for old, new in pairs:
        assert mid.count(old) == 1, (path, mid.count(old), old[:70])
        mid = mid.replace(old, new)
    t = t[:a] + mid + t[b:]
    open(p, "w", encoding="utf-8", newline="").write(t.replace("\n", nl))

if "--first" in sys.argv: edit("src/llama-ext.h", [("LLAMA_API const int32_t * llama_model_mtp_trim_map(const struct llama_model * model, int32_t * n);",
"""LLAMA_API const int32_t * llama_model_mtp_trim_map(const struct llama_model * model, int32_t * n);
// E4: refresh the dynamic part of the trimmed draft vocabulary from the token history (no-op unless LLAMA_MTP_TRIM_FIXED < K)
LLAMA_API void llama_model_mtp_trim_update(const struct llama_model * model, const int32_t * hist, size_t n);""")])

if "--first" in sys.argv: edit("src/llama-model.h", [("    std::vector<int32_t> mtp_trim_map;            // subset index -> token id\n",
"""    std::vector<int32_t> mtp_trim_map;            // subset index -> token id
    // E4: dynamic part of the trimmed vocabulary (rows [mtp_trim_fixed, K) are rewritten from the context)
    int32_t              mtp_trim_fixed = 0;
    size_t               mtp_trim_row_bytes = 0;
    std::vector<uint8_t> mtp_head_host;           // host copy of the full head (only kept when dynamic)
    std::vector<int32_t> mtp_trim_default;        // initial map (rank order)
    std::vector<char>    mtp_trim_fixed_flag;     // [n_vocab] token is in the fixed part
    std::vector<uint8_t> mtp_tok_script;          // [n_vocab] script class of each token (0 = latin/other)
""")])

if "--first" in sys.argv: edit("src/llama-model.cpp", [
("// B4: gather the K most frequent rows of the (tied) LM head",
"""// E4: coarse script class of a token piece (1 CJK, 2 Indic, 3 Cyrillic, 4 Arabic/Hebrew, 5 Hangul, 6 Thai, 7 Greek, 0 other), also from truncated UTF-8
static uint8_t mtp_script_class(const std::string & s) {
    int cnt[8] = {0};
    auto cls = [](uint32_t cp) -> int {
        if (cp >= 0x900 && cp < 0xE00) return 2;
        if (cp >= 0xE00 && cp < 0xE80) return 6;
        if (cp >= 0x1100 && cp < 0x1200) return 5;
        if ((cp >= 0x2E80 && cp < 0xA000) || (cp >= 0xF900 && cp < 0xFB00) || (cp >= 0xFF00 && cp < 0xFFF0)) return 1;
        if (cp >= 0xAC00 && cp < 0xD800) return 5;
        return 0;
    };
    size_t i = 0, n = s.size();
    while (i < n) {
        const unsigned char b = (unsigned char) s[i];
        if (b < 0x80) { i++; continue; }
        if ((b & 0xE0) == 0xC0 && i + 1 < n) {
            const uint32_t cp = ((b & 0x1Fu) << 6) | ((unsigned char) s[i + 1] & 0x3Fu);
            if (cp >= 0x370 && cp < 0x400) cnt[7]++; else if (cp >= 0x400 && cp < 0x530) cnt[3]++; else if (cp >= 0x590 && cp < 0x700) cnt[4]++;
            i += 2; continue;
        }
        if ((b & 0xF0) == 0xE0) {
            if (i + 2 < n) {
                const uint32_t cp = ((b & 0xFu) << 12) | (((unsigned char) s[i + 1] & 0x3Fu) << 6) | ((unsigned char) s[i + 2] & 0x3Fu);
                cnt[cls(cp)]++; i += 3; continue;
            }
            if (i + 1 < n) { const uint32_t cp = ((b & 0xFu) << 12) | (((unsigned char) s[i + 1] & 0x3Fu) << 6); cnt[cls(cp)]++; i += 2; continue; }
            if (b >= 0xE4 && b <= 0xE9) cnt[1]++; else if (b >= 0xEA && b <= 0xED) cnt[5]++;
            i++; continue;
        }
        i++;
    }
    int best = 0, bc = 0;
    for (int c = 1; c < 8; ++c) { if (cnt[c] > bc) { bc = cnt[c]; best = c; } }
    return (uint8_t) best;
}

// B4: gather the K most frequent rows of the (tied) LM head"""),
("    mtp_head_sub  = t_sub;\n    mtp_head_mask = t_mask;\n    mtp_trim_map  = std::move(ids);\n",
"""    // E4: LLAMA_MTP_TRIM_FIXED=Kf (< K): rows [0,Kf) stay fixed, rows [Kf,K) are rewritten from the context by llama_model_mtp_trim_update
    {
        const char * fs = getenv("LLAMA_MTP_TRIM_FIXED");
        int64_t Kf = fs ? strtoll(fs, nullptr, 10) : K;
        if (Kf <= 0 || Kf > K) { Kf = K; }
        mtp_trim_fixed = (int32_t) Kf;
        if (Kf < K) {
            mtp_head_host = std::move(all);
            mtp_trim_row_bytes = row_bytes;
            mtp_trim_default = ids;
            mtp_trim_fixed_flag.assign(n_vocab, 0);
            for (int64_t k = 0; k < Kf; ++k) { mtp_trim_fixed_flag[ids[k]] = 1; }
            mtp_tok_script.assign(n_vocab, 0);
            for (int64_t t = 0; t < n_vocab; ++t) { mtp_tok_script[t] = mtp_script_class(vocab.token_to_piece((llama_token) t)); }
            LLAMA_LOG_INFO("%s: MTP trim: dynamic vocabulary, %d fixed + %d context rows\\n", __func__, (int) Kf, (int) (K - Kf));
        }
    }

    mtp_head_sub  = t_sub;
    mtp_head_mask = t_mask;
    mtp_trim_map  = std::move(ids);
"""),
("const int32_t * llama_model_mtp_trim_map(const struct llama_model * model, int32_t * n) {",
"""void llama_model_mtp_trim_update(const struct llama_model * model_c, const int32_t * hist, size_t n) {
    llama_model * m = const_cast<llama_model *>(model_c);
    if (m->mtp_head_host.empty() || !m->mtp_head_sub || m->mtp_trim_map.empty()) { return; }
    const int32_t K = (int32_t) m->mtp_trim_map.size(), Kf = m->mtp_trim_fixed, Kd = K - Kf;
    const int64_t n_vocab = (int64_t) m->mtp_trim_fixed_flag.size();
    std::vector<int32_t> dyn; dyn.reserve(Kd);
    std::vector<char> seen(n_vocab, 0);
    int cnt[8] = {0}; int tot = 0;
    // 1. context tokens, most recent first
    for (size_t i = n; i-- > 0;) {
        const int32_t t = hist[i];
        if (t < 0 || t >= n_vocab) { continue; }
        if (tot < 256) { cnt[m->mtp_tok_script[t]]++; tot++; }
        if ((int32_t) dyn.size() < Kd && !m->mtp_trim_fixed_flag[t] && !seen[t]) { seen[t] = 1; dyn.push_back(t); }
        if ((int32_t) dyn.size() >= Kd && tot >= 256) { break; }
    }
    // 2. all tokens of the non-latin scripts that dominate the context (id order ~ BPE merge order ~ frequency), half of the remaining room each
    for (int c = 1; c < 8; ++c) {
        if (tot > 0 && cnt[c] * 5 >= tot) {
            const int32_t room = Kd - (int32_t) dyn.size();
            int32_t add = 0;
            for (int64_t t = 0; t < n_vocab && add < room; ++t) {
                if (m->mtp_tok_script[t] == c && !m->mtp_trim_fixed_flag[t] && !seen[t]) { seen[t] = 1; dyn.push_back((int32_t) t); add++; }
            }
        }
    }
    // 3. default (rank order) fill
    for (int32_t k = Kf; k < K && (int32_t) dyn.size() < Kd; ++k) {
        const int32_t t = m->mtp_trim_default[k];
        if (!seen[t]) { seen[t] = 1; dyn.push_back(t); }
    }
    while ((int32_t) dyn.size() < Kd) { dyn.push_back(dyn.empty() ? m->mtp_trim_default[Kf] : dyn.back()); }
    bool same = true;
    for (int32_t k = 0; k < Kd; ++k) { if (m->mtp_trim_map[Kf + k] != dyn[k]) { same = false; break; } }
    if (same) { return; }
    std::vector<uint8_t> rows((size_t) Kd * m->mtp_trim_row_bytes);
    for (int32_t k = 0; k < Kd; ++k) {
        memcpy(rows.data() + (size_t) k * m->mtp_trim_row_bytes, m->mtp_head_host.data() + (size_t) dyn[k] * m->mtp_trim_row_bytes, m->mtp_trim_row_bytes);
        m->mtp_trim_map[Kf + k] = dyn[k];
    }
    ggml_backend_tensor_set(m->mtp_head_sub, rows.data(), (size_t) Kf * m->mtp_trim_row_bytes, rows.size());
}

const int32_t * llama_model_mtp_trim_map(const struct llama_model * model, int32_t * n) {"""),
])

edit("common/speculative.cpp", [
("        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));\n\n        i_last.assign(n_seq, -1);",
"""        // E2/E4 knobs (env): LLAMA_MTP_ADAPT=1 history-adaptive draft length in [LO, n_max]; LLAMA_MTP_DYN=<period> refresh the dynamic draft vocabulary every <period> tokens
        {
            const char * a = getenv("LLAMA_MTP_ADAPT");
            e_adapt = a && atoi(a) > 0;
            const char * lo = getenv("LLAMA_MTP_ADAPT_LO"); e_adapt_lo = lo ? std::max(1, atoi(lo)) : 1;
            const char * ini = getenv("LLAMA_MTP_ADAPT_INIT");
            e_cur_n.assign(n_seq, ini ? atoi(ini) : std::min(2, this->params.n_max));
            e_last_dr.assign(n_seq, 0);
            const char * dy = getenv("LLAMA_MTP_DYN"); e_dyn_period = dy ? atoi(dy) : 0;
            e_dyn_last.assign(n_seq, (size_t) -1 / 2);
        }

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_last.assign(n_seq, -1);"""),
("    std::vector<int>                i_last;\n    std::vector<std::vector<float>> chain_h;\n",
"""    std::vector<int>                i_last;
    std::vector<std::vector<float>> chain_h;

    // E2/E4 (env driven, see ctor)
    bool                e_adapt = false;
    int                 e_adapt_lo = 1;
    std::vector<int>    e_cur_n, e_last_dr;
    int                 e_dyn_period = 0;
    std::vector<size_t> e_dyn_last;
"""),
("        batch.clear();\n\n        // keep track of which sequences are still drafting\n",
"""        batch.clear();

        // E4: refresh the dynamic part of the trimmed draft vocabulary from the history
        if (e_dyn_period > 0 && llama_model_mtp_trim_map(llama_get_model(ctx_dft), nullptr)) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                auto & dp = dparams[seq_id];
                if (!dp.drafting || !dp.prompt) { continue; }
                const size_t L = dp.prompt->size();
                if (L < e_dyn_last[seq_id] || L >= e_dyn_last[seq_id] + (size_t) e_dyn_period) {
                    llama_model_mtp_trim_update(llama_get_model(ctx_dft), dp.prompt->data(), L);
                    e_dyn_last[seq_id] = L;
                }
            }
        }

        // keep track of which sequences are still drafting
"""),
("            drafting[seq_id] = true;\n            // greedy drafting leaves no candidates behind",
 "            drafting[seq_id] = true;\n            e_last_dr[seq_id] = 0;\n            // greedy drafting leaves no candidates behind"),
("                result.push_back(id);\n\n                if (dp.result_q) {\n                    dp.result_q->emplace_back(cur_p->data, cur_p->data + cur_p->size);\n                    if (trim)",
 "                result.push_back(id);\n                e_last_dr[seq_id] = (int) result.size();\n\n                if (dp.result_q) {\n                    dp.result_q->emplace_back(cur_p->data, cur_p->data + cur_p->size);\n                    if (trim)"),
("                if (params.n_max <= (int) result.size()) {\n                    drafting[seq_id] = false;\n                    n_drafting--;\n                    continue;\n                }\n\n                if (chain_heads) {\n                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546",
 "                if ((e_adapt ? std::min(e_cur_n[seq_id], params.n_max) : params.n_max) <= (int) result.size()) {\n                    drafting[seq_id] = false;\n                    n_drafting--;\n                    continue;\n                }\n\n                if (chain_heads) {\n                    // ref: https://github.com/ggml-org/llama.cpp/pull/24340#discussion_r3448031546"),
("    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {\n        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {\n            return;\n        }\n\n        const int32_t n_rows = verify_h_rows[seq_id];",
 "    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {\n        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {\n            return;\n        }\n\n        if (e_adapt && e_last_dr[seq_id] > 0) {\n            const int d = e_last_dr[seq_id];\n            if ((int) n_accepted >= d) { e_cur_n[seq_id] = std::min(e_cur_n[seq_id] + 1, params.n_max); }\n            else if (n_accepted == 0) { e_cur_n[seq_id] = std::max(e_cur_n[seq_id] - 1, e_adapt_lo); }\n        }\n\n        const int32_t n_rows = verify_h_rows[seq_id];"),
], region=("struct common_speculative_impl_draft_mtp", "struct common_speculative_impl_ngram_simple"))
print("edited ok")
