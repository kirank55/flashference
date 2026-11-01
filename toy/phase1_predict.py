"""Phase 1 predictions, derived from the models' config.json shapes (no weights loaded, no torch).

Prints the byte and parameter counts that predictions P-011 (1.3) and P-012 (1.4) state, so the
arithmetic behind them can be re-run. Reads config.json from the local HF cache only.

  python toy/phase1_predict.py
"""
import json

from huggingface_hub import hf_hub_download


def config(repo):
    return json.load(open(hf_hub_download(repo, "config.json", local_files_only=True)))


def smollm2(repo):
    c = config(repo)
    h, i, L, V = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"], c["vocab_size"]
    hd = h // c["num_attention_heads"]
    kv = c["num_key_value_heads"] * hd
    attn = 2 * h * h + 2 * h * kv                       # q and o: h x h; k and v: h x kv
    mlp = 3 * h * i                                     # gate, up, down
    norms = 2 * h                                       # input and post-attention RMSNorm
    emb = V * h                                         # tied: the LM head is this same matrix
    total = emb + L * (attn + mlp + norms) + h          # + final norm
    print(f"{repo}: h={h} i={i} L={L} V={V} head_dim={hd} kv_dim={kv} tied={c['tie_word_embeddings']}")
    print(f"  params {total:,}; bf16 bytes {2 * total:,}; embedding {emb:,} = {100 * emb / total:.3f} % of all")


def qwen35(repo="Qwen/Qwen3.5-0.8B"):
    t = config(repo)["text_config"]
    h, i, V = t["hidden_size"], t["intermediate_size"], t["vocab_size"]
    heads, kvh, hd = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    gate = 2 if t["attn_output_gate"] else 1            # the gate doubles q_proj's output
    full = [n for n, x in enumerate(t["layer_types"]) if x == "full_attention"]
    mlp = 3 * h * i
    # weights stored in bf16, and the small ones assumed f32 (norms, ssm_a, ssm_dt, conv1d)
    att_bf = h * gate * heads * hd + 2 * h * kvh * hd + heads * hd * h
    att_f32 = 2 * hd + 2 * h                            # q_norm, k_norm, attn_norm, post_attention_norm
    kd, vd = t["linear_num_key_heads"] * t["linear_key_head_dim"], t["linear_num_value_heads"] * t["linear_value_head_dim"]
    nv, kc = t["linear_num_value_heads"], t["linear_conv_kernel_dim"]
    rec_bf = h * (2 * kd + 2 * vd) + h * 2 * nv + vd * h            # in_proj q,k,v,z; b,a; out_proj
    rec_f32 = (2 * kd + vd) * kc + t["linear_value_head_dim"] + 2 * nv + 2 * h   # conv1d, ssm_norm, a, dt, 2 norms
    mtp_bf = att_bf + mlp + 2 * h * h                   # one full-attention layer + eh_proj (2h -> h)
    mtp_f32 = att_f32 + 3 * h                           # + enorm, hnorm, shared_head_norm
    nfull, nrec = len(full), t["num_hidden_layers"] - len(full)
    b = lambda bf, f: 2 * bf + 4 * f
    emb = 2 * V * h
    total = emb + 4 * h + nfull * b(att_bf + mlp, att_f32) + nrec * b(rec_bf + mlp, rec_f32) \
        + t["mtp_num_hidden_layers"] * b(mtp_bf, mtp_f32)
    params = V * h + h + nfull * (att_bf + mlp + att_f32) + nrec * (rec_bf + mlp + rec_f32) \
        + t["mtp_num_hidden_layers"] * (mtp_bf + mtp_f32)
    print(f"{repo}: h={h} i={i} V={V} layers={t['num_hidden_layers']} full_attention={full} mtp_layers={t['mtp_num_hidden_layers']} tied={t['tie_word_embeddings']}")
    print(f"  token_embd {emb:,} bytes = {100 * emb / total:.3f} % of tensor bytes")
    print(f"  attention block {b(att_bf + mlp, att_f32):,} bytes x {nfull}; recurrent block {b(rec_bf + mlp, rec_f32):,} bytes x {nrec}; "
          f"MTP block {b(mtp_bf, mtp_f32):,} bytes")
    print(f"  tensor bytes {total:,}; params {params:,}")


if __name__ == "__main__":
    smollm2("HuggingFaceTB/SmolLM2-360M-Instruct")
    smollm2("HuggingFaceTB/SmolLM2-135M-Instruct")
    qwen35()
