| config | set | tok/s mean +/- SD | vs plain (same set) | vs ref best | accept | out = plain | out = ref | what |
|---|---|---|---|---|---|---|---|---|
| e1-free-plain | free | 65.24 +/- 1.16 | 1.000x | 0.609x | - | self | - | free: plain bf16 (release build) |
| e1-free-untrim-n2 | free | 75.15 +/- 0.74 | 1.152x | 0.701x | 49.7% | 10/30 | - | free: MTP n-max 2 -bs, untrimmed (own build) |
| e1-free-trim8k-n2 | free | 107.15 +/- 0.89 | 1.642x | 1.000x | 48.3% | 10/30 | - | free: MTP n-max 2 -bs, trimmed 8k (own build) |
| e1-free-trim32k-n2 | free | 104.15 +/- 0.07 | 1.596x | 0.972x | 49.0% | 10/30 | - | free: MTP n-max 2 -bs, trimmed 32k (own build) |
| e1-code-plain | code | 66.48 +/- 2.22 | 1.000x | 0.620x | - | self | - | code: plain bf16 (release build) |
| e1-code-untrim-n2 | code | 93.72 +/- 0.27 | 1.410x | 0.875x | 74.1% | 15/20 | - | code: MTP n-max 2 -bs, untrimmed (own build) |
| e1-t07-plain | free | 67.30 +/- 0.35 | 1.032x | 0.628x | - | 0/30 | - | free T=0.7 top-p 0.9: plain bf16 (release, CPU sampling) |
| e1-t07-plain-bs | free | 70.18 +/- 0.45 | 1.076x | 0.655x | - | 0/30 | - | free T=0.7 top-p 0.9: plain bf16 with -bs (own build) |
| e1-t07-untrim-n2-greedy | free | 67.39 +/- 0.46 | 1.033x | 0.629x | 45.9% | 0/30 | - | free T=0.7: MTP n-max 2 -bs untrimmed, draft sampling greedy |
| e1-t07-untrim-n2-probabilistic | free | 77.65 +/- 0.10 | 1.190x | 0.725x | 52.3% | 0/30 | - | free T=0.7: MTP n-max 2 -bs untrimmed, draft sampling probabilistic |
| e1-code-trim8k-n2 | ERROR timed out |
| e1-code-trim32k-n2 | code | 124.89 +/- 0.12 | 1.879x | 1.166x | 68.5% | 15/20 | - | code: MTP n-max 2 -bs, trimmed 32k (own build) |
| e1-code-trim8k-n1 | code | 92.14 +/- 0.12 | 1.386x | 0.860x | 54.6% | 15/20 | - | code: MTP n-max 1 -bs, trimmed 8k |
| e1-lang-plain | lang | 68.48 +/- 0.10 | 1.000x | 0.639x | - | self | - | lang: plain bf16 (release build) |
| e1-lang-untrim-n2 | lang | 68.19 +/- 0.10 | 0.996x | 0.636x | 39.1% | 5/20 | - | lang: MTP n-max 2 -bs, untrimmed (own build) |
| e1-lang-trim8k-n2 | lang | 69.02 +/- 0.04 | 1.008x | 0.644x | 12.2% | 5/20 | - | lang: MTP n-max 2 -bs, trimmed 8k (own build) |
| e1-lang-trim32k-n2 | lang | 70.88 +/- 0.06 | 1.035x | 0.662x | 16.9% | 5/20 | - | lang: MTP n-max 2 -bs, trimmed 32k (own build) |
| e1-lang-trim8k-n1 | lang | 71.54 +/- 0.05 | 1.045x | 0.668x | 19.8% | 5/20 | - | lang: MTP n-max 1 -bs, trimmed 8k |
| e1-t07-trim8k-n2-greedy | free | 104.78 +/- 0.04 | 1.606x | 0.978x | 45.0% | 0/30 | - | free T=0.7: MTP n-max 2 -bs trimmed 8k, draft sampling greedy |
| e1-t07-trim8k-n2-probabilistic | free | 105.71 +/- 0.10 | 1.620x | 0.987x | 46.1% | 0/30 | - | free T=0.7: MTP n-max 2 -bs trimmed 8k, draft sampling probabilistic |
| e1-t07-trim8k-n1-greedy | free | 95.41 +/- 0.09 | 1.462x | 0.890x | 60.1% | 0/30 | - | free T=0.7: MTP n-max 1 -bs trimmed 8k, draft sampling greedy |
| e1-free-plain-end | free | 68.34 +/- 0.14 | 1.048x | 0.638x | - | 30/30 | - | free: plain bf16 (release build) |
