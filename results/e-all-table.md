| config | set | tok/s mean +/- SD | vs plain (same set) | vs ref best | accept | out = plain | out = ref | what |
|---|---|---|---|---|---|---|---|---|
| e1-free-plain | free | 65.24 +/- 1.16 | 1.002x | 0.609x | - | 18/18 | - | free: plain bf16 (release build) |
| e1-free-untrim-n2 | free | 75.15 +/- 0.74 | 1.154x | 0.701x | 49.7% | 6/18 | - | free: MTP n-max 2 -bs, untrimmed (own build) |
| e1-free-trim8k-n2 | free | 107.15 +/- 0.89 | 1.646x | 1.000x | 48.3% | 6/18 | - | free: MTP n-max 2 -bs, trimmed 8k (own build) |
| e1-free-trim32k-n2 | free | 104.15 +/- 0.07 | 1.600x | 0.972x | 49.0% | 6/18 | - | free: MTP n-max 2 -bs, trimmed 32k (own build) |
| e1-code-plain | code | 66.48 +/- 2.22 | 1.000x | 0.532x | - | self | - | code: plain bf16 (release build) |
| e1-code-untrim-n2 | code | 93.72 +/- 0.27 | 1.410x | 0.750x | 74.1% | 15/20 | - | code: MTP n-max 2 -bs, untrimmed (own build) |
| e1-t07-plain | free | 67.30 +/- 0.35 | 1.034x | 0.628x | - | 0/18 | - | free T=0.7 top-p 0.9: plain bf16 (release, CPU sampling) |
| e1-t07-plain-bs | free | 70.18 +/- 0.45 | 1.078x | 0.655x | - | 0/18 | - | free T=0.7 top-p 0.9: plain bf16 with -bs (own build) |
| e1-t07-untrim-n2-greedy | free | 67.39 +/- 0.46 | 1.035x | 0.629x | 45.9% | 0/18 | - | free T=0.7: MTP n-max 2 -bs untrimmed, draft sampling greedy |
| e1-t07-untrim-n2-probabilistic | free | 77.65 +/- 0.10 | 1.193x | 0.725x | 52.3% | 0/18 | - | free T=0.7: MTP n-max 2 -bs untrimmed, draft sampling probabilistic |
| e1-code-trim8k-n2 | code | 98.52 +/- 0.48 | 1.482x | 0.789x | 40.8% | 15/20 | - | code: MTP n-max 2 -bs, trimmed 8k (own build) |
| e1-code-trim32k-n2 | code | 124.89 +/- 0.12 | 1.879x | 1.000x | 68.5% | 15/20 | - | code: MTP n-max 2 -bs, trimmed 32k (own build) |
| e1-code-trim8k-n1 | code | 92.14 +/- 0.12 | 1.386x | 0.738x | 54.6% | 15/20 | - | code: MTP n-max 1 -bs, trimmed 8k |
| e1-lang-plain | lang | 68.48 +/- 0.10 | 1.000x | 1.004x | - | self | - | lang: plain bf16 (release build) |
| e1-lang-untrim-n2 | lang | 68.19 +/- 0.10 | 0.996x | 1.000x | 39.1% | 5/20 | - | lang: MTP n-max 2 -bs, untrimmed (own build) |
| e1-lang-trim8k-n2 | lang | 69.02 +/- 0.04 | 1.008x | 1.012x | 12.2% | 5/20 | - | lang: MTP n-max 2 -bs, trimmed 8k (own build) |
| e1-lang-trim32k-n2 | lang | 70.88 +/- 0.06 | 1.035x | 1.039x | 16.9% | 5/20 | - | lang: MTP n-max 2 -bs, trimmed 32k (own build) |
| e1-lang-trim8k-n1 | lang | 71.54 +/- 0.05 | 1.045x | 1.049x | 19.8% | 5/20 | - | lang: MTP n-max 1 -bs, trimmed 8k |
| e1-t07-trim8k-n2-greedy | free | 104.78 +/- 0.04 | 1.609x | 0.978x | 45.0% | 0/18 | - | free T=0.7: MTP n-max 2 -bs trimmed 8k, draft sampling greedy |
| e1-t07-trim8k-n2-probabilistic | free | 105.71 +/- 0.10 | 1.624x | 0.987x | 46.1% | 0/18 | - | free T=0.7: MTP n-max 2 -bs trimmed 8k, draft sampling probabilistic |
| e1-t07-trim8k-n1-greedy | free | 95.41 +/- 0.09 | 1.466x | 0.890x | 60.1% | 0/18 | - | free T=0.7: MTP n-max 1 -bs trimmed 8k, draft sampling greedy |
| e1-free-plain-end | free | 68.34 +/- 0.14 | 1.050x | 0.638x | - | 18/18 | - | free: plain bf16 (release build) |
| e4-lang-fix4k-dyn4k | lang | 86.59 +/- 0.21 | 1.264x | 1.270x | 32.7% | 3/12 | 12/12 | lang: 4096 fixed + 4096 context/script rows (refresh 16 tok), n-max 2 |
| e4-code-fix4k-dyn4k | code | 114.04 +/- 0.27 | 1.715x | 0.913x | 58.1% | 9/12 | 12/12 | code: 4096 fixed + 4096 context/script rows (refresh 16 tok), n-max 2 |
| e4-free-fix4k-dyn4k | free | 105.60 +/- 0.49 | 1.622x | 0.986x | 48.5% | 6/18 | 18/18 | free: 4096 fixed + 4096 context/script rows (refresh 16 tok), n-max 2 |
| e4-lang-fix2k-dyn6k | lang | 85.54 +/- 0.65 | 1.249x | 1.255x | 33.3% | 3/12 | 12/12 | lang: 2048 fixed + 6144 context/script rows, n-max 2 |
| e2-free-base | free | 108.99 +/- 0.40 | 1.674x | 1.017x | 48.3% | 6/18 | 18/18 | free: trim8k n-max 2 (control on the E build) |
| e2-free-adapt4 | free | 92.99 +/- 1.03 | 1.428x | 0.868x | 42.4% | 3/18 | 9/18 | free: trim8k history-adaptive n in [1,4] |
| e2-free-n3 | free | 108.59 +/- 0.04 | 1.668x | 1.013x | 35.9% | 6/18 | 18/18 | free: trim8k n-max 3 fixed |
| e2-free-n3-p0.5 | free | 84.99 +/- 0.60 | 1.305x | 0.793x | 70.4% | 6/18 | 9/18 | free: trim8k n-max 3, --spec-draft-p-min 0.5 |
| e2-free-n4-p0.5 | free | 82.36 +/- 0.34 | 1.265x | 0.769x | 69.2% | 3/18 | 9/18 | free: trim8k n-max 4, p-min 0.5 |
| e2-free-adapt3 | free | 93.44 +/- 1.16 | 1.435x | 0.872x | 45.4% | 6/18 | 18/18 | free: trim8k history-adaptive n in [1,3] |
| e3-rep-mtp | rep | 146.74 +/- 0.50 | 2.171x | 1.000x | 84.4% | 18/18 | 18/18 | rep: trim8k MTP n-max 2 (control) |
| e3-rep-ngs-mtp-n4m8 | rep | 201.10 +/- 1.00 | 2.975x | 1.370x | 58.5% | 18/18 | 18/18 | rep: ngram-simple(N4,M8) then MTP trim8k, n-max 8 |
| e3-rep-ngmod-mtp | rep | 592.24 +/- 30.38 | 8.760x | 4.036x | 77.0% | 18/18 | 18/18 | rep: ngram-mod then MTP trim8k |
| e3-rep-ngs-only | rep | 109.75 +/- 2.12 | 1.623x | 0.748x | 67.4% | 18/18 | 18/18 | rep: ngram-simple(N4,M8) only, n-max 8 |
| e3-free-mtp | free | 107.78 +/- 0.70 | 1.655x | 1.006x | 48.3% | 6/18 | 18/18 | free: trim8k MTP n-max 2 (control) |
| e3-free-ngs-mtp-n4m8 | free | 89.09 +/- 0.29 | 1.368x | 0.831x | 39.1% | 3/18 | 9/18 | free: ngram-simple(N4,M8) then MTP trim8k, n-max 8 |
| e5-plain-q8head | free | 77.09 +/- 0.80 | 1.184x | 0.719x | - | 3/18 | 3/18 | free: plain, bf16 body + Q8_0 token_embd/output head (speed ceiling, lossy) |
| e5-trim8k-q8head | free | 121.33 +/- 0.46 | 1.864x | 1.132x | 47.5% | 6/18 | 0/18 | free: trim8k MTP n2 -bs, bf16 body + Q8_0 head (speed ceiling, lossy) |
| e3-rep-plain | rep | 67.61 +/- 0.18 | 1.000x | 0.461x | - | self | - | rep: plain bf16 (release build) |
| e3-free-ngmod-mtp | free | 480.79 +/- 257.70 | 7.385x | 4.487x | 80.5% | 6/18 | 12/18 | free: ngram-mod then MTP trim8k |
| e1-free-plain-e234 | free | 65.10 +/- 0.49 | 1.000x | 0.608x | - | self | - | free: plain bf16 (release build) |
| e3-free-ngmod-mtp-cold-c1 | free | 84.79 +/- 0.00 | 1.302x | 0.791x | 39.6% | 0/6 | 4/6 | free: ngram-mod then MTP trim8k [COLD: no warm-up, 1 pass, fresh server] |
| e3-rep-ngmod-mtp-cold-c1 | rep | 190.21 +/- 0.00 | 2.813x | 1.296x | 57.7% | 6/6 | 6/6 | rep: ngram-mod then MTP trim8k [COLD: no warm-up, 1 pass, fresh server] |
| e3-free-ngmod-mtp-cold-c2 | free | 86.77 +/- 0.00 | 1.333x | 0.810x | 39.6% | 0/6 | 4/6 | free: ngram-mod then MTP trim8k [COLD: no warm-up, 1 pass, fresh server] |
| e3-rep-ngmod-mtp-cold-c2 | rep | 189.07 +/- 0.00 | 2.797x | 1.288x | 57.7% | 6/6 | 6/6 | rep: ngram-mod then MTP trim8k [COLD: no warm-up, 1 pass, fresh server] |
| e3-free-ngmod-mtp-cold-c3 | free | 88.27 +/- 0.00 | 1.356x | 0.824x | 39.6% | 0/6 | 4/6 | free: ngram-mod then MTP trim8k [COLD: no warm-up, 1 pass, fresh server] |
| e3-rep-ngmod-mtp-cold-c3 | rep | 195.38 +/- 0.00 | 2.890x | 1.331x | 57.7% | 6/6 | 6/6 | rep: ngram-mod then MTP trim8k [COLD: no warm-up, 1 pass, fresh server] |
