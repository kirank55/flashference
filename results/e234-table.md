| config | set | tok/s mean +/- SD | vs plain (same set) | vs ref best | accept | out = plain | out = ref | what |
|---|---|---|---|---|---|---|---|---|
| e1-code-trim8k-n2 | code | 98.52 +/- 0.48 | 1.482x | 0.789x | 40.8% | 15/20 | - | code: MTP n-max 2 -bs, trimmed 8k (own build) |
| e4-lang-fix4k-dyn4k | lang | 86.59 +/- 0.21 | 1.264x | 1.270x | 32.7% | 3/12 | 12/12 | lang: 4096 fixed + 4096 context/script rows (refresh 16 tok), n-max 2 |
| e4-lang-fix2k-dyn6k | lang | 85.54 +/- 0.65 | 1.249x | 1.255x | 33.3% | 3/12 | 12/12 | lang: 2048 fixed + 6144 context/script rows, n-max 2 |
| e4-code-fix4k-dyn4k | code | 114.04 +/- 0.27 | 1.715x | 0.913x | 58.1% | 9/12 | 12/12 | code: 4096 fixed + 4096 context/script rows (refresh 16 tok), n-max 2 |
| e4-free-fix4k-dyn4k | free | 105.60 +/- 0.49 | 1.545x | 0.986x | 48.5% | 6/18 | 18/18 | free: 4096 fixed + 4096 context/script rows (refresh 16 tok), n-max 2 |
| e2-free-base | free | 108.99 +/- 0.40 | 1.595x | 1.017x | 48.3% | 6/18 | 18/18 | free: trim8k n-max 2 (control on the E build) |
| e2-free-n3 | free | 108.59 +/- 0.04 | 1.589x | 1.013x | 35.9% | 6/18 | 18/18 | free: trim8k n-max 3 fixed |
| e2-free-n3-p0.5 | free | 84.99 +/- 0.60 | 1.244x | 0.793x | 70.4% | 6/18 | 9/18 | free: trim8k n-max 3, --spec-draft-p-min 0.5 |
| e2-free-adapt3 | free | 93.44 +/- 1.16 | 1.367x | 0.872x | 45.4% | 6/18 | 18/18 | free: trim8k history-adaptive n in [1,3] |
| e3-rep-mtp | rep | 146.74 +/- 0.50 | - | 1.000x | 84.4% | - | 18/18 | rep: trim8k MTP n-max 2 (control) |
| e3-rep-ngs-mtp-n4m8 | rep | 201.10 +/- 1.00 | - | 1.370x | 58.5% | - | 18/18 | rep: ngram-simple(N4,M8) then MTP trim8k, n-max 8 |
| e3-rep-ngmod-mtp | rep | 592.24 +/- 30.38 | - | 4.036x | 77.0% | - | 18/18 | rep: ngram-mod then MTP trim8k |
| e3-rep-ngs-only | rep | 109.75 +/- 2.12 | - | 0.748x | 67.4% | - | 18/18 | rep: ngram-simple(N4,M8) only, n-max 8 |
| e3-free-mtp | free | 107.78 +/- 0.70 | 1.577x | 1.006x | 48.3% | 6/18 | 18/18 | free: trim8k MTP n-max 2 (control) |
| e3-free-ngs-mtp-n4m8 | free | 89.09 +/- 0.29 | 1.304x | 0.831x | 39.1% | 3/18 | 9/18 | free: ngram-simple(N4,M8) then MTP trim8k, n-max 8 |
| e5-plain-q8head | free | 77.09 +/- 0.80 | 1.128x | 0.719x | - | 3/18 | 3/18 | free: plain, bf16 body + Q8_0 token_embd/output head (speed ceiling, lossy) |
| e5-trim8k-q8head | free | 121.33 +/- 0.46 | 1.775x | 1.132x | 47.5% | 6/18 | 0/18 | free: trim8k MTP n2 -bs, bf16 body + Q8_0 head (speed ceiling, lossy) |
