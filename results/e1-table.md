| config | set | tok/s mean +/- SD | vs plain (same set) | vs ref best | accept | out = plain | out = ref | what |
|---|---|---|---|---|---|---|---|---|
| e1-free-plain | free | 65.24 +/- 1.16 | 1.000x | 0.609x | - | self | - | free: plain bf16 (release build) |
| e1-free-untrim-n2 | free | 75.15 +/- 0.74 | 1.152x | 0.701x | 49.7% | 10/30 | - | free: MTP n-max 2 -bs, untrimmed (own build) |
| e1-free-trim8k-n2 | free | 107.15 +/- 0.89 | 1.642x | 1.000x | 48.3% | 10/30 | - | free: MTP n-max 2 -bs, trimmed 8k (own build) |
| e1-free-trim32k-n2 | free | 104.15 +/- 0.07 | 1.596x | 0.972x | 49.0% | 10/30 | - | free: MTP n-max 2 -bs, trimmed 32k (own build) |
