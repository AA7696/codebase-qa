# Known retrieval failures (fix in Step 5, measure first)

1. "Where are cookies stored in a session?" misses Session.__init__.
   Cause: all-MiniLM-L6-v2 reads only the first 256 tokens; Session.__init__ has
   677 and the cookies line is near the end. 65 of 287 chunks are longer than 256.
2. "How does requests handle redirects?" ranks resolve_redirects 5th, below weaker chunks.
3. Tiny chunks (4 to 5 lines, like HTTPBasicAuth) score high even when they only match on keywords.

# Step 3 observations

- The LLM sees the top 5 chunks, so a chunk ranked 5th (resolve_redirects) still produced a good answer.
- Cookies question answered correctly even though Session.__init__ was missed: the class header chunk saved it. Retrieval failure != answer failure; Step 4 must score both.
- Refusal worked on the GraphQL question (only 1 example so far).
- The redirects answer was 13 bullets and 3243 tokens: correct but verbose. Consider a "be concise" instruction later.

# Step 3 results

- Citation checker works: all cited numbers were real chunks. It checks existence only, not whether the chunk supports the claim.
- GraphQL question: correct refusal ("I don't know based on the retrieved code").
- FAILURE: "What does Session.mount do?" -> Session.mount not in top 5 -> model correctly refused.
  A retrieval failure, not a generation failure. Hypotheses to test in Step 5:
  (a) class name is missing from the embedded method text, (b) keyword search would catch "mount".
- Model mislabels methods (Session.resolve_redirects instead of SessionRedirectMixin.resolve_redirects).

# Baseline (vector search only, all-MiniLM-L6-v2, code text only, top 5)

Answerable questions: 57   (66 total, 9 unanswerable excluded)
Hit rate@5:  0.74
Recall@5:    0.67
MRR@10:      0.54
By kind (hits): lookup 19/27, behaviour 15/20, multi 8/10

# Step 4: why questions missed

- 13 of 15 misses have an expected chunk longer than 256 tokens (model reading limit), vs 22 of 42 hits.
  Suggestive, small sample. Truncation is a risk factor, not the only cause.
- Two misses are short chunks (Session.mount 110 tokens, RequestException 42 tokens): truncation does not explain them.

# Step 4: refusal and answer quality

- Unanswerable: 9/9 refused. Retrieved-chunk questions: 11/12 answered (row 18 refused).
- Row 18 (default hook events): answer is in module-level HOOKS constant, which no chunk contains (Step 1 edge case 3).
- Row 32: valid citation, misleading answer. Retrieval missed resolve_redirects, so the model described rewind_body instead of "body is dropped unless 307/308". A citation checker cannot catch this.
- Metric limits: "refused" matches the phrase anywhere (row 14 was a partial answer); row 8's label omits the Session header chunk that does contain the answer.

# Step 5: Ablation table (57 answerable questions, top 5)

| Version                                   | Hit@5 | Recall@5 | MRR@10 |
|-------------------------------------------|-------|----------|--------|
| Baseline: vector only, code only          | 0.74  | 0.67     | 0.54   |
| + file and symbol name in embedded text   | 0.75  | 0.69     | 0.60   |

5.1: fixed rows 3 (Session.mount) and 13 (RequestException); newly missed row 5.
Rows 8, 9, 31, 48 ranked worse. One question = 1.75 points of hit rate, so +0.01 is a single question.

| + hybrid search (BM25 + vector, RRF)      | 0.79  | 0.71     | 0.62   |

Keyword only (BM25): 0.72 / 0.65 / 0.57.
5.2: fixed rows 32, 5, 14, 7; broke rows 1, 13. Net +2 questions (43 -> 45 of 57).
Of 12 remaining misses, 7 are at rank 6 to 9 (reranker candidates); 5 are far (long chunks?).

| + reranker (ms-marco MiniLM), top 10      | 0.86  | 0.76     | 0.77   |

Reranker depth: 10 -> 0.86/0.76/0.77, 20 -> 0.86/0.79/0.74, 40 -> 0.84/0.77/0.74.
Chose depth 10 (same hit rate as 20, higher MRR, half the cost). Depth was chosen on the same 57 questions.
5.3: fixed rows 57, 1, 20, 28, 48, 9; broke rows 32, 46. Net +4 questions (45 -> 49 of 57).
Still missed (8): rows 32, 36, 46 (rank 8-10); rows 2, 13, 31, 8, 40 (rank 14-40, outside the reranker's pool).

| + pieces for long chunks (vector only)    | 0.81  | 0.74     | 0.63   |
| + pieces + hybrid                         | 0.84  | 0.76     | 0.66   |
| + pieces + hybrid + rerank top 10         | 0.86  | 0.77     | 0.76   |

5.4: pieces help the search stages (+3 questions each) but the final pipeline is unchanged
(49 of 57; the same 8 questions missed). Far misses became near misses (rank 6-16).
Untested idea: the reranker also truncates long chunks (max_length 512); try reranking best pieces.
Decision on keeping pieces: ____ because ____

| 5.5 control: core chunks only (new chunker)  | 0.88 | 0.79 | 0.77 |
| + docstring chunks                            | 0.86 | 0.77 | 0.74 |
| + constants chunks                            | 0.84 | 0.76 | 0.74 |
| + both                                        | 0.84 | 0.76 | 0.71 |

Control is 1 question above the 5.4 result (docs/ chunk removed, chunk order changed; cause not isolated).
Probe ranks: docstrings P2=1. Constants: P1=22 (fails), P3=5, P4=1.
Newly missed vs control: row 2 (docstring, constants, both); row 48 (constants, both).
Decision: docstring chunks in search index; constants kept out of ranking, tried as context expansion (5.5c).

5.5c (constants as context after ranking; docstring chunks in index):
Row 18 / P1 fixed: refusal -> correct answer ("response", hooks.py:22). P3 answered from models.py:95-105.
P4 not a clean test (Session.__init__ comment already says 30). 4 probes only.
Limitation: expansion adds unrelated constants (compat.py:110-115 in 3 of 4 prompts); tighten later.
Search ranking = "+ docstring" row (0.86 / 0.77 / 0.74); answer quality on 57 questions not re-measured.

5.6a: pipeline in scripts/rag.py (pieces + hybrid + rerank 10 + constants as context).
Sample of 20 (every 3rd row; only 1 unanswerable): expected chunk in top 5 for 15 of 19 answerable.
Answered without the expected chunk: rows 13, 31, 46 (to check by hand). Refused honestly: row 40.
~2,500 tokens per question.

5.6b judge (gpt-5-mini, same model as generator), 18 judged answers from a sample of 20:
Expected chunk found (15): faithful 15/15, correct yes 13 / partly 2.
Expected chunk not found (3): faithful 3/3, correct yes 1 / partly 1 / no 1 (row 31).
Faithfulness was at ceiling, so it did not discriminate; correctness did.
Rows 34, 49 "partly" despite chunk found: reference answer may be broader than the question (to check).
Limits: tiny sample, same-model judge, reference answers only verified for sessions.py.

# Retrieval ablation (57 answerable questions, top 5)

| Version                                          | Hit@5 | Recall@5 | MRR@10 |
|--------------------------------------------------|-------|----------|--------|
| Baseline: vector search, code only               | 0.74  | 0.67     | 0.54   |
| + file and symbol name in embedded text (5.1)    | 0.75  | 0.69     | 0.60   |
| + hybrid search, BM25 + vector with RRF (5.2)    | 0.79  | 0.71     | 0.62   |
| + cross-encoder reranker, top 10 (5.3)           | 0.86  | 0.76     | 0.77   |
| + pieces for long chunks (5.4)                   | 0.86  | 0.77     | 0.76   |
| New chunker, docs/ skipped, core chunks (5.5)    | 0.88  | 0.79     | 0.77   |
| + docstring chunks (final index)                 | 0.86  | 0.77     | 0.74   |

Final pipeline = pieces + hybrid + rerank 10 + docstring chunks, with constants added as context after ranking.
One question = 1.75 points of hit rate. Rows 5.5 onward are not strictly comparable to earlier rows
(the chunk set changed), and the depth/pieces choices were made on the same 57 questions.

6.2: Postgres 17 + pgvector 0.8.7 in Docker (pgvector/pgvector:pg17), Python connects via psycopg.
Gotcha 1: "@" in the password broke the DATABASE_URL (use letters and numbers only).
Gotcha 2: a Postgres service already installed on the laptop (postgres.exe) held port 5432, so the app was
talking to the wrong database. Docker's database now uses host port 5433. Found with netstat + tasklist.