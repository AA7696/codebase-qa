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