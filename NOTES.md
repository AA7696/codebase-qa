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