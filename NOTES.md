# Known retrieval failures (fix in Step 5, measure first)

1. "Where are cookies stored in a session?" misses Session.__init__.
   Cause: all-MiniLM-L6-v2 reads only the first 256 tokens; Session.__init__ has
   677 and the cookies line is near the end. 65 of 287 chunks are longer than 256.
2. "How does requests handle redirects?" ranks resolve_redirects 5th, below weaker chunks.
3. Tiny chunks (4 to 5 lines, like HTTPBasicAuth) score high even when they only match on keywords.