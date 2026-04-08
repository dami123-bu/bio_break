---
name: review
description: Review Python code for general code quality. Trigger when the user asks to review code, check for issues, or asks "is this good Python".
---

# Code Review

Review the provided Python code for general code quality. Focus on things that make code hard to read, understand, or maintain.

## What to check

1. **Repeated code** — copy-pasted blocks or logic that should be a function
2. **Overly long methods** — functions doing too many things; suggest how to break them up
3. **Unclear naming** — variables or functions with names that don't explain their purpose
4. **Magic numbers/strings** — unexplained literals that should be named constants
5. **Unnecessary complexity** — convoluted logic that could be simplified
6. **File placement** — code in the wrong location:
   - Scripts that belong in `src/` or a named package sitting at the repo root
   - Utilities mixed in with entry points
   - Config or data files inside source packages
6. **LangChain/RAG patterns** — common mistakes in agent and retrieval code:
   - Building prompts with f-strings instead of `PromptTemplate` / `ChatPromptTemplate`
   - Not using `RunnablePassthrough` or LCEL pipes (`|`) when chaining — stringing `.run()` calls together manually
   - Embedding documents at query time instead of at ingest time
   - Re-instantiating the vector store or embeddings model on every call instead of once at startup
   - Ignoring the `metadata` field on documents (useful for filtering and attribution)
   - Using `similarity_search` and discarding scores when `similarity_search_with_score` would give useful signal

## Output Format

```
## Summary
[1-2 sentences on overall impression]

## Issues
[For each issue: what it is, where it is (file + line), and a simple fix suggestion]

## What's Done Well
[1-2 things if warranted — skip if nothing stands out]
```

Keep it short and practical. Skip anything that isn't a real problem.
