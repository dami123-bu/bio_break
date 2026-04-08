---
name: changes
description: Append a single-line entry to CHANGES.md summarizing uncommitted changes
---

## Changes Log

1. Run `git diff HEAD` to examine all changes since the last commit.
2. Summarize the changes in a single line, in this format:
   ```
   YYYY-MM-DD: <concise summary of what changed>
   ```
3. Append that line to `CHANGES.md` at the project root (create the file if it doesn't exist).
4. Do not include anything else — no headers, no bullet points, just the single line appended to the end of the file.