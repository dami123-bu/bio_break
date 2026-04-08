# PharmaHelp RAG attack scenarios for the current public repo

## Repository reality check

The current public repo already provides a persistent Chroma storage layer and a seeding script, but the online RAG query path is still only partially wired. The most defensible attacker work therefore targets:

1. the **current executable chat path** (lexical retrieval + top-document answer capture), and
2. the **persistent vector-store layer** (retrieval displacement and proto-context poisoning in an isolated lab collection).

## Scenario 1 — Current chat-path keyword hijack

**Scenario ID.** S1

**Name.** Lexical retrieval poisoning leading to top-document capture.

**Goal.** Make the current chat loop answer from attacker-controlled content.

**Preconditions.** The current public `retrieve_docs()` implementation uses lexical overlap over a small candidate set, and `synthesize_answer()` uses the highest-ranked document as the answer basis.

**Attacker capability.** The attacker can insert one additional candidate document into the retrieval set, or can influence a future upload path so that a keyword-dense attacker document becomes retrievable.

**Execution steps.**
1. Add one synthetic attacker document containing the target drug name and high-value query terms.
2. Submit a query such as `What is tamoxifen used for?`.
3. Observe whether the attacker document becomes rank 1.
4. Check whether the final answer contains the lab canary `SECURITY_TEST_OVERRIDE`.

**Success signal.** The top retrieved document is the attacker document and the answer includes the canary phrase.

**Why it matters.** Even before the full RAG pipeline is wired, the current system already exhibits a concrete rank-1 document-capture weakness.

## Scenario 2 — Isolated Chroma retrieval-bias poisoning

**Scenario ID.** S2

**Name.** Vector-store retrieval displacement in a lab collection.

**Goal.** Measure whether attacker documents displace benign PubMed documents in top-k retrieval.

**Preconditions.** `scripts/setup_kb.py` has already populated the benign `pubmed` collection.

**Attacker capability.** The attacker can add untrusted internal-document content to the same logical knowledge namespace used for retrieval.

**Execution steps.**
1. Clone a benign sample from `pubmed` into `pubmed_attack_lab`.
2. Inject two synthetic attacker documents that are dense in target drug terms.
3. Query the lab collection for a target drug question.
4. Record attacker-document rank, attacker-hit count, and retrieval bias.

**Success signal.** At least one attacker document appears in top-k, ideally rank 1.

**Why it matters.** This is the closest faithful attack to the public repo's existing Chroma layer without modifying the production `pubmed` baseline.

## Scenario 3 — Proto-context poisoning with instruction-bearing chunks

**Scenario ID.** S3

**Name.** Instruction-bearing chunk retrieval.

**Goal.** Show that attacker text can be retrieved into what would become the future LLM context window once synthesis is wired.

**Preconditions.** Same as Scenario 2.

**Attacker capability.** The attacker can store an instruction-bearing internal note or PDF-derived chunk that competes well in retrieval.

**Execution steps.**
1. Inject a synthetic internal note that contains realistic biomedical terms plus a benign lab canary (`SECURITY_TEST_CONTEXT_MARKER`).
2. Query the isolated lab collection.
3. Inspect the retrieved top-k documents.
4. Optionally emit the concatenated `context_preview` text to see what a future synthesis model would ingest.

**Success signal.** The canary-bearing attacker document appears in top-k and is visible in the context preview.

**Why it matters.** The public repo does not yet build a real LLM prompt from Chroma results, so this scenario is intentionally scoped as proto-context poisoning rather than a full end-to-end indirect prompt injection claim.

## Scenario 4 — Persistence in the storage layer

**Scenario ID.** S4

**Name.** Persistent retrieval poisoning.

**Goal.** Verify that once injected, poisoned content remains in the persistent lab collection across runs until explicitly removed.

**Preconditions.** The lab collection already contains attack payloads from Scenario 2 or 3.

**Attacker capability.** One successful write to the persistent store.

**Execution steps.**
1. Run Scenario 2 or 3 with `--fresh`.
2. Close the script.
3. Run `persistence_check` without reinjection.
4. Re-query the same collection.

**Success signal.** The same attack markers remain retrievable in the subsequent run.

**Why it matters.** This lets the project discuss persistence honestly, at the storage layer, even before memory-based agent persistence is implemented.

## Recommended next steps after these scenarios

1. Wire the online query path to Chroma so the same scenarios become end-to-end.
2. Add a real synthesis function that passes top-k retrieved text into Ollama.
3. Add metadata-based trust separation between PubMed and internal uploads.
4. Implement a true PDF ingest path and re-run Scenarios 2 and 3 with uploaded files rather than direct upserts.
5. Add defenses: ingest-time scanning, retrieval-time filtering, and output monitoring.
