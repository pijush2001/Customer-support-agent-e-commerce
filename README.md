# Runnable example: the customer support agent

A small LangGraph implementation of the agent designed in the [case study](../README.md), now backed by real PDF extraction, chunking, local semantic embeddings, and a persistent Chroma vector database. The decision policy remains deterministic by default, order lookup and ticket creation remain mocks, and the guardrail is a teaching example, not production-grade security.

## Run it

From this `code` directory in Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe ingest.py --sample
.\venv\Scripts\python.exe run.py
.\venv\Scripts\python.exe run.py "Can I send my purchase back?"
```

On macOS/Linux, create a venv with `python3 -m venv venv` and use `./venv/bin/python` for the remaining commands.

The first ingestion downloads the ONNX `all-MiniLM-L6-v2` embedding model (approximately 80 MB). Subsequent embedding and search run locally without an API key or a database server. Installing packages and downloading the model require internet access initially. Model files are cached under your user home directory; the vector database lives in `data/chroma/`.

`--sample` creates `data/policies/fictional_store_policies.pdf` if it does not exist. It contains six clearly fictional policies: returns, refunds, cancellations, shipping, damaged items, and account security. Do not use these invented policies as real business rules. Generated PDFs and the database are ignored by Git.

With no argument, `run.py` executes five scenarios and asserts expected answers/escalations. Pass a question in quotes for a single request. The agent does not build the index automatically: a missing index raises a setup instruction rather than silently answering from hard-coded data.

## Two separate code flows

**Ingestion, when the documents change:**

```text
ingest.py main()
  -> generate_sample_pdf()              optional fictional PDF creation
  -> ingest_documents()
	  -> load_pdf_chunks()
		  -> PdfReader                   extract text page by page
		  -> RecursiveCharacterTextSplitter
													split text with sentence/paragraph preference
	  -> open_collection(create=True)   persistent Chroma collection, cosine distance
	  -> collection.upsert()            ONNX MiniLM embeds chunks; Chroma stores/indexes them
	  -> collection.delete()            remove stale IDs after successful upserts
```

**Answering, for each user question:**

```text
run.py main() -> show() -> support_agent.answer() -> APP.invoke()
  -> n_retrieve() -> kb.retrieve() -> ingest.search_policies()
	  -> open_collection()              reopen existing persistent database
	  -> collection.query()             embed question with the same model; search vectors
	  -> cosine-distance cutoff         keep up to k relevant matches (default k=2)
	  -> (source, text) pairs            PDF filename, page number, and chunk number
  -> n_agent() -> llm.decide()          answer / tool / escalate
  -> optional mock tool -> n_agent()
  -> n_guardrail() -> answer or escalation
```

An **embedding** is a 384-number vector representing a chunk's meaning. The embedding model is different from the optional chat model that chooses agent actions. **Chroma** stores the vectors alongside the original text and metadata, and manages a **cosine HNSW index** for similarity search (very small insert batches can initially be searched in Chroma's exact-search buffer). **Retrieval** embeds the question and finds nearby vectors; it does not rebuild document embeddings on every question.

Chunks are at most 800 characters, with a target overlap of 120 characters so nearby chunks can share context. Splitting prefers paragraphs and sentences, falls back to lines/words, and never joins different PDF pages. This is character-based, not token-based; MiniLM truncates input beyond its 256-token limit, so tune sizes on your own PDFs. Every stored chunk carries source filename, 1-based page, 1-based chunk number, and a SHA-256 file fingerprint.

## Use your own PDFs and refresh the index

Put text-based policy PDFs under `data/policies/`, remove the fictional sample when replacing it with real policies, and run:

```powershell
.\venv\Scripts\python.exe ingest.py
```

Re-ingestion uses deterministic IDs and upserts, so it does not duplicate unchanged chunks. Changed PDFs get new IDs; removed or replaced chunks are deleted after the new chunks are stored. Empty input and unreadable/blank/encrypted pages fail before index updates rather than silently deleting a working corpus. Scanned/image-only PDFs require an OCR step, which is not implemented here.

You can choose ingestion paths with `--source-dir` and `--db-dir`. The agent still queries the default `data/chroma/`; use the default database for `run.py`, or pass your custom path directly to `search_policies(..., db_dir=...)` in Python. All default paths are based on the script directory, not the terminal's current directory.

Run ingestion while the agent is stopped. Refresh is manual, not scheduled, and batch updates are not an atomic index swap. This local starter has no multi-tenant access control, ingestion worker, automatic rollback, or production deployment setup.

## Checks and guardrails

- PDF parsing rejects missing text and encrypted documents; it does not sanitize arbitrary document instructions.
- Semantic retrieval accepts matches with cosine distance at most `0.55`. Smaller distance means more similar, not a probability of correctness. This provisional cutoff is checked on the sample queries, not calibrated for a real help center.
- The graph's existing guardrail runs after the agent decision/optional tools. It checks for any context/tool result and two literal injection phrases in retrieved content; it does not verify full answer faithfulness or protect tools before execution.
- Refund/cancellation escalation remains in the offline policy and real-model prompt. There is no actual refund tool. Existing order/ticket tools are still mocked, with no authentication or live order records.
- The offline answer policy prints the best chunk verbatim with its citation. It does not synthesize all retrieved chunks or infer intent semantically; keyword-based tool routing can still misclassify a policy question containing an action keyword.

Run the real PDF/vector integration checks and the existing agent checks:

```powershell
.\venv\Scripts\python.exe -m unittest test_ingest -v
.\venv\Scripts\python.exe run.py
```

The integration suite covers page/chunk provenance, 384-dimensional persisted embeddings, paraphrase retrieval, out-of-scope abstention, duplicate-free refresh, updated/deleted PDFs, blank pages, and a missing index. The first run needs the model download.

## Where to start reading

Read the files in this order:
1. `support_agent.py` the LangGraph state machine: the nodes, the edges, and the bounded agent loop. Start here to see the shape of the whole agent.
2. `ingest.py` the PDF-to-vector pipeline, sample PDF generator, persistent collection, and semantic search implementation; `kb.py` preserves the graph's retrieval interface.
3. `llm.py` the decision layer: a deterministic offline policy, plus a provider-agnostic real path that works with any model through LangChain's `init_chat_model`.
4. `run.py` the scenarios, the assertions, and the command-line entry point.

## Use a different framework

LangGraph is one choice for wiring the agent together, and the design does not depend on it. If you prefer another stack, ask your coding agent to reimplement the same architecture (retrieval, a bounded tool loop, a guardrail, and escalation) in the SDK you use, for example the OpenAI Agents SDK, the Anthropic SDK, LlamaIndex, or plain Python. The spine and the design decisions in the case study carry over unchanged.

## Plug in a real model (any provider)

The real path uses LangChain's `init_chat_model`, so it works with any provider. Install the integration for the one you want, set a model, and set that provider's key:

```bash
# OpenAI
pip install langchain-openai
export SUPPORT_AGENT_MODEL="gpt-4o-mini" OPENAI_API_KEY="sk-..."

# Anthropic
pip install langchain-anthropic
export SUPPORT_AGENT_MODEL="claude-sonnet-5" ANTHROPIC_API_KEY="sk-ant-..."

# Gemini
pip install langchain-google-genai
export SUPPORT_AGENT_MODEL="gemini-3.6-flash" GOOGLE_API_KEY="..."

python run.py
```

If `SUPPORT_AGENT_MODEL` is unset, the provider is auto-detected from whichever key is present. With no key at all, the agent runs on the deterministic offline policy, so the graph and the tests never depend on a provider.

## Production observability with Arize

This example prints a `trace` for each run. In production you would not eyeball prints, you would send structured traces to an observability platform and run evaluations on them. **[Arize](https://app.arize.com/auth/join?utm_source=aish&utm_medium=partner&utm_campaign=learnarize2026&utm_content=free-signup)** (Phoenix / AX) is the natural fit here: instrument the LangGraph app with OpenInference so every node, tool call, and model call becomes a span, then run online evals (faithfulness, refusal correctness, tool-call correctness) over live traffic and alert on drift. See the scale and evaluation follow-ups in the [case study](../README.md) for how this fits the design.
