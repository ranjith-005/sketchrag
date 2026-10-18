## Answer quality (task score x100)

| System | NrtvQA | Qasper | MFQA | HotpotQA | 2WikiMQA | MuSiQue | PRetr | PassKey | KV | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25_rag | 16.2 | 31.4 | 55.0 | 42.4 | 12.1 | 26.7 | 40.0 | 100.0 | 90.0 | **46.0** |
| dense_rag | 18.8 | 37.5 | 45.0 | 46.3 | 29.8 | 27.7 | 17.5 | 100.0 | 10.0 | **37.0** |
| full_context | 22.4 | 39.7 | 41.6 | 38.3 | 35.0 | 11.5 | 60.0 | 20.0 | 10.0 | **31.0** |
| sketchrag_gate | 19.1 | 40.3 | 47.4 | 47.3 | 16.2 | 20.0 | 37.5 | 100.0 | 90.0 | **46.4** |
| sketchrag_raw | 25.3 | 34.1 | 40.3 | 56.4 | 18.6 | 24.5 | 43.5 | 100.0 | 90.0 | **48.1** |

## Efficiency (mean per question)

| System | LLM ctx tok | out tok | calls | LLM s / q | hot KB | ingest ms/chunk | retrieve ms | fallback rate | ans. recall |
|---|---|---|---|---|---|---|---|---|---|
| bm25_rag | 1438 | 16 | 1.00 | 12.8 | 151 | - | 2.0 | 0.00 | 0.58 |
| dense_rag | 1457 | 15 | 1.00 | 15.0 | 258 | - | 0.3 | 0.00 | 0.52 |
| full_context | 13699 | 10 | 1.00 | 104.6 | 88 | - | - | 0.00 | 0.67 |
| sketchrag_gate | 1343 | 15 | 1.00 | 11.3 | 466 | 14.07 | 16.2 | 0.48 | 0.53 |
| sketchrag_raw | 1724 | 15 | 1.00 | 13.5 | 466 | 13.36 | 18.4 | 1.00 | 0.58 |
