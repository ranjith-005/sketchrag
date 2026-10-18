# Dev-split tuning log (seed 1000, 10 q/task, 9 tasks, answer recall; no LLM)

```
SUMMARY sketchrag_gate set=default recall=0.511 words=863 fallback=0.71
SUMMARY sketchrag_sketch set=default recall=0.400 words=527 fallback=0.00
SUMMARY sketchrag_gate@ret.w_lex=0 set=ret.w_lex=0 recall=0.544 words=828 fallback=0.73
SUMMARY sketchrag_sketch@ret.w_lex=0 set=ret.w_lex=0 recall=0.422 words=524 fallback=0.00
SUMMARY sketchrag_gate@ret.w_lex=1.0 set=ret.w_lex=1.0 recall=0.511 words=873 fallback=0.71
SUMMARY sketchrag_sketch@ret.w_lex=1.0 set=ret.w_lex=1.0 recall=0.400 words=527 fallback=0.00
SUMMARY sketchrag_gate@ret.confidence=noise set=ret.confidence=noise recall=0.544 words=926 fallback=0.83
SUMMARY sketchrag_sketch@ret.confidence=noise set=ret.confidence=noise recall=0.400 words=527 fallback=0.00
SUMMARY sketchrag_gate@ret.top_k=3 set=ret.top_k=3 recall=0.500 words=730 fallback=0.68
SUMMARY sketchrag_sketch@ret.top_k=3 set=ret.top_k=3 recall=0.400 words=432 fallback=0.00
SUMMARY sketchrag_gate@ret.top_k=6 set=ret.top_k=6 recall=0.544 words=1106 fallback=0.71
SUMMARY sketchrag_sketch@ret.top_k=6 set=ret.top_k=6 recall=0.433 words=716 fallback=0.00
SUMMARY sketchrag_gate@mem.max_cells=16 set=mem.max_cells=16 recall=0.500 words=955 fallback=0.78
SUMMARY sketchrag_sketch@mem.max_cells=16 set=mem.max_cells=16 recall=0.322 words=533 fallback=0.00
SUMMARY sketchrag_gate@mem.max_cells=64 set=mem.max_cells=64 recall=0.533 words=691 fallback=0.50
SUMMARY sketchrag_sketch@mem.max_cells=64 set=mem.max_cells=64 recall=0.456 words=523 fallback=0.00
SUMMARY sketchrag_gate@mem.max_cells=128 set=mem.max_cells=128 recall=0.500 words=626 fallback=0.40
SUMMARY sketchrag_sketch@mem.max_cells=128 set=mem.max_cells=128 recall=0.422 words=519 fallback=0.00
SUMMARY sketchrag_gate@mem.cms_width=128 set=mem.cms_width=128 recall=0.544 words=857 fallback=0.80
SUMMARY sketchrag_sketch@mem.cms_width=128 set=mem.cms_width=128 recall=0.411 words=526 fallback=0.00
SUMMARY sketchrag_gate@mem.cms_width=2048 set=mem.cms_width=2048 recall=0.533 words=859 fallback=0.74
SUMMARY sketchrag_sketch@mem.cms_width=2048 set=mem.cms_width=2048 recall=0.411 words=527 fallback=0.00
SUMMARY sketchrag_gate@ret.raw_per_cell=1 set=ret.raw_per_cell=1 recall=0.511 words=701 fallback=0.71
SUMMARY sketchrag_sketch@ret.raw_per_cell=1 set=ret.raw_per_cell=1 recall=0.400 words=527 fallback=0.00
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sketchrag_gate set=default recall=0.489 words=693 fallback=0.71
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sketchrag_sketch set=default recall=0.400 words=527 fallback=0.00
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sketchrag_raw set=default recall=0.589 words=960 fallback=1.00
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sketchrag_gate@ret.w_lex=0 set=ret.w_lex=0 recall=0.522 words=666 fallback=0.73
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sketchrag_sketch@ret.w_lex=0 set=ret.w_lex=0 recall=0.422 words=524 fallback=0.00
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sketchrag_raw@ret.w_lex=0 set=ret.w_lex=0 recall=0.622 words=950 fallback=1.00
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sketchrag_gate@ret.cooccur=false set=ret.cooccur=false recall=0.489 words=692 fallback=0.71
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sketchrag_sketch@ret.cooccur=false set=ret.cooccur=false recall=0.400 words=527 fallback=0.00
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sketchrag_raw@ret.cooccur=false set=ret.cooccur=false recall=0.589 words=960 fallback=1.00
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sketchrag_gate@ret.replace_digest=false set=ret.replace_digest=false recall=0.511 words=856 fallback=0.71
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sketchrag_sketch@ret.replace_digest=false set=ret.replace_digest=false recall=0.400 words=527 fallback=0.00
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sketchrag_raw@ret.replace_digest=false set=ret.replace_digest=false recall=0.611 words=1469 fallback=1.00
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sketchrag_gate@mem.max_cells=64 set=mem.max_cells=64 recall=0.533 words=593 fallback=0.52
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sketchrag_sketch@mem.max_cells=64 set=mem.max_cells=64 recall=0.456 words=523 fallback=0.00
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sketchrag_raw@mem.max_cells=64 set=mem.max_cells=64 recall=0.611 words=805 fallback=1.00
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sketchrag_gate@ret.w_lex=0,mem.max_cells=64 set=ret.w_lex=0 mem.max_cells=64 recall=0.600 words=580 fallback=0.56
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sketchrag_sketch@ret.w_lex=0,mem.max_cells=64 set=ret.w_lex=0 mem.max_cells=64 recall=0.489 words=519 fallback=0.00
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sketchrag_raw@ret.w_lex=0,mem.max_cells=64 set=ret.w_lex=0 mem.max_cells=64 recall=0.656 words=782 fallback=1.00
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY bm25_rag set=default recall=0.589 words=639 fallback=0.00
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY dense_rag set=default recall=0.578 words=630 fallback=0.00
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bud935dq9.output:SUMMARY sara_lite set=default recall=0.611 words=670 fallback=0.00
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bmz748gz4.output:SUMMARY sketchrag_gate@ret.w_lex=0.2,mem.max_cells=64 set=ret.w_lex=0.2 mem.max_cells=64 recall=0.544 words=587 fallback=0.52
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bmz748gz4.output:SUMMARY sketchrag_gate@ret.w_lex=0,mem.max_cells=128 set=ret.w_lex=0 mem.max_cells=128 recall=0.589 words=549 fallback=0.50
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bmz748gz4.output:SUMMARY sketchrag_gate@ret.w_lex=0,mem.max_cells=64,ret.top_k=5 set=ret.w_lex=0 mem.max_cells=64 ret.top_k=5 recall=0.600 words=672 fallback=0.53
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bmz748gz4.output:SUMMARY sketchrag_gate@ret.w_lex=0,mem.max_cells=64,ret.raw_per_cell=3 set=ret.w_lex=0 mem.max_cells=64 ret.raw_per_cell=3 recall=0.600 words=631 fallback=0.56
/Users/ADMIN/AppData/Local/Temp/claude/C--Users-ADMIN/0850ac34-2761-4fa0-893b-41a51fa0c463/tasks/bmz748gz4.output:SUMMARY sketchrag_gate@ret.w_lex=0,mem.max_cells=64,ret.contiguity=0 set=ret.w_lex=0 mem.max_cells=64 ret.contiguity=0 recall=0.578 words=489 fallback=0.56
```

Rounds 1 (tune2) used append-raw + term-only raw rerank; later rounds use replace_digest + dense+lex raw rerank + co-occurrence gate.
Frozen: mem.max_cells=64, ret.w_lex=0, ret.top_k=4, ret.contiguity=1, ret.raw_per_cell=2.
