# app/graph/nodes

## OVERVIEW
LangGraph agent nodes (8 files) - core workflow logic.

## STRUCTURE
```
nodes/
├── __init__.py       # Exports
├── analyzer.py       # Task analysis (DO NOT modify task_type/next_node)
├── task_router.py    # Routing logic
├── retrieval.py      # RAG retrieval
├── generator.py      # Response generation
├── email_agent.py    # Email drafting
├── rfp_agent.py      # RFP drafting
└── file_extractor.py # Document extraction
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Add new node | Copy existing pattern | Follow TypedDict state updates |
| Modify routing | `task_router.py` | Controls which subgraph runs |
| Change retrieval | `retrieval.py` | Chroma query logic |

## ANTI-PATTERNS
- **DO NOT** modify `task_type` or `next_node` in analyzer.py (Korean comment: "수정 금지")
- Node functions must return state dict updates
- All nodes should be type-annotated
