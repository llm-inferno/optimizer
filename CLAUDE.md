# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the **Inference System Optimizer** — a Go service that assigns GPU types to LLM inference servers and determines optimal replica counts and batch sizes given traffic loads and service classes. It uses queueing theory for performance modeling and supports multiple solver backends (greedy, MILP).

## Build & Run

**Build via Docker (recommended — handles lpsolve dependency):**
```bash
docker build -t inferno . --load
```

**Build locally (requires lpsolve installed from https://github.com/llm-inferno/lpsolve):**
```bash
export CGO_CFLAGS="-I/usr/include/lpsolve"
export CGO_LDFLAGS="-llpsolve55 -lm -ldl -lcolamd"
go build ./...
```

**Run REST API server:**
```bash
cd cmd/optimizer
go run main.go        # Stateless mode (default)
go run main.go -F     # Stateful mode
```

**Run demo (requires sample data submodule):**
```bash
git submodule init && git submodule update
cd demos/main
go run main.go [small|large]
```

## Service environment variables

| Variable | Purpose | Default |
|---|---|---|
| `INFERNO_HOST` | REST server listen address | `""` (all interfaces) |
| `INFERNO_PORT` | REST server listen port | `8080` |

## Architecture

### Solver Selection Logic (`pkg/solver/solver.go`)
Three solver modes determined by `OptimizerSpec`. The local `Solver` embeds `optimizer-light`'s `Solver` and overrides `Solve()` to add the MILP branch; greedy and unlimited are fully delegated to `optimizer-light`:
- **Unlimited**: No capacity constraints — picks minimum-cost feasible allocation per server (delegated to `optimizer-light`)
- **MILP**: Integer programming via lpsolve for globally optimal solution under capacity constraints (local `milpsolver.go`)
- **Greedy** (default): Sorts servers by priority/cost-delta, allocates greedily until capacity exhausted (delegated to `optimizer-light`)

### Performance Model (`github.com/llm-inferno/queue-analysis/pkg/analyzer`)
Uses M/G/c queueing theory to predict per-server metrics given `(accelerator, replicas, batchSize, requestRate)`:
- **TTFT** (time to first token): includes queueing wait + prefill time
- **ITL** (inter-token latency): decode time per token
- **TPS** (tokens per second): throughput
- Prefill model: `γ + δ × inputTokens × batchSize`
- Decode model: `α + β × batchSize`

These parameters (`alpha`, `beta`, `gamma`, `delta`) are per `(model, accelerator)` pair in `ModelSpec`.

### Core Domain (`github.com/llm-inferno/optimizer-light/pkg/core`)
The core domain types live in `optimizer-light` and are imported directly. Key types:
- `system.go`: `TheSystem` singleton — central registry of all entities
- `allocation.go`: `Allocation` for a server — holds `(accelerator, replicas, batchSize)`; `FeasibleAllocations()` generates and filters candidates via QueueAnalyzer against SLO targets
- `server.go`: `Server` — maps to a `(serviceClass, model)` pair with a target request rate
- `serviceclass.go`: `ServiceClass` — priority + SLO targets (ITL, TTFT, TPS)

### REST Server (`rest-server/`)
- `stateless.go`: Single `/optimizeOne` POST endpoint — full system data in each request
- `statefull.go`: Full CRUD state management + `/optimize` endpoint
- API spec documented in `rest-server/README.md`

### Configuration Types (`github.com/llm-inferno/optimizer-light/pkg/config/types.go`)
All JSON data structures live in `optimizer-light`: `AcceleratorSpec`, `ModelSpec`, `ServerSpec`, `ServiceClassSpec`, `CapacitySpec`, `OptimizerSpec`, `AllocationSolution`. These are the wire format for both file-based and REST API input/output. `OptimizerSpec` includes the MILP-specific fields `MILPSolver`, `UseCplex`, and `Heterogeneous` (ignored by `optimizer-light`).

## Key Relationships

```
ServiceClass (priority, SLO targets)
    └─ Server (model + request rate)
           └─ Allocation (accelerator + replicas + batchSize)
                  └─ QueueAnalyzer validates against SLOs
```

The optimizer finds the best `Allocation` for each `Server` subject to `Capacity` (available GPU counts per type) and `ServiceClass` SLO constraints.

## Testing

No dedicated test files currently. Use the demo (`demos/main/main.go`) and sample data submodule for integration-style validation.
