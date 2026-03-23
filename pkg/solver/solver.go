package solver

import (
	"bytes"
	"fmt"

	lightsolver "github.com/llm-inferno/optimizer-light/pkg/solver"

	"github.com/llm-inferno/optimizer-light/pkg/config"
	"github.com/llm-inferno/optimizer-light/pkg/core"
)

// Solver extends optimizer-light's Solver with MILP support
type Solver struct {
	*lightsolver.Solver
	spec          *config.OptimizerSpec
	milpDiffAlloc map[string]*core.AllocationDiff
}

func NewSolver(spec *config.OptimizerSpec) *Solver {
	return &Solver{
		Solver: lightsolver.NewSolver(spec),
		spec:   spec,
	}
}

// Solve overrides the base solver's Solve to add MILP dispatch
func (s *Solver) Solve() error {
	if !s.spec.MILPSolver {
		return s.Solver.Solve() // delegate to light solver (greedy or unlimited)
	}

	// MILP path: snapshot current allocations, solve, then compute diffs
	prev := make(map[string]*core.Allocation)
	for serverName, server := range core.GetServers() {
		if alloc := server.CurAllocation(); alloc != nil {
			prev[serverName] = alloc
		}
	}

	if err := s.SolveMILP(); err != nil {
		return err
	}

	s.milpDiffAlloc = make(map[string]*core.AllocationDiff)
	for serverName, server := range core.GetServers() {
		if allocDiff := core.CreateAllocationDiff(prev[serverName], server.Allocation()); allocDiff != nil {
			s.milpDiffAlloc[serverName] = allocDiff
		}
	}
	return nil
}

func (s *Solver) SolveMILP() error {
	mip := NewMILPSolver(s.spec)
	return mip.Solve()
}

// AllocationDiff overrides the base method to return MILP diffs when applicable
func (s *Solver) AllocationDiff() map[string]*core.AllocationDiff {
	if s.milpDiffAlloc != nil {
		return s.milpDiffAlloc
	}
	return s.Solver.AllocationDiff()
}

// String overrides the base method to include MILP diffs
func (s *Solver) String() string {
	if s.milpDiffAlloc != nil {
		var b bytes.Buffer
		b.WriteString("Solver: \n")
		for serverName, allocDiff := range s.milpDiffAlloc {
			fmt.Fprintf(&b, "sName=%s, allocDiff=%v \n", serverName, allocDiff)
		}
		return b.String()
	}
	return s.Solver.String()
}
