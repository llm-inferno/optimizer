package solver

import (
	"bytes"
	"cmp"
	"fmt"
	"maps"
	"math"
	"slices"

	"github.com/llm-inferno/optimizer/pkg/config"
	"github.com/llm-inferno/optimizer/pkg/core"
)

// Entry for a server, used during greedy allocation
type serverEntry struct {
	serverName  string             // server name
	priority    int                // priority of service class for server
	curIndex    int                // current index in allocation list
	allocations []*core.Allocation // ordered list of allocations
	delta       float32            // delta penalty if current allocation not allowed and next allocation is allowed
}

func (e *serverEntry) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "sName=%s, prio=%d, curIndex=%d, delta=%v, allocations=%v \n",
		e.serverName, e.priority, e.curIndex, e.delta, e.allocations)
	return b.String()
}

// Find optimal allocations using greedy algorithm, assuming limited accelerator capacity
func (s *Solver) SolveGreedy() {

	// make a copy of count of available accelerator types
	available := make(map[string]int)
	maps.Copy(available, core.GetCapacities())

	// create entries for all servers, sorting candidate allocations per server
	var entries []*serverEntry = make([]*serverEntry, 0)
	for serverName, server := range core.GetServers() {
		server.RemoveAllocation()
		allAllocs := server.AllAllocations()
		if len(allAllocs) == 0 {
			continue
		}
		e := &serverEntry{
			serverName:  serverName,
			priority:    server.Priority(),
			curIndex:    0,
			allocations: make([]*core.Allocation, len(allAllocs)),
			delta:       0,
		}
		i := 0
		for _, alloc := range allAllocs {
			e.allocations[i] = alloc
			i++
		}
		slices.SortFunc(e.allocations, func(a, b *core.Allocation) int {
			return cmp.Compare(a.Value(), b.Value())
		})
		if len(e.allocations) > 1 {
			// value is difference between this and next allocation
			e.delta = e.allocations[1].Value() - e.allocations[0].Value()
		} else {
			// last choice, large value for not selecting this allocation
			e.delta = math.MaxFloat32
		}
		entries = append(entries, e)
	}

	// sorting function for server entries
	// - straight priorities, then delta values
	orderFunc := func(a, b *serverEntry) int {
		if a.priority == b.priority {
			if a.delta == b.delta {
				return cmp.Compare(b.allocations[b.curIndex].Value(), a.allocations[a.curIndex].Value())
			}
			return cmp.Compare(b.delta, a.delta)
		} else {
			return cmp.Compare(a.priority, b.priority)
		}
	}
	// sort server entries
	slices.SortFunc(entries, orderFunc)

	// keep track of unallocated servers, will process later
	unallocatedServers := make([]*serverEntry, 0)

	// start allocation greedily, in order
	for len(entries) > 0 {
		// pick top entry and remove from list
		top := entries[0]
		entries = entries[1:]
		// check if no more candidate allocations
		if len(top.allocations) == 0 {
			continue
		}

		// check if current allocation in entry can be satisfied
		serverName := top.serverName
		server := core.GetServer(serverName)
		if server == nil {
			continue
		}
		model := core.GetModel(server.ModelName())
		if model == nil {
			continue
		}
		alloc := top.allocations[top.curIndex]
		gName := alloc.Accelerator()
		acc := core.GetAccelerator(gName)
		tName := acc.Type()
		unitsPerReplica := model.NumInstances(gName) * acc.Spec().Multiplicity
		count := alloc.NumReplicas() * unitsPerReplica

		// check if accelerator type of current allocation is available, allocate
		if available[tName] >= count {
			available[tName] -= count
			server.SetAllocation(alloc)
		} else {
			// otherwise, move to next candidate allocation
			top.curIndex++
			if top.curIndex+1 < len(top.allocations) {
				// not last allocation, calculate delta
				top.delta = top.allocations[top.curIndex+1].Value() - top.allocations[top.curIndex].Value()
			} else if top.curIndex == len(top.allocations) {
				// no more allocations, could not satisfy any, add server to unallocated list
				unallocatedServers = append(unallocatedServers, top)
				continue
			} else {
				// last allocation, set large delta value
				top.delta = math.MaxFloat32
			}
			// reorder server entries
			i, _ := slices.BinarySearchFunc(entries, top, orderFunc)
			entries = slices.Insert(entries, i, top)
		}
	}

	// process unallocated servers based on saturation policy
	switch config.SaturatedAllocationPolicyEnum(s.optimizerSpec.SaturationPolicy) {
	case config.PriorityExhaustive:
		processUnallocatedServers(unallocatedServers, available)
	case config.PriorityRoundRobin:
		processGroupsOfUnallocatedServers(unallocatedServers, available)
	case config.RoundRobin:
		processUnallocatedServerGroup(unallocatedServers, available)
	case config.None:
	}
}

// Allocate remaining accelerators among unallocated servers
//   - priority ordering: one server at a time exhaustively, until no resources to satisfy requirements
func processUnallocatedServers(serverEntries []*serverEntry, available map[string]int) {
	// fmt.Println("Unallocated server entries: ", serverEntries)
	for _, entry := range serverEntries {
		for _, alloc := range entry.allocations {
			accName := alloc.Accelerator()
			serverName := entry.serverName
			server := core.GetServer(serverName)
			model := core.GetModel(server.ModelName())
			if acc := core.GetAccelerator(accName); acc != nil && model != nil && server != nil {
				if unitsPerReplica := model.NumInstances(accName) * acc.Spec().Multiplicity; unitsPerReplica > 0 {
					maxReplicas := available[acc.Type()] / unitsPerReplica
					if maxReplicas = min(maxReplicas, alloc.NumReplicas()); maxReplicas > 0 {
						curNumReplicas := alloc.NumReplicas()
						// adjust cost and value
						factor := float32(maxReplicas) / float32(curNumReplicas)
						alloc.SetCost(alloc.Cost() * factor)
						alloc.SetValue(alloc.Value() * factor)
						alloc.SetNumReplicas(maxReplicas)
						server.SetAllocation(alloc)
						count := maxReplicas * unitsPerReplica
						available[acc.Type()] -= count
						// fmt.Printf("updated allocation: server=%s, acc=%s, maxReplicas=%d, type=%s, count=%d \n",
						// 	serverName, accName, maxReplicas, acc.Type(), count)
						break
					}
				}
			}
		}
	}
}

// Allocate remaining accelerators among unallocated servers based on priorities
//   - priority grouping: one group of servers with same priority at a time
//   - round-robin within the group, until no resources to satisfy requirements
func processGroupsOfUnallocatedServers(serverEntries []*serverEntry, available map[string]int) {
	// fmt.Println("Unallocated server entries: ", serverEntries)

	// make groups of same priority servers, then process each group
	i := 0
	for i < len(serverEntries) {
		group := make([]*serverEntry, 0)
		group = append(group, serverEntries[i])
		groupPriority := serverEntries[i].priority
		i++
		for i < len(serverEntries) && serverEntries[i].priority == groupPriority {
			group = append(group, serverEntries[i])
			i++
		}
		processUnallocatedServerGroup(group, available)
	}
}

type serverAllocationTicket struct {
	entry  *serverEntry
	active bool // receiving allocation in round-robin
	server *core.Server
	model  *core.Model

	accType         string // type of accelerator allocated to server
	unitsPerReplica int
	numReplicas     int
	finalAlloc      *core.Allocation
}

// Allocate remaining accelerators among a group of unallocated servers
//   - round-robin allocation to members in group until no resources to satisfy requirements
func processUnallocatedServerGroup(serverEntries []*serverEntry, available map[string]int) {
	// fmt.Println("Unallocated server group entries: ", serverEntries)

	// create allocation tickets for all valid members in group
	tickets := make(map[string]*serverAllocationTicket)
	for _, serverEntry := range serverEntries {
		serverName := serverEntry.serverName
		server := core.GetServer(serverName)
		model := core.GetModel(server.ModelName())
		if model == nil || server == nil {
			continue
		}
		tickets[serverEntry.serverName] = &serverAllocationTicket{
			entry:  serverEntry,
			active: false,
			server: server,
			model:  model,
		}
	}

	// visit members in round-robin way
	allocatedTickets := make(map[string]*serverAllocationTicket)
	for len(tickets) > 0 {
		for _, serverEntry := range serverEntries {
			serverName := serverEntry.serverName
			var ticket *serverAllocationTicket
			if ticket = tickets[serverName]; ticket == nil {
				continue
			}
			// determine candidate allocation for not yet processed members
			if !ticket.active {
				for _, alloc := range serverEntry.allocations {
					accName := alloc.Accelerator()
					if acc := core.GetAccelerator(accName); acc != nil {
						unitsPerReplica := ticket.model.NumInstances(accName) * acc.Spec().Multiplicity
						if unitsPerReplica > 0 && available[acc.Type()] >= unitsPerReplica {
							ticket.active = true
							ticket.accType = acc.Type()
							ticket.unitsPerReplica = unitsPerReplica
							ticket.finalAlloc = alloc
							break
						}
					}
				}
				// check if no candidate allocation was found
				if !ticket.active {
					delete(tickets, serverName)
					continue
				}
			}
			// make one allocation (replica) to member
			replicasAvailable := available[ticket.accType] / ticket.unitsPerReplica
			if replicasAllocatable := min(replicasAvailable, ticket.finalAlloc.NumReplicas()); replicasAllocatable > 0 {
				ticket.numReplicas++
				available[ticket.accType] -= ticket.unitsPerReplica
				allocatedTickets[serverName] = ticket
			} else {
				// remove ticket if can no longer allocate
				delete(tickets, serverName)
			}
		}
	}
	// update allocated members
	for _, ticket := range allocatedTickets {
		alloc := ticket.finalAlloc
		numReplicas := ticket.numReplicas
		curNumReplicas := alloc.NumReplicas()
		// adjust cost and value
		factor := float32(numReplicas) / float32(curNumReplicas)
		alloc.SetCost(alloc.Cost() * factor)
		alloc.SetValue(alloc.Value() * factor)
		alloc.SetNumReplicas(numReplicas)
		ticket.server.SetAllocation(alloc)
		// count := ticket.numReplicas * ticket.unitsPerReplica
		// fmt.Printf("updated allocation: server=%s, acc=%s, accCount=%d, type=%s, count=%d \n",
		// 	ticket.server.Name(), alloc.Accelerator(), ticket.numReplicas, ticket.accType, count)
	}
}
