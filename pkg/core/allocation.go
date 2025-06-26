package core

import (
	"bytes"
	"fmt"
	"math"

	"github.com/llm-inferno/optimizer/pkg/config"
	"github.com/llm-inferno/queue-analysis/pkg/queue"
	"github.com/llm-inferno/queue-analysis/pkg/utils"
)

// Allocation details of an accelerator to a server
type Allocation struct {
	accelerator string  // name of accelerator
	numReplicas int     // number of server replicas
	batchSize   int     // max batch size
	cost        float32 // cost of this allocation
	value       float32 // value of this allocation
	servTime    float32 // expected average token service time
	waitTime    float32 // expected average request queueing time
	rho         float32 // expected busy server defined as (1 - probability of at least one request running)

	maxArrvRatePerReplica float32 // maximum arrival rate per replica
}

// queueing model used in performance analysis
var queueModel *queue.MM1ModelStateDependent

// Create an allocation of an accelerator to a server; nil if not feasible
func CreateAllocation(serverName string, gName string) *Allocation {
	var (
		acc *Accelerator

		server *Server
		load   *config.ServerLoadSpec

		model *Model
		perf  *config.ModelAcceleratorPerfData

		svc    *ServiceClass
		target *Target
	)

	// get accelerator info
	if acc = GetAccelerator(gName); acc == nil {
		return nil
	}

	// get server info
	if server = GetServer(serverName); server == nil {
		return nil
	}
	if load = server.Load(); load == nil || load.ArrivalRate <= 0 || load.AvgLength <= 0 {
		return nil
	}

	// get model info
	modelName := server.ModelName()
	if model = GetModel(modelName); model == nil {
		return nil
	}
	if perf = model.PerfData(gName); perf == nil {
		return nil
	}

	// get service class info
	if svc = GetServiceClass(server.ServiceClassName()); svc == nil {
		return nil
	}
	if target = svc.ModelTarget(modelName); target == nil {
		return nil
	}

	// calculate max batch size (N) based on average request length (K)
	K := load.AvgLength
	N := max(perf.MaxBatchSize*perf.AtTokens/K, 1)
	maxQueue := N * config.MaxQueueToBatchRatio

	// distribution of token time assumed deterministic
	servTimeLimit := float32(K) * target.ITL
	// distribution of waiting time assumed exponential
	waitTimeLimit := target.TTW / config.SLOMargin
	// desired throughput (requests/msec)
	throughputLimit := target.TPS / (1000 * float32(K))

	// calculate state-dependent service rate for queueuing model
	servRate := make([]float32, N)
	for n := 1; n <= N; n++ {
		servTime := perf.Alpha + perf.Beta*float32(n)
		servRate[n-1] = float32(n) / (servTime * float32(K))
	}

	// analyze queueuing model
	queueModel = queue.NewMM1ModelStateDependent(maxQueue, servRate)
	lambdaMin := servRate[0] * config.Delta
	lambdaMax := servRate[N-1] * (1 - config.Delta)

	// determine rate at which the average service time is below the service time limit
	lambdaStarService := lambdaMax
	if target.ITL > 0 {
		lambda, ind, err := utils.BinarySearch(lambdaMin, lambdaMax, servTimeLimit, EvalServTime)
		if err != nil {
			fmt.Println(err.Error())
			return nil
		}
		if ind < 0 {
			return nil // unattainable service time limit
		}
		lambdaStarService = lambda
	}

	// determine rate at which the average waiting time is below to the waiting time limit
	lambdaStarWait := lambdaMax
	if target.TTW > 0 {
		lambda, ind, err := utils.BinarySearch(lambdaMin, lambdaMax, waitTimeLimit, EvalWaitingTime)
		if err != nil {
			fmt.Println(err.Error())
			return nil
		}
		if ind < 0 {
			return nil // unattainable waiting time limit
		}
		lambdaStarWait = lambda
	}

	// determine rate for max throughput
	lambdaStarThroughput := lambdaMax
	if target.TPS > 0 {
		lambdaStarThroughput = lambdaMax * (1 - config.StabilitySafetyFraction)
	}

	// arrival rate satisfying all SLOs
	lambdaStar := float32(math.Min(float64(lambdaStarService), float64(lambdaStarWait)))
	lambdaStar = float32(math.Min(float64(lambdaStar), float64(lambdaStarThroughput)))

	// calculate number of replicas
	var totalLambda float32
	if target.TPS == 0 {
		totalLambda = load.ArrivalRate / 60 / 1000
	} else {
		totalLambda = throughputLimit
	}
	numReplicas := int(math.Ceil(float64(totalLambda) / float64(lambdaStar)))

	// calculate cost
	totalNumInstances := model.NumInstances(gName) * numReplicas
	cost := acc.Cost() * float32(totalNumInstances)

	// queueModel.Solve(lambdaStar, 1)
	// fmt.Printf("model=%s; accelerator=%s; lambdaMin=%v; lambdaMax=%v; servTimeLimit= %v; waitTimeLimit=%v; lambdaStarService=%v; lambdaStarWait=%v; lambdaStarThroughput= %v, lambdaStar=%v \n",
	// 	model.Name(), gName,
	// 	lambdaMin, lambdaMax, servTimeLimit, waitTimeLimit, lambdaStarService, lambdaStarWait, lambdaStarThroughput, lambdaStar)
	// fmt.Println(queueModel)

	// calculate queue statistics
	lambda := totalLambda / float32(numReplicas)
	queueModel.Solve(lambda, 1)
	rho := queueModel.GetRho()
	servTime := queueModel.GetAvgServTime() / float32(K)
	wait := queueModel.GetAvgWaitTime()
	// fmt.Printf("numReplicas=%d; batchSize=%d; lambda=%v, tokenTime=%v; wait=%v; \n", numReplicas, N, lambda, servTime, wait)

	alloc := &Allocation{accelerator: gName, numReplicas: numReplicas, batchSize: N,
		cost: cost, servTime: servTime, waitTime: wait, rho: rho, maxArrvRatePerReplica: lambdaStar}
	alloc.SetValue(alloc.cost)
	return alloc
}

func EvalWaitingTime(x float32) (float32, error) {
	queueModel.Solve(x, 1)
	if !queueModel.IsValid() {
		return 0, fmt.Errorf("invalid model %v", queueModel)
	}
	return queueModel.GetAvgWaitTime(), nil
}

func EvalServTime(x float32) (float32, error) {
	queueModel.Solve(x, 1)
	if !queueModel.IsValid() {
		return 0, fmt.Errorf("invalid model %v", queueModel)
	}
	return queueModel.GetAvgServTime(), nil
}

// Create an allocation for an accelerator to a server; nil if not feasible
// (using G/G/m model approximation)
func CreateAllocationUsingGGm(serverName string, gName string) *Allocation {
	var (
		acc *Accelerator

		server *Server
		load   *config.ServerLoadSpec

		model *Model
		perf  *config.ModelAcceleratorPerfData

		svc    *ServiceClass
		target *Target
	)

	// get accelerator info
	if acc = GetAccelerator(gName); acc == nil {
		return nil
	}

	// get server info
	if server = GetServer(serverName); server == nil {
		return nil
	}
	if load = server.Load(); load == nil {
		return nil
	}

	// get model info
	modelName := server.ModelName()
	if model = GetModel(modelName); model == nil {
		return nil
	}
	if perf = model.PerfData(gName); perf == nil {
		return nil
	}

	// get service class info
	if svc = GetServiceClass(server.ServiceClassName()); svc == nil {
		return nil
	}
	if target = svc.ModelTarget(modelName); target == nil {
		return nil
	}

	K := load.AvgLength
	N := max(perf.MaxBatchSize*perf.AtTokens/K, 1)

	servTime := perf.Alpha + perf.Beta*float32(N)
	if target.ITL > 0 && servTime > target.ITL {
		return nil
	}

	numReplicas := 0
	gamma := ((load.ArrivalCOV * load.ArrivalCOV) + (load.ServiceCOV * load.ServiceCOV)) / 2
	if target.ITL > 0 && target.TTW > 0 {
		waitTimeLimit := target.TTW / config.SLOMargin
		xStar := float32(perf.MaxBatchSize) * waitTimeLimit / (float32(K) * servTime * gamma)
		rhoStar := xStar / (1 + xStar)
		lambdaStar := rhoStar / (float32(K) * servTime)
		numReplicas = int(math.Ceil(float64(load.ArrivalRate) / (float64(lambdaStar) * 60 * 1000)))
	}
	if target.TPS > 0 {
		lambdaMax := float32(N) / (servTime * float32(K))
		lambdaStarThroughput := lambdaMax * (1 - config.StabilitySafetyFraction)
		throughputTarget := target.TPS / (1000 * float32(K))
		n := int(math.Ceil(float64(throughputTarget) / float64(lambdaStarThroughput)))
		numReplicas = max(numReplicas, n)
	}
	if numReplicas == 0 {
		return nil
	}

	// calculate cost
	totalNumInstances := model.NumInstances(gName) * numReplicas
	cost := acc.Cost() * float32(totalNumInstances)

	rho := load.ArrivalRate * float32(K) * servTime / (float32(numReplicas) * 60 * 1000)
	x := rho / (1 - rho)
	wait := (float32(K) * servTime) * gamma * x / float32(perf.MaxBatchSize)

	alloc := &Allocation{accelerator: gName, numReplicas: numReplicas, batchSize: N,
		cost: cost, servTime: servTime, waitTime: wait, rho: rho}
	alloc.SetValue(alloc.cost)
	return alloc
}

func (a *Allocation) Scale(serverName string) (alloc *Allocation, inc int) {
	var (
		acc    *Accelerator
		server *Server
		load   *config.ServerLoadSpec
	)

	// get server info
	if server = GetServer(serverName); server == nil {
		return nil, 0
	}
	if load = server.Load(); load == nil {
		return nil, 0
	}

	// get accelerator info
	gName := a.accelerator
	if acc = GetAccelerator(gName); acc == nil {
		return nil, 0
	}

	// create new allocation
	alloc = CreateAllocation(serverName, gName)
	inc = alloc.numReplicas - a.numReplicas
	return alloc, inc
}

func (a *Allocation) ReAllocate(serverName string) (*Allocation, string) {
	minVal := float32(0)
	var minAlloc *Allocation
	for gName := range GetAccelerators() {
		if alloc := CreateAllocation(serverName, gName); alloc != nil {
			if minVal == 0 || alloc.value < minVal {
				minVal = alloc.value
				minAlloc = alloc
			}
		}
	}
	if minAlloc == nil {
		return nil, ""
	}
	return minAlloc, minAlloc.accelerator
}

func (a *Allocation) Accelerator() string {
	return a.accelerator
}

func (a *Allocation) NumReplicas() int {
	return a.numReplicas
}

func (a *Allocation) MaxArrvRatePerReplica() float32 {
	return a.maxArrvRatePerReplica
}

// Set the value for this allocation (may depend on cost, performance, ...)
func (a *Allocation) SetValue(value float32) {
	a.value = value
}

func (a *Allocation) Value() float32 {
	return a.value
}

// Calculate penalty for transitioning from this allocation (a) to another allocation (b)
func (a *Allocation) TransitionPenalty(b *Allocation) float32 {
	if a.accelerator == b.accelerator {
		if a.numReplicas == b.numReplicas {
			return 0
		} else {
			return b.cost - a.cost
		}
	}
	return config.AccelPenaltyFactor*(a.cost+b.cost) + (b.cost - a.cost)
}

func (a *Allocation) Clone() *Allocation {
	return &Allocation{
		accelerator: a.accelerator,
		numReplicas: a.numReplicas,
		batchSize:   a.batchSize,
		cost:        a.cost,
		value:       a.value,
		servTime:    a.servTime,
		waitTime:    a.waitTime,
		rho:         a.rho,

		maxArrvRatePerReplica: a.maxArrvRatePerReplica,
	}
}

func (a *Allocation) AllocationData() *config.AllocationData {
	return &config.AllocationData{
		Accelerator: a.accelerator,
		NumReplicas: a.numReplicas,
		MaxBatch:    a.batchSize,
		Cost:        a.cost,
		ITLAverage:  a.servTime,
		WaitAverage: a.waitTime,
	}
}

func AllocationFromData(data *config.AllocationData) *Allocation {
	return &Allocation{
		accelerator: data.Accelerator,
		numReplicas: data.NumReplicas,
		batchSize:   data.MaxBatch,
		cost:        data.Cost,
		servTime:    data.ITLAverage,
		waitTime:    data.WaitAverage,
	}
}

func (a *Allocation) String() string {
	return fmt.Sprintf("{acc=%s; num=%d; maxBatch=%d; cost=%v, val=%v, servTime=%v, waitTime=%v, rho=%v}",
		a.accelerator, a.numReplicas, a.batchSize, a.cost, a.value, a.servTime, a.waitTime, a.rho)
}

// Orchestration difference between two allocations
type AllocationDiff struct {
	oldAccelerator string
	newAccelerator string
	oldNumReplicas int
	newNumReplicas int
	costDiff       float32
}

func CreateAllocationDiff(a *Allocation, b *Allocation) *AllocationDiff {
	if a == nil && b == nil {
		return nil
	}
	oldAccelerator := "none"
	newAccelerator := "none"
	oldNumReplicas := 0
	newNumReplicas := 0
	oldCost := float32(0)
	newCost := float32(0)
	if a != nil {
		oldAccelerator = a.accelerator
		oldNumReplicas = a.numReplicas
		oldCost = a.cost
	}
	if b != nil {
		newAccelerator = b.accelerator
		newNumReplicas = b.numReplicas
		newCost = b.cost
	}
	return &AllocationDiff{
		oldAccelerator: oldAccelerator,
		newAccelerator: newAccelerator,
		oldNumReplicas: oldNumReplicas,
		newNumReplicas: newNumReplicas,
		costDiff:       newCost - oldCost,
	}
}

func (d *AllocationDiff) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "{ %s -> %s, %d -> %d, %v }",
		d.oldAccelerator, d.newAccelerator, d.oldNumReplicas, d.newNumReplicas, d.costDiff)
	return b.String()
}
