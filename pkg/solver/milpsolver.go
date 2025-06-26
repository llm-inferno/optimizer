package solver

import (
	"fmt"

	lpsolveConfig "github.com/llm-inferno/lpsolve/pkg/config"
	lpsolve "github.com/llm-inferno/lpsolve/pkg/core"
	lpsolveUtils "github.com/llm-inferno/lpsolve/pkg/utils"
	"github.com/llm-inferno/optimizer/pkg/config"
	"github.com/llm-inferno/optimizer/pkg/core"
)

type MILPSolver struct {
	optimizerSpec *config.OptimizerSpec

	numServers             int         // number of servers (a pair of service class and model)
	numAccelerators        int         // number of accelerators
	instanceCost           []float64   // [numAccelerators]
	numInstancesPerReplica [][]int     // [sumServers][numAccelerators]
	ratePerReplica         [][]float64 // [sumServers][numAccelerators]
	arrivalRates           []float64   // [sumServers]

	numAcceleratorTypes    int     // number of accelerator types
	unitsAvail             []int   // [numAcceleratorTypes]
	acceleratorTypesMatrix [][]int // [numAcceleratorTypes][numAccelerators]

	numReplicas   [][]int // resulting number of replicas [numServers][numAccelerators]
	instancesUsed []int   // number of used accelerator units [numAccelerators]
	unitsUsed     []int   // [numAcceleratorTypes]

	accIndex      map[string]int // acceleratorName -> index in accelerator arrays
	accLookup     []string       // index -> acceleratorName
	serverIndex   map[string]int // serverName -> index in server arrays
	serverLookup  []string       // index -> serverName
	accTypeIndex  map[string]int // acceleratorTypeName -> index in acceleratorType arrays
	accTypeLookup []string       // index -> acceleratorTypeName
}

func NewMILPSolver(optimizerSpec *config.OptimizerSpec) *MILPSolver {
	return &MILPSolver{
		optimizerSpec: optimizerSpec,
	}
}

func (v *MILPSolver) Solve() error {
	v.preProcess()

	isLimited := !v.optimizerSpec.Unlimited
	isMulti := v.optimizerSpec.Heterogeneous
	useCplex := v.optimizerSpec.UseCplex
	if err := v.optimize(isLimited, isMulti, useCplex); err != nil {
		return err
	}

	v.postProcess()
	return nil
}

// prepare input date for MILP solver
func (v *MILPSolver) preProcess() {

	// create map and lookup arrays for accelerators
	accMap := core.GetAccelerators()
	v.numAccelerators = len(accMap)
	v.accIndex = make(map[string]int)
	v.accLookup = make([]string, v.numAccelerators)

	// set cost values
	v.instanceCost = make([]float64, v.numAccelerators)
	index := 0
	for accName, acc := range accMap {
		v.accIndex[accName] = index
		v.accLookup[index] = accName
		v.instanceCost[index] = float64(acc.Spec().Cost)
		index++
	}

	// fmt.Println(v.accIndex)
	// fmt.Println(v.accLookup)
	// fmt.Println(lpsolveUtils.Pretty1D("unitCost", v.instanceCost))

	// create map and lookup arrays for accelerator types
	capMap := core.GetCapacities()
	v.numAcceleratorTypes = len(capMap)
	v.accTypeIndex = make(map[string]int)
	v.accTypeLookup = make([]string, v.numAcceleratorTypes)

	// set available accelerator types
	v.unitsAvail = make([]int, v.numAcceleratorTypes)
	v.acceleratorTypesMatrix = make([][]int, v.numAcceleratorTypes)
	index = 0
	for accTypeName, accTypeCount := range capMap {
		v.accTypeIndex[accTypeName] = index
		v.accTypeLookup[index] = accTypeName
		v.unitsAvail[index] = accTypeCount
		v.acceleratorTypesMatrix[index] = make([]int, v.numAccelerators)
		index++
	}

	// set matrix of accelerator types to accelerators
	for accName, acc := range accMap {
		accType := acc.Type()
		if accIndex, exists := v.accIndex[accName]; exists {
			accTypeIndex := v.accTypeIndex[accType]
			v.acceleratorTypesMatrix[accTypeIndex][accIndex] = acc.Spec().Multiplicity
		}
	}

	// fmt.Println(v.accTypeIndex)
	// fmt.Println(v.accTypeLookup)
	// fmt.Println(lpsolveUtils.Pretty1D("unitsAvailByType", v.unitsAvail))
	// fmt.Println(lpsolveUtils.Pretty2D("acceleratorTypesMatrix", v.acceleratorTypesMatrix))

	// create map and lookup arrays for servers
	index = 0
	v.serverIndex = make(map[string]int)
	srvMap := core.GetServers()
	for srvName := range srvMap {
		v.serverIndex[srvName] = index
		index++
	}
	v.numServers = index
	v.serverLookup = make([]string, v.numServers)
	for srvName, index := range v.serverIndex {
		v.serverLookup[index] = srvName
	}

	// set values for arrival rates and per replica arrivals and number of instances
	v.arrivalRates = make([]float64, v.numServers)
	v.numInstancesPerReplica = make([][]int, v.numServers)
	v.ratePerReplica = make([][]float64, v.numServers)
	for i := 0; i < v.numServers; i++ {
		v.numInstancesPerReplica[i] = make([]int, v.numAccelerators)
		v.ratePerReplica[i] = make([]float64, v.numAccelerators)
	}
	modelMap := core.GetModels()
	for srvName, srv := range srvMap {
		if i, exists := v.serverIndex[srvName]; exists {
			load := srv.Load()
			if load == nil {
				continue
			}
			v.arrivalRates[i] = float64(load.ArrivalRate / 60 / 1000)
			m := modelMap[srv.ModelName()]
			for accName, j := range v.accIndex {
				//acc := accMap[accName]
				v.numInstancesPerReplica[i][j] = m.NumInstances(accName)
				if alloc := srv.AllAllocations()[accName]; alloc != nil {
					v.ratePerReplica[i][j] = float64(alloc.MaxArrvRatePerReplica())
				}
			}
		}

	}

	// fmt.Println(v.serverIndex)
	// fmt.Println(lpsolveUtils.Pretty1D("arrivalRates", v.arrivalRates))
	// fmt.Println(lpsolveUtils.Pretty2D("ratePerReplica", v.ratePerReplica))
	// fmt.Println(lpsolveUtils.Pretty2D("numInstancesPerReplica", v.numInstancesPerReplica))
}

// call MILP solver to optimize problem
func (v *MILPSolver) optimize(isLimited bool, isMulti bool, useCplex bool) error {
	problemType := lpsolveConfig.SINGLE
	if isMulti {
		problemType = lpsolveConfig.MULTI
	}
	p, err := v.createProblem(problemType, isLimited, useCplex)
	if err != nil {
		return err
	}
	if err := p.Solve(); err != nil {
		return err
	}
	v.printResults(problemType, p)
	return nil
}

func (v *MILPSolver) createProblem(problemType lpsolveConfig.ProblemType, isLimited bool, useCplex bool) (lpsolve.Problem, error) {
	// create a new problem instance
	var p lpsolve.Problem
	var err error
	switch problemType {
	case lpsolveConfig.SINGLE:
		if useCplex {
			p, err = lpsolve.CreateCplexProblem(v.numServers, v.numAccelerators, v.instanceCost, v.numInstancesPerReplica,
				v.ratePerReplica, v.arrivalRates)
		} else {
			p, err = lpsolve.CreateSingleAssignProblem(v.numServers, v.numAccelerators, v.instanceCost, v.numInstancesPerReplica,
				v.ratePerReplica, v.arrivalRates)
		}
	case lpsolveConfig.MULTI:
		if useCplex {
			p, err = lpsolve.CreateCplexProblem(v.numServers, v.numAccelerators, v.instanceCost, v.numInstancesPerReplica,
				v.ratePerReplica, v.arrivalRates)
		} else {
			p, err = lpsolve.CreateMultiAssignProblem(v.numServers, v.numAccelerators, v.instanceCost, v.numInstancesPerReplica,
				v.ratePerReplica, v.arrivalRates)
		}
	default:
		return nil, fmt.Errorf("unknown problem type: %s", problemType)
	}
	if err != nil {
		return nil, err
	}

	// set accelerator count limited option
	if isLimited {
		if err = p.SetLimited(v.numAcceleratorTypes, v.unitsAvail, v.acceleratorTypesMatrix); err != nil {
			return nil, err
		}
		if useCplex {
			switch problemType {
			case lpsolveConfig.SINGLE:
				SetFileNames(p, "single-limited")
			case lpsolveConfig.MULTI:
				SetFileNames(p, "multi-limited")
			}
		}
	} else {
		p.UnSetLimited()
		if useCplex {
			switch problemType {
			case lpsolveConfig.SINGLE:
				SetFileNames(p, "single-unlimited")
			case lpsolveConfig.MULTI:
				SetFileNames(p, "multi-unlimited")
			}
		}
	}
	return p, nil
}

func SetFileNames(p lpsolve.Problem, name string) {
	pc := p.(*lpsolve.CplexProblem)
	pc.SetModelFileName(name + ".mod")
	pc.SetDataFileName(name + ".dat")
	pc.SetOutputFileName(name + ".txt")
}

// print solution details
func (v *MILPSolver) printResults(problemType lpsolveConfig.ProblemType, p lpsolve.Problem) {
	fmt.Printf("Problem type: %v\n", problemType)
	fmt.Printf("Solution type: %v\n", p.GetSolutionType())
	fmt.Printf("Solution time: %d msec\n", p.GetSolutionTimeMsec())
	fmt.Printf("Objective value: %v\n", p.GetObjectiveValue())

	fmt.Println()
	fmt.Printf("Accelerators=%v \n", v.accLookup)
	fmt.Printf("Servers=%v \n", v.serverLookup)
	fmt.Println()

	v.numReplicas = p.GetNumReplicas()
	fmt.Println(lpsolveUtils.Pretty2D("numReplicas", v.numReplicas))

	v.instancesUsed = p.GetInstancesUsed()
	fmt.Println(lpsolveUtils.Pretty1D("instancesUsed", v.instancesUsed))

	if p.IsLimited() {
		fmt.Printf("AcceleratorTypes=%v \n", v.accTypeLookup)
		fmt.Println(lpsolveUtils.Pretty1D("unitsAvail", v.unitsAvail))
		v.unitsUsed = p.GetUnitsUsed()
		fmt.Println(lpsolveUtils.Pretty1D("unitsUsed", v.unitsUsed))
	}
	fmt.Println()
}

// process output data from MILP solver
func (v *MILPSolver) postProcess() {
	for i := 0; i < v.numServers; i++ {
		for j := 0; j < v.numAccelerators; j++ {
			n := v.numReplicas[i][j]
			if n == 0 {
				continue
			}
			accName := v.accLookup[j]
			sc := core.GetServer(v.serverLookup[i])
			// TODO: Fix this
			if alloc := sc.AllAllocations()[accName]; alloc != nil {
				sc.SetAllocation(alloc)
			}
		}
	}
}
