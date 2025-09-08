package main

import (
	"fmt"
	"os"

	"github.com/llm-inferno/optimizer/pkg/config"
	"github.com/llm-inferno/optimizer/pkg/core"
	"github.com/llm-inferno/optimizer/pkg/manager"
	"github.com/llm-inferno/optimizer/pkg/solver"
	"github.com/llm-inferno/optimizer/pkg/utils"
)

func main() {
	size := "large"
	if len(os.Args) > 1 {
		size = os.Args[1]
	}
	prefix := "../../sample-data/" + size + "/"
	fn_acc := prefix + "accelerator-data.json"
	fn_cap := prefix + "capacity-data.json"
	fn_mod := prefix + "model-data.json"
	fn_svc := prefix + "serviceclass-data.json"
	fn_srv := prefix + "server-data.json"
	fn_opt := prefix + "optimizer-data.json"

	system := core.NewSystem()

	bytes_acc, err_acc := os.ReadFile(fn_acc)
	if err_acc != nil {
		fmt.Println(err_acc)
	}
	if d, err := utils.FromDataToSpec(bytes_acc, config.AcceleratorData{}); err == nil {
		system.SetAcceleratorsFromSpec(d)
	} else {
		fmt.Println(err)
		return
	}

	bytes_cap, err_cap := os.ReadFile(fn_cap)
	if err_cap != nil {
		fmt.Println(err_cap)
	}
	if d, err := utils.FromDataToSpec(bytes_cap, config.CapacityData{}); err == nil {
		system.SetCapacityFromSpec(d)
	} else {
		fmt.Println(err)
		return
	}

	bytes_mod, err_mod := os.ReadFile(fn_mod)
	if err_mod != nil {
		fmt.Println(err_mod)
	}
	if d, err := utils.FromDataToSpec(bytes_mod, config.ModelData{}); err == nil {
		system.SetModelsFromSpec(d)
	} else {
		fmt.Println(err)
		return
	}

	bytes_svc, err_svc := os.ReadFile(fn_svc)
	if err_svc != nil {
		fmt.Println(err_svc)
	}
	if d, err := utils.FromDataToSpec(bytes_svc, config.ServiceClassData{}); err == nil {
		system.SetServiceClassesFromSpec(d)
	} else {
		fmt.Println(err)
		return
	}

	bytes_srv, err_srv := os.ReadFile(fn_srv)
	if err_srv != nil {
		fmt.Println(err_srv)
	}
	if d, err := utils.FromDataToSpec(bytes_srv, config.ServerData{}); err == nil {
		system.SetServersFromSpec(d)
	} else {
		fmt.Println(err)
		return
	}

	var optimizer *solver.Optimizer
	bytes_opt, err_opt := os.ReadFile(fn_opt)
	if err_opt != nil {
		fmt.Println(err_acc)
	}
	if d, err := utils.FromDataToSpec(bytes_opt, config.OptimizerData{}); err == nil {
		optimizer = solver.NewOptimizerFromSpec(&d.Spec)
	} else {
		fmt.Println(err)
		return
	}

	manager := manager.NewManager(system, optimizer)

	system.Calculate()
	if err := manager.Optimize(); err != nil {
		fmt.Println(err)
		return
	}

	serverName := "Premium-llama3_8b"

	server := system.Server(serverName)
	if server == nil {
		fmt.Printf("No server %s\n", serverName)
		return
	}
	allocBefore := server.Allocation()
	if allocBefore == nil {
		fmt.Printf("No allocation for server %s \n", serverName)
		return
	}
	// change load on server
	load := server.Load()
	if load == nil {
		fmt.Printf("No model load data for server %s \n", serverName)
		return
	}
	fmt.Println("AllocBefore: ", allocBefore)
	newArv := load.ArrivalRate * 2.5
	newOutTokens := int(float32(load.AvgOutTokens) * 1.5)
	newInTokens := int(float32(load.AvgInTokens) * 1.5)
	newLoad := config.ServerLoadSpec{
		ArrivalRate:  newArv,
		AvgInTokens:  newInTokens,
		AvgOutTokens: newOutTokens,
	}
	server.SetLoad(&newLoad)

	// scale allocation
	allocAfter, inc := allocBefore.Scale(serverName)
	fmt.Println("AllocAfter: ", allocAfter)
	fmt.Println("Inc: ", inc)

	// reallocate
	var gName string
	allocAfter, gName = allocBefore.ReAllocate(serverName)
	fmt.Println("AllocAfter: ", allocAfter)
	fmt.Println("gName: ", gName)
}
