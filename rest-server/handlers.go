package rest

import (
	"fmt"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"github.com/llm-inferno/optimizer/pkg/config"
	"github.com/llm-inferno/optimizer/pkg/core"
	"github.com/llm-inferno/optimizer/pkg/manager"
	"github.com/llm-inferno/optimizer/pkg/solver"
)

// Handlers for REST API calls

func setAccelerators(c *gin.Context) {
	var acceleratorData config.AcceleratorData
	if err := c.BindJSON(&acceleratorData); err != nil {
		return
	}
	system.SetAcceleratorsFromSpec(&acceleratorData)
	c.IndentedJSON(http.StatusOK, acceleratorData)
}

func getAccelerators(c *gin.Context) {
	accMap := system.Accelerators()
	gpus := make([]config.AcceleratorSpec, len(accMap))
	i := 0
	for _, acc := range accMap {
		gpus[i] = *acc.Spec()
		i++
	}
	c.IndentedJSON(http.StatusOK, gpus)
}

func getAccelerator(c *gin.Context) {
	name := c.Param("name")
	acc := system.Accelerator(name)
	if acc == nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "accelerator " + name + " not found"})
		return
	}
	c.IndentedJSON(http.StatusOK, acc.Spec())
}

func addAccelerator(c *gin.Context) {
	var acc config.AcceleratorSpec
	if err := c.BindJSON(&acc); err != nil {
		return
	}
	system.AddAcceleratorFromSpec(acc)
	c.IndentedJSON(http.StatusOK, acc)
}

func removeAccelerator(c *gin.Context) {
	name := c.Param("name")
	acc := system.Accelerator(name)
	if err := system.RemoveAccelerator(name); err != nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "accelerator " + name + " not found"})
		return
	}
	c.IndentedJSON(http.StatusOK, acc.Spec())
}

func setCapacities(c *gin.Context) {
	var capacityData config.CapacityData
	if err := c.BindJSON(&capacityData); err != nil {
		return
	}
	system.SetCapacityFromSpec(&capacityData)
	c.IndentedJSON(http.StatusOK, capacityData)
}

func getCapacities(c *gin.Context) {
	capMap := system.Capacities()
	capacities := make([]config.AcceleratorCount, len(capMap))
	i := 0
	for k, v := range capMap {
		capacities[i] = config.AcceleratorCount{
			Type:  k,
			Count: v,
		}
		i++
	}
	c.IndentedJSON(http.StatusOK, config.CapacityData{
		Count: capacities,
	})
}

func getCapacity(c *gin.Context) {
	t := c.Param("type")
	cap, exists := system.Capacity(t)
	if !exists {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "capacity for " + t + " not found"})
		return
	}
	c.IndentedJSON(http.StatusOK, config.AcceleratorCount{
		Type:  t,
		Count: cap,
	})
}

func setCapacity(c *gin.Context) {
	var count config.AcceleratorCount
	if err := c.BindJSON(&count); err != nil {
		return
	}
	system.SetCountFromSpec(count)
	c.IndentedJSON(http.StatusOK, count)
}

func removeCapacity(c *gin.Context) {
	t := c.Param("type")
	cap, _ := system.Capacity(t)
	if !system.RemoveCapacity(t) {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "accelerator type " + t + " not found"})
		return
	}
	c.IndentedJSON(http.StatusOK, config.AcceleratorCount{
		Type:  t,
		Count: cap,
	})
}

func setModels(c *gin.Context) {
	var modelData config.ModelData
	if err := c.BindJSON(&modelData); err != nil {
		return
	}
	system.SetModelsFromSpec(&modelData)
	c.IndentedJSON(http.StatusOK, modelData)
}

func getModels(c *gin.Context) {
	modelMap := system.Models()
	modelNames := make([]string, len(modelMap))
	i := 0
	for _, model := range modelMap {
		modelNames[i] = model.Name()
		i++
	}
	c.IndentedJSON(http.StatusOK, modelNames)
}

func getModel(c *gin.Context) {
	name := c.Param("name")
	model := system.Model(name)
	if model == nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "model " + name + " not found"})
		return
	}
	c.IndentedJSON(http.StatusOK, model.Spec())
}

func addModel(c *gin.Context) {
	name := c.Param("name")
	system.AddModel(name)
	c.IndentedJSON(http.StatusOK, name)
}

func removeModel(c *gin.Context) {
	name := c.Param("name")
	if err := system.RemoveModel(name); err != nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "model " + name + " not found"})
		return
	}
	c.IndentedJSON(http.StatusOK, name)
}

func setServiceClasses(c *gin.Context) {
	var serviceClassData config.ServiceClassData
	if err := c.BindJSON(&serviceClassData); err != nil {
		return
	}
	system.SetServiceClassesFromSpec(&serviceClassData)
	c.IndentedJSON(http.StatusOK, serviceClassData)
}

func getServiceClasses(c *gin.Context) {
	svcMap := system.ServiceClasses()
	svcs := &config.ServiceClassData{
		Spec: []config.ServiceClassSpec{},
	}
	for _, svc := range svcMap {
		svcs.Spec = append(svcs.Spec, svc.Spec())
	}
	c.IndentedJSON(http.StatusOK, svcs)
}

func getServiceClass(c *gin.Context) {
	name := c.Param("name")
	svc := system.ServiceClass(name)
	if svc == nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "service class " + name + " not found"})
		return
	}
	c.IndentedJSON(http.StatusOK, svc.Spec())
}

func addServiceClass(c *gin.Context) {
	name := c.Param("name")
	priority := config.DefaultServiceClassPriority
	if prioStr := c.Param("priority"); prioStr != "" {
		if prioInt, err := strconv.Atoi(prioStr); err != nil {
			c.IndentedJSON(http.StatusBadRequest, gin.H{"message": "service class priority " + prioStr + " invalid"})
			return
		} else {
			priority = prioInt
		}
	}
	system.AddServiceClass(name, priority)
	svc := system.ServiceClass(name)
	c.IndentedJSON(http.StatusOK, svc.Spec())
}

func removeServiceClass(c *gin.Context) {
	name := c.Param("name")
	svc := system.ServiceClass(name)
	if err := system.RemoveServiceClass(name); err != nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "service class " + name + " not found"})
		return
	}
	c.IndentedJSON(http.StatusOK, svc.Spec())
}

func addServiceClassModelTargets(c *gin.Context) {
	var svcSpec config.ServiceClassSpec
	if err := c.BindJSON(&svcSpec); err != nil {
		return
	}
	svcName := svcSpec.Name
	svc := system.ServiceClass(svcName)
	if svc == nil {
		c.IndentedJSON(http.StatusNotFound, "service class "+svcName+" not found")
		return
	}
	if !svc.UpdateModelTargets(&svcSpec) {
		c.IndentedJSON(http.StatusBadRequest, "inconsistent specs: svcName="+svcName+" ; svcPrio="+strconv.Itoa(svcSpec.Priority))
		return
	}
	c.IndentedJSON(http.StatusOK, svc.Spec())
}

func getServiceClassModelTarget(c *gin.Context) {
	name := c.Param("name")
	model := c.Param("model")
	svc := system.ServiceClass(name)
	if svc == nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "service class " + name + " not found"})
		return
	}
	target := svc.ModelTarget(model)
	if target == nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "model " + model + " not found"})
		return
	}
	c.IndentedJSON(http.StatusOK, config.ModelTarget{
		Model:    model,
		SLO_ITL:  target.ITL,
		SLO_TTFT: target.TTFT,
		SLO_TPS:  target.TPS,
	})
}

func removeServiceClassModelTarget(c *gin.Context) {
	name := c.Param("name")
	model := c.Param("model")
	svc := system.ServiceClass(name)
	if svc == nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "service class " + name + " not found"})
		return
	}
	target := svc.ModelTarget(model)
	if target == nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "model " + model + " not found"})
		return
	}
	svc.RemoveModelTarget(model)
	c.IndentedJSON(http.StatusOK, config.ModelTarget{
		Model:    model,
		SLO_ITL:  target.ITL,
		SLO_TTFT: target.TTFT,
		SLO_TPS:  target.TPS,
	})
}

func setServers(c *gin.Context) {
	var serverData config.ServerData
	if err := c.BindJSON(&serverData); err != nil {
		return
	}
	system.SetServersFromSpec(&serverData)
	c.IndentedJSON(http.StatusOK, serverData)
}

func getServers(c *gin.Context) {
	srvMap := system.Servers()
	servers := make([]config.ServerSpec, len(srvMap))
	i := 0
	for _, server := range srvMap {
		servers[i] = *server.Spec()
		i++
	}
	serverData := &config.ServerData{
		Spec: servers,
	}
	c.IndentedJSON(http.StatusOK, serverData)
}

func getServer(c *gin.Context) {
	name := c.Param("name")
	server := system.Server(name)
	if server == nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "server " + name + " not found"})
		return
	}
	c.IndentedJSON(http.StatusOK, server.Spec())
}

func addServer(c *gin.Context) {
	var server config.ServerSpec
	if err := c.BindJSON(&server); err != nil {
		return
	}
	system.AddServerFromSpec(server)
	c.IndentedJSON(http.StatusOK, server)
}

func removeServer(c *gin.Context) {
	name := c.Param("name")
	server := system.Server(name)
	if err := system.RemoveServer(name); err != nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "server " + name + " not found"})
		return
	}
	c.IndentedJSON(http.StatusOK, server.Spec())
}

func getModelAcceleratorPerf(c *gin.Context) {
	name := c.Param("name")
	acc := c.Param("acc")
	model := system.Model(name)
	if model == nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "model " + name + " not found"})
		return
	}
	perfData := model.PerfData(acc)
	if perfData == nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "accelerator " + acc + " not found"})
		return
	}
	c.IndentedJSON(http.StatusOK, perfData)
}

func addModelAcceleratorPerf(c *gin.Context) {
	var perfData config.ModelAcceleratorPerfData
	if err := c.BindJSON(&perfData); err != nil {
		return
	}
	modelName := perfData.Name
	model := system.Model(modelName)
	if model == nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "model " + modelName + " not found"})
		return
	}
	model.AddPerfDataFromSpec(&perfData)
	c.IndentedJSON(http.StatusOK, perfData)
}

func removeModelAcceleratorPerf(c *gin.Context) {
	name := c.Param("name")
	acc := c.Param("acc")
	model := system.Model(name)
	if model == nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "model " + name + " not found"})
		return
	}
	perfData := model.PerfData(acc)
	if perfData == nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "accelerator " + acc + " not found"})
		return
	}
	model.RemovePerfData(acc)
	c.IndentedJSON(http.StatusOK, perfData)
}

func optimize(c *gin.Context) {
	var optimizerSpec config.OptimizerSpec
	if err := c.BindJSON(&optimizerSpec); err != nil {
		return
	}
	optimizer := solver.NewOptimizerFromSpec(&optimizerSpec)
	manager := manager.NewManager(system, optimizer)
	system.Calculate()
	if err := manager.Optimize(); err != nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "optimization error: " + err.Error()})
		return
	}
	solution := system.GenerateSolution()
	fmt.Println(system)
	c.IndentedJSON(http.StatusOK, solution)
}

func optimizeOne(c *gin.Context) {
	var systemData config.SystemData
	if err := c.BindJSON(&systemData); err != nil {
		return
	}
	// start with fresh system
	system = core.NewSystem()
	optimizerSpec := system.SetFromSpec(&systemData.Spec)
	optimizer := solver.NewOptimizerFromSpec(optimizerSpec)
	manager := manager.NewManager(system, optimizer)
	system.Calculate()
	if err := manager.Optimize(); err != nil {
		c.IndentedJSON(http.StatusNotFound, gin.H{"message": "optimization error: " + err.Error()})
		return
	}
	solution := system.GenerateSolution()
	fmt.Println(system)
	c.IndentedJSON(http.StatusOK, solution)
}

func applyAllocation(c *gin.Context) {
	for _, server := range system.Servers() {
		server.ApplyDesiredAlloc()
	}
	c.IndentedJSON(http.StatusOK, "Done")
}
