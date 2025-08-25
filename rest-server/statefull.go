package rest

// A statefull REST server with many GET and many POST API calls
type StateFullServer struct {
	BaseServer
}

// create a statefull REST server
func NewStateFullServer() *StateFullServer {
	server := &StateFullServer{
		BaseServer: *NewBaseServer(),
	}

	server.router.POST("/setAccelerators", setAccelerators)
	server.router.GET("/getAccelerators", getAccelerators)
	server.router.GET("/getAccelerator/:name", getAccelerator)
	server.router.POST("/addAccelerator", addAccelerator)
	server.router.GET("/removeAccelerator/:name", removeAccelerator)

	server.router.POST("/setCapacities", setCapacities)
	server.router.GET("/getCapacities", getCapacities)
	server.router.GET("/getCapacity/:type", getCapacity)
	server.router.POST("/setCapacity", setCapacity)
	server.router.GET("/removeCapacity/:type", removeCapacity)

	server.router.POST("/setModels", setModels)
	server.router.GET("/getModels", getModels)
	server.router.GET("/getModel/:name", getModel)
	server.router.GET("/addModel/:name", addModel)
	server.router.GET("/removeModel/:name", removeModel)

	server.router.POST("/setServiceClasses", setServiceClasses)
	server.router.GET("/getServiceClasses", getServiceClasses)
	server.router.GET("/getServiceClass/:name", getServiceClass)
	server.router.GET("/addServiceClass/:name/:priority", addServiceClass)
	server.router.GET("/removeServiceClass/:name", removeServiceClass)

	server.router.POST("/addServiceClassModelTargets", addServiceClassModelTargets)
	server.router.GET("/getServiceClassModelTarget/:name/:model", getServiceClassModelTarget)
	server.router.GET("/removeServiceClassModelTarget/:name/:model", removeServiceClassModelTarget)

	server.router.POST("/setServers", setServers)
	server.router.GET("/getServers", getServers)
	server.router.GET("/getServer/:name", getServer)
	server.router.POST("/addServer", addServer)
	server.router.GET("/removeServer/:name", removeServer)

	server.router.GET("/getModelAcceleratorPerf/:name/:acc", getModelAcceleratorPerf)
	server.router.POST("/addModelAcceleratorPerf", addModelAcceleratorPerf)
	server.router.GET("/removeModelAcceleratorPerf/:name/:acc", removeModelAcceleratorPerf)

	server.router.POST("/optimize", optimize)
	server.router.POST("/optimizeOne", optimizeOne)
	server.router.GET("/applyAllocation", applyAllocation)

	return server
}
