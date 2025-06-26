package core

import (
	"fmt"

	"github.com/llm-inferno/optimizer/pkg/config"
)

// A service class
type ServiceClass struct {
	name     string             // unique name
	priority int                // non-negative priority (smaller values for higher priority)
	targets  map[string]*Target // target SLOs for each model
}

// target SLOs for service class
type Target struct {
	ITL float32
	TTW float32
	TPS float32
}

func (t *Target) String() string {
	return fmt.Sprintf("[ITL=%v, TTW=%v, TPS=%v]",
		t.ITL, t.TTW, t.TPS)
}

func NewServiceClass(name string, priority int) *ServiceClass {
	if priority < 0 {
		priority = config.DefaultServiceClassPriority
	}
	return &ServiceClass{
		name:     name,
		priority: priority,
		targets:  map[string]*Target{},
	}
}

// set target SLOs for a model in a service class (replace if already exists)
func (c *ServiceClass) SetTargetFromSpec(spec *config.ServiceClassSpec) {
	if spec.Name == c.name {
		c.targets[spec.Model] = &Target{
			ITL: spec.SLO_ITL,
			TTW: spec.SLO_TTW,
			TPS: spec.SLO_TPS,
		}
	}
}

func (c *ServiceClass) Name() string {
	return c.name
}

func (c *ServiceClass) Priority() int {
	return c.priority
}

func (c *ServiceClass) ModelTarget(modelName string) *Target {
	return c.targets[modelName]
}

func (c *ServiceClass) RemoveModelTarget(modelName string) {
	delete(c.targets, modelName)
}

func (c *ServiceClass) Spec() []config.ServiceClassSpec {
	specs := make([]config.ServiceClassSpec, len(c.targets))
	i := 0
	for modelName, target := range c.targets {
		specs[i] = config.ServiceClassSpec{
			Name:     c.name,
			Priority: c.priority,
			Model:    modelName,
			SLO_ITL:  target.ITL,
			SLO_TTW:  target.TTW,
			SLO_TPS:  target.TPS,
		}
		i++
	}
	return specs
}

func (c *ServiceClass) String() string {
	return fmt.Sprintf("ServiceClass: name=%s; priority=%d; targets=%v",
		c.name, c.priority, c.targets)
}
