package config

import (
	"math"
)

/**
 * Parameters
 */

// Tolerated percentile for SLOs
var SLOPercentile = 0.95

// Multiplier of average of exponential distribution to attain percentile
var SLOMargin = -float32(math.Log(1 - SLOPercentile))

// small disturbance around a value
var Delta = float32(0.001)

// maximum number of requests in queueing system as multiples of maximum batch size
var MaxQueueToBatchRatio = 10

// accelerator transition penalty factor
var AccelPenaltyFactor = float32(0.1)

// default name of a service class
const DefaultServiceClassName string = "Free"

// default priority of a service class
const DefaultServiceClassPriority int = 0

// default option for allocation under saturated condition
var DefaultSaturatedAllocationPolicy SaturatedAllocationPolicy = None

// fraction of maximum server throughput to provide stability (running this fraction below the maximum)
var StabilitySafetyFraction float32 = 0.1
