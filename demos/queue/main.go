package main

import (
	"fmt"

	"github.com/llm-inferno/optimizer/pkg/analyzer"
)

func main() {

	// queue configuration
	maxBatchSize := 256
	maxQueueSize := 100

	// prefill and decode parameters
	gamma := float32(86.615)
	delta := float32(1.446e-03)
	alpha := float32(6.958)
	beta := float32(0.042)

	// request rate
	requestRate := float32(25)

	// request size
	avgInputTokens := 128
	avgOutputTokens := 512

	// target values
	targetTTFT := float32(120)
	targetITL := float32(14)
	targetTPS := float32(20 * 512)

	// create queue analyzer
	config := &analyzer.Configuration{
		MaxBatchSize: maxBatchSize,
		MaxQueueSize: maxQueueSize,
		ServiceParms: &analyzer.ServiceParms{
			Prefill: &analyzer.PrefillParms{
				Gamma: gamma,
				Delta: delta,
			},
			Decode: &analyzer.DecodeParms{
				Alpha: alpha,
				Beta:  beta,
			},
		},
	}

	requestSize := &analyzer.RequestSize{
		AvgInputTokens:  avgInputTokens,
		AvgOutputTokens: avgOutputTokens,
	}

	targetPerf := &analyzer.TargetPerf{
		TargetTTFT: targetTTFT,
		TargetITL:  targetITL,
		TargetTPS:  targetTPS,
	}

	fmt.Println()
	fmt.Printf("configuration=%v\n", config)
	fmt.Printf("requestSize=%v\n", requestSize)
	fmt.Println()

	queueAnalyzer, err := analyzer.NewQueueAnalyzer(config, requestSize)
	if err != nil {
		fmt.Printf("NewQueueAnalyzer() failed: %v\n", err)
		return
	}

	// analyze queue under a given load
	fmt.Printf("Analyzing at request rate = %v req/sec ...\n", requestRate)
	metrics, err := queueAnalyzer.Analyze(requestRate)
	if err != nil {
		fmt.Printf("AnalyzeQueue() %v\n", err)
		return
	}
	fmt.Printf("model=%v\n", queueAnalyzer.Model)
	fmt.Printf("metrics=%v\n", metrics)
	fmt.Println()

	// size queue for given targets
	fmt.Printf("Sizing for targets ...\n")
	fmt.Printf("targetPerf=%v\n", targetPerf)
	if targetRate, metrics, targetPerf, err := queueAnalyzer.Size(targetPerf); err != nil {
		fmt.Printf("Size() %v\n", err)
		return
	} else {
		fmt.Printf("achieved=%v\n", targetPerf)
		fmt.Printf("targetRate=%v\n", targetRate)
		fmt.Printf("metrics=%v\n", metrics)
		fmt.Println()
	}
}
