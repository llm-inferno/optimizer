package main

import (
	"fmt"

	"github.com/llm-inferno/optimizer/pkg/analyzer"
)

func main() {

	// queue configuration
	maxBatchSize := 256
	maxQueueSize := 100

	// performance parameters
	alpha := float32(6.958)
	beta := float32(0.042)
	gamma := float32(0.001)

	// request rate
	requestRate := float32(25)

	// request size
	avgInputTokens := float32(128)
	avgOutputTokens := float32(512)

	// target values
	targetTTFT := float32(120)
	targetITL := float32(14)
	targetTPS := float32(20 * 512)

	// create queue analyzer
	config := &analyzer.Configuration{
		MaxBatchSize: maxBatchSize,
		MaxNumTokens: analyzer.DefaultMaxNumTokens,
		MaxQueueSize: maxQueueSize,
		ServiceParms: &analyzer.ServiceParms{
			Alpha: alpha,
			Beta:  beta,
			Gamma: gamma,
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

	queueAnalyzer, err := analyzer.NewLLMQueueAnalyzer(config, requestSize)
	if err != nil {
		fmt.Printf("NewLLMQueueAnalyzer() failed: %v\n", err)
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
