package analyzer

import (
	"fmt"

	"github.com/llm-inferno/queue-analysis/pkg/queue"
	utils "github.com/llm-inferno/queue-analysis/pkg/utils"
)

// small disturbance around a value
const Epsilon = float32(0.001)

// fraction of maximum server throughput to provide stability (running this fraction below the maximum)
const StabilitySafetyFraction = float32(0.1)

// maximum number of tokens per batch (iteration)
const DefaultMaxNumTokens = 8192

// Analyzer of inference server queue
type LLMQueueAnalyzer struct {
	MaxBatchSize int                           // maximum batch size
	MaxNumTokens int                           // maximum number of tokens per batch
	MaxQueueSize int                           // maximum queue size
	ServiceParms *ServiceParms                 // request processing parameters
	RequestSize  *RequestSize                  // number of input and output tokens per request
	Model        *queue.MM1ModelStateDependent // queueing model
	RateRange    *RateRange                    // range of request rates for model stability
}

// queue configuration parameters
type Configuration struct {
	MaxBatchSize int           // maximum batch size (limit on the number of requests concurrently receiving service >0)
	MaxNumTokens int           // maximum number of tokens per batch (limit on the number of tokens per batch >0)
	MaxQueueSize int           // maximum queue size (limit on the number of requests queued for servive >=0)
	ServiceParms *ServiceParms // request processing parameters
}

// request processing parameters:
// iterationTime = alpha + beta * computeTime + gamma * memoryAccessTime
type ServiceParms struct {
	Alpha float32 // base
	Beta  float32 // slope for compute time
	Gamma float32 // slope for memory access time
}

// request tokens data
type RequestSize struct {
	AvgInputTokens  float32 // average number of input tokens per request
	AvgOutputTokens float32 // average number of output tokens per request
}

// range of request rates (requests/sec)
type RateRange struct {
	Min float32 // lowest rate (slightly larger than zero)
	Max float32 // highest rate (slightly less than maximum service rate)
}

// analysis solution metrics data
type AnalysisMetrics struct {
	Throughput     float32 // effective throughput (requests/sec)
	AvgRespTime    float32 // average request response time (aka latency) (msec)
	AvgWaitTime    float32 // average request queueing time (msec)
	AvgNumInServ   float32 // average number of requests in service
	AvgPrefillTime float32 // average request prefill time (msec)
	AvgTokenTime   float32 // average token decode time (msec)
	AvgTTFT        float32 // average time to first token (msec)
	MaxRate        float32 // maximum throughput (requests/sec)
	Rho            float32 // utilization
}

// queue performance targets
type TargetPerf struct {
	TargetTTFT float32 // target time to first token (queueing + prefill) (msec)
	TargetITL  float32 // target inter-token latency (msec)
	TargetTPS  float32 // target token generation throughtput (tokens/sec)
}

// queue max request rates to achieve performance targets
type TargetRate struct {
	RateTargetTTFT float32 // max request rate for target TTFT (requests/sec)
	RateTargetITL  float32 // max request rate for target ITL (requests/sec)
	RateTargetTPS  float32 // max request rate for target TPS (requests/sec)
}

// create a new queue analyzer from config
func NewLLMQueueAnalyzer(qConfig *Configuration, requestSize *RequestSize) (*LLMQueueAnalyzer, error) {
	if err := qConfig.check(); err != nil {
		return nil, err
	}
	if err := requestSize.check(); err != nil {
		return nil, err
	}
	// build queueing model
	return BuildModel(qConfig, requestSize), nil
}

// build queueing model using service rates, leaving arrival rate as parameter
func BuildModel(c *Configuration, r *RequestSize) (modelData *LLMQueueAnalyzer) {
	parms := c.ServiceParms

	// calculate state-dependent service rate
	servRate := make([]float32, c.MaxBatchSize)
	for n := 1; n <= c.MaxBatchSize; n++ {
		prefillTime := parms.PrefillTime(r, float32(n))
		decodeTime := r.AvgOutputTokens * parms.DecodeTime(r, float32(n))
		servRate[n-1] = float32(n) / (prefillTime + decodeTime)
	}

	// set and check limits
	lambdaMin := servRate[0] * Epsilon
	lambdaMax := servRate[c.MaxBatchSize-1] * (1 - Epsilon)
	rateRange := &RateRange{Min: lambdaMin * 1000, Max: lambdaMax * 1000}

	// create and solve model
	occupancyUpperBound := c.MaxQueueSize + c.MaxBatchSize
	model := queue.NewMM1ModelStateDependent(occupancyUpperBound, servRate)

	return &LLMQueueAnalyzer{
		MaxBatchSize: c.MaxBatchSize,
		MaxNumTokens: c.MaxNumTokens,
		MaxQueueSize: c.MaxQueueSize,
		ServiceParms: parms,
		RequestSize:  r,
		Model:        model,
		RateRange:    rateRange,
	}
}

// evaluate performance metrics given request rate
func (qa *LLMQueueAnalyzer) Analyze(requestRate float32) (metrics *AnalysisMetrics, err error) {
	if requestRate <= 0 {
		return nil, fmt.Errorf("invalid request rate %v", requestRate)
	}
	model := qa.Model
	rateRange := qa.RateRange
	if requestRate > rateRange.Max {
		err = fmt.Errorf("rate=%v, max allowed rate=%v", requestRate, rateRange.Max)
		return nil, err
	}

	//solve model
	model.Solve(requestRate/1000, 1)
	if !model.IsValid() {
		err = fmt.Errorf("invalid model %s", model)
		return nil, err
	}

	// get statistics
	avgNumInServ := model.GetAvgNumInServers()
	avgPrefillTime := qa.ServiceParms.PrefillTime(qa.RequestSize, avgNumInServ)
	avgDecodeTime := (model.GetAvgServTime() - avgPrefillTime) / qa.RequestSize.AvgOutputTokens
	avgTTFT := model.GetAvgWaitTime() + avgPrefillTime + avgDecodeTime

	rho := avgNumInServ / float32(qa.MaxBatchSize)
	rho = min(max(rho, 0), 1)

	// return solution
	metrics = &AnalysisMetrics{
		Throughput:     model.GetThroughput() * 1000,
		AvgRespTime:    model.GetAvgRespTime(),
		AvgWaitTime:    model.GetAvgWaitTime(),
		AvgNumInServ:   avgNumInServ,
		AvgPrefillTime: avgPrefillTime,
		AvgTokenTime:   avgDecodeTime,
		AvgTTFT:        avgTTFT,
		MaxRate:        rateRange.Max,
		Rho:            rho,
	}
	return metrics, nil
}

// model and parameters used in functional evaluation
type evalFuncData struct {
	model        *queue.MM1ModelStateDependent
	requestSize  *RequestSize
	serviceParms *ServiceParms
	maxBatchSize int
}

// evaluate max request rates to achieve a given target performance, returns
//   - max request rates
//   - performance metrics at min of max request rates
//   - achieved values of targets
func (qa *LLMQueueAnalyzer) Size(targetPerf *TargetPerf) (targetRate *TargetRate, metrics *AnalysisMetrics, achieved *TargetPerf, err error) {
	if err := targetPerf.check(); err != nil {
		return nil, nil, nil, err
	}
	targetTTFT := targetPerf.TargetTTFT
	targetITL := targetPerf.TargetITL
	targetTPS := targetPerf.TargetTPS

	lambdaMin := qa.RateRange.Min / 1000
	lambdaMax := qa.RateRange.Max / 1000

	var ind int

	// find max rate to achieve target TTFT time
	lambdaStarTTFT := lambdaMax
	if targetTTFT > 0 {
		evalTTFT := makeTTFTEval(&evalFuncData{
			model:        qa.Model,
			requestSize:  qa.RequestSize,
			serviceParms: qa.ServiceParms,
			maxBatchSize: qa.MaxBatchSize,
		})
		lambdaStarTTFT, ind, err = utils.BinarySearch(lambdaMin, lambdaMax, targetTTFT, evalTTFT)
		if ind < 0 {
			err = fmt.Errorf("target is below the bounded region")
		}
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to calculate lambdaStarTTFT, targetTTFT=%v, range=%s, ind=%d, err=%v",
				targetTTFT, qa.RateRange, ind, err)
		}
	}

	// find max rate to achieve target ITL time
	lambdaStarITL := lambdaMax
	if targetITL > 0 {
		evalITL := makeITLEval(&evalFuncData{
			model:        qa.Model,
			requestSize:  qa.RequestSize,
			serviceParms: qa.ServiceParms,
			maxBatchSize: qa.MaxBatchSize,
		})
		lambdaStarITL, ind, err = utils.BinarySearch(lambdaMin, lambdaMax, targetITL, evalITL)
		if ind < 0 {
			err = fmt.Errorf("target is below the bounded region")
		}
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to calculate lambdaStarITL, targetITL=%v, range=%s, ind=%d, err=%v",
				targetITL, qa.RateRange, ind, err)
		}
	}

	// find max rate to achieve target TPS
	lambdaStarTPS := lambdaMax
	if targetTPS > 0 {
		lambdaStarTPS = lambdaMax * (1 - StabilitySafetyFraction)
	}

	// analyze queue with smaller of rates
	lambda := min(lambdaStarTTFT, lambdaStarITL, lambdaStarTPS)
	requestRate := lambda * 1000 // convert to per-second rate
	if metrics, err = qa.Analyze(requestRate); err != nil {
		return nil, nil, nil, err
	}

	targetRate = &TargetRate{
		RateTargetTTFT: lambdaStarTTFT * 1000,
		RateTargetITL:  lambdaStarITL * 1000,
		RateTargetTPS:  lambdaStarTPS * 1000,
	}

	achieved = &TargetPerf{
		TargetTTFT: metrics.AvgTTFT,
		TargetITL:  metrics.AvgTokenTime,
		TargetTPS:  metrics.Throughput * qa.RequestSize.AvgOutputTokens,
	}
	return targetRate, metrics, achieved, nil
}

// Average iteration time as a function of the batch size T(n)
func (p *ServiceParms) IterationTime(r *RequestSize, batchSize float32) float32 {
	tokensCompute := (r.AvgInputTokens + r.AvgOutputTokens) / (r.AvgOutputTokens + 1)
	tokensMemory := r.AvgInputTokens + r.AvgOutputTokens/2
	return p.Alpha + batchSize*(p.Beta*tokensCompute+p.Gamma*tokensMemory)
}

// Average prefill time as a function of the batch size
func (p *ServiceParms) PrefillTime(r *RequestSize, batchSize float32) float32 {
	if r.AvgInputTokens == 0 {
		return 0
	}
	return p.IterationTime(r, batchSize) + (p.Beta+p.Gamma)*r.AvgInputTokens
}

// Average decode time (generation of one token) as a function of the batch size
func (p *ServiceParms) DecodeTime(r *RequestSize, batchSize float32) float32 {
	return p.IterationTime(r, batchSize) +
		p.Beta + p.Gamma*(r.AvgInputTokens+r.AvgOutputTokens/2)
}

// closure for binary search evaluating TTFT at a given lambda (req/msec)
func makeTTFTEval(data *evalFuncData) func(x float32) (float32, error) {
	return func(x float32) (float32, error) {
		data.model.Solve(x, 1)
		if !data.model.IsValid() {
			return 0, fmt.Errorf("invalid model %s", data.model)
		}
		avgPrefillTime := data.serviceParms.PrefillTime(data.requestSize, data.model.GetAvgNumInServers())
		avgDecodeTime := (data.model.GetAvgServTime() - avgPrefillTime) / data.requestSize.AvgOutputTokens
		ttft := data.model.GetAvgWaitTime() + avgPrefillTime + avgDecodeTime
		return ttft, nil
	}
}

// closure for binary search evaluating ITL at a given lambda (req/msec)
func makeITLEval(data *evalFuncData) func(x float32) (float32, error) {
	return func(x float32) (float32, error) {
		data.model.Solve(x, 1)
		if !data.model.IsValid() {
			return 0, fmt.Errorf("invalid model %s", data.model)
		}
		avgPrefillTime := data.serviceParms.PrefillTime(data.requestSize, data.model.GetAvgNumInServers())
		avgDecodeTime := (data.model.GetAvgServTime() - avgPrefillTime) / data.requestSize.AvgOutputTokens
		return avgDecodeTime, nil
	}
}

// check validity of configuration parameters
func (c *Configuration) check() error {
	if c.MaxBatchSize <= 0 || c.MaxQueueSize < 0 || c.ServiceParms == nil {
		return fmt.Errorf("invalid configuration %s", c)
	}
	return nil
}

// check validity of request size
func (rq *RequestSize) check() error {
	if rq.AvgInputTokens < 0 || rq.AvgOutputTokens < 1 {
		return fmt.Errorf("invalid request size %s", rq)
	}
	return nil
}

// check validity of target values
func (targetPerf *TargetPerf) check() error {
	if targetPerf.TargetITL < 0 ||
		targetPerf.TargetTTFT < 0 ||
		targetPerf.TargetTPS < 0 {
		return fmt.Errorf("invalid target data values %s", targetPerf)
	}
	return nil
}

/*
 * toString() functions
 */

func (c *Configuration) String() string {
	return fmt.Sprintf("{maxBatch=%d, maxTokens=%d, maxQueue=%d, servParms:%s}",
		c.MaxBatchSize, c.MaxNumTokens, c.MaxQueueSize, c.ServiceParms)
}

func (qa *LLMQueueAnalyzer) String() string {
	return fmt.Sprintf("{maxBatch=%d, maxTokens=%d, maxQueue=%d, servParms:%s, reqSize:%s, model:%s, rates:%s}",
		qa.MaxBatchSize, qa.MaxNumTokens, qa.MaxQueueSize, qa.ServiceParms, qa.RequestSize, qa.Model, qa.RateRange)
}

func (sp *ServiceParms) String() string {
	return fmt.Sprintf("{alpha=%.3f, beta=%.5f, gamma=%.5f}", sp.Alpha, sp.Beta, sp.Gamma)
}

func (rq *RequestSize) String() string {
	return fmt.Sprintf("{inTokens=%.0f, outTokens=%.0f}", rq.AvgInputTokens, rq.AvgOutputTokens)
}

func (rr *RateRange) String() string {
	return fmt.Sprintf("[%.3f, %.3f]", rr.Min, rr.Max)
}

func (am *AnalysisMetrics) String() string {
	return fmt.Sprintf("{tput=%.3f, lat=%.3f, wait=%.3f, conc=%.3f, prefill=%.3f, itl=%.3f, ttft=%.3f, maxRate=%.3f, rho=%0.3f}",
		am.Throughput, am.AvgRespTime, am.AvgWaitTime, am.AvgNumInServ, am.AvgPrefillTime, am.AvgTokenTime, am.AvgTTFT, am.MaxRate, am.Rho)
}

func (tp *TargetPerf) String() string {
	return fmt.Sprintf("{TTFT=%.3f, ITL=%.3f, TPS=%.3f}",
		tp.TargetTTFT, tp.TargetITL, tp.TargetTPS)
}

func (tr *TargetRate) String() string {
	return fmt.Sprintf("{rateTTFT=%.3f, rateITL=%.3f, rateTPS=%.3f}",
		tr.RateTargetTTFT, tr.RateTargetITL, tr.RateTargetTPS)
}
