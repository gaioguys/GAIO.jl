# benchmarks can be defined like this
SUITE["example"] = BenchmarkGroup()

SUITE["example"]["test"] = @benchmarkable 1+1
