.PHONY: test documentation benchmark

test:
	julia --color=yes --project=@. -e 'using Pkg; Pkg.build(); Pkg.test(coverage=true)'

documentation:
	julia --color=yes --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate(); include("docs/make.jl")'

benchmark:
	julia --color=yes --project=@. -e 'using Pkg; Pkg.build(); include("benchmark/runbenchmarks.jl");'
