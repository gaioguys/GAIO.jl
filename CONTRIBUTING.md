# How To Contribute

GAIO.jl is a small project maintained by only two people so we are very happy if you want to contribute! We only ask that you run tests on your contribution before making a pull request: 

## Testing

Tests can be run from the GAIO.jl Project.toml in the REPL
```
(GAIO) pkg> test
```
or by devving GAIO.jl in your local environment. For example, after acivating the Project.toml in `GAIO.jl/examples` you can test GAIO.jl by specifying
```
(examples) pkg> dev /relative/path/to/GAIO.jl

(examples) pkg> test GAIO
```

## Generating Documentation

After activating the environment in `GAIO.jl/docs` and devving GAIO.jl you can run the contents of the file `GAIO.jl/docs/make.jl` which will create the docs. It should take about five to ten minutes. It would be supremely helpful if you generate the docs and have a look through to check that nothing breaks from your contribution. 

## Opening a Pull Request

Once you've run the tests, make a pull request to [gaioguys/GAIO.jl](https://github.com/gaioguys/GAIO.jl). At the end of the PR please tag [@April-Hannah-Lena](https://github.com/April-Hannah-Lena) so that I see it. Since we're only two people it might take us a couple days to get to the PR! 

Thank you very very much for contributing!!
