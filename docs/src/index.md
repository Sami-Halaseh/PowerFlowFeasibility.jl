# PowerFlowFeasibility.jl

Documentation for `PowerFlowFeasibility.jl` — Polynomial Optimization Problem (POP) modeling for AC power flow feasibility using `Powsybl.jl` and `DynamicPolynomials.jl`.

## Quick Start

```julia
using Powsybl, PowerFlowFeasibility

net = Powsybl.Network.create_ieee14()
pop = build_pop_model(net; loss = :l2)
pt  = flat_start(pop)
```

See the [API Reference](@ref) for full function and type documentation.
