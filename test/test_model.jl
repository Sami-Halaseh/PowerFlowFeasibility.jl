using PowerFlowFeasibility
using Powsybl
using Test

Powsybl.LibPowsybl.set_config_read(false)

function _solve_ac_point(net, pop::POPModel)
    params = Powsybl.LoadFlow.load_flow_parameters()
    params.use_reactive_limits = false
    params.distributed_slack = false
    params.transformer_voltage_control_on = false
    params.phase_shifter_regulation_on = false
    params.shunt_compensator_voltage_control_on = false
    params.twt_split_shunt_admittance = false
    @assert Powsybl.LoadFlow.run_ac(net, params).component_results.status[1] == Powsybl.LoadFlow.CONVERGED

    prev_pu = Powsybl.Network.per_unit
    Powsybl.Network.per_unit = true
    buses = Powsybl.Network.get_buses(net)
    Powsybl.Network.per_unit = prev_pu

    pt = flat_start(pop)
    for r in eachrow(buses)
        haskey(pop.var_map.bus_to_idx, r.id) || continue
        pt[r.id, :VR] = r.v_mag * cos(r.v_angle)
        pt[r.id, :VI] = r.v_mag * sin(r.v_angle)
    end
    return pt
end

@testset "model.jl" begin
    @testset "AC LoadFlow exactness (IEEE14 & IEEE30)" begin
        for (net, tol) in ((Powsybl.Network.create_ieee14(), 1e-6), (Powsybl.Network.create_ieee30(), 1e-6))
            for loss in (:l2, "l1")
                pop = build_pop_model(net; loss)
                @test !Powsybl.Network.per_unit # global restored cleanly
                pt = _solve_ac_point(net, pop)
                sub = pop.variables => Vector(pt)
                @test pop.objective(sub) == 0.0
                @test maximum(abs(h(sub)) for h in pop.equality_constraints) <= tol
                @test minimum(g(sub) for g in pop.inequality_constraints) >= -1e-12
            end
        end
    end

    @testset "L1 epigraph & squared complementarity (IEEE9)" begin
        pop = build_pop_model(Powsybl.Network.create_ieee9(); loss = :l1)
        pt = flat_start(pop)
        n = pop.var_map.n_buses
        n_d = (length(pop.variables) - 2n) ÷ 2
        sub0 = pop.variables => pt.values
        for k in 1:n_d
            dval = -pop.equality_constraints[1 + k](sub0)
            pt.values[2n + k] = dval
            pt.values[2n + n_d + k] = abs(dval)
        end
        sub = pop.variables => pt.values
        @test maximum(abs(h(sub)) for h in pop.equality_constraints) <= 1e-13
        @test minimum(g(sub) for g in pop.inequality_constraints) >= -1e-13
        @test pop.objective(sub) ≈ sum(abs, pt.values[(2n + 1):(2n + n_d)])
    end

    @testset "Objective weights & zero-weight edge case" begin
        net14 = Powsybl.Network.create_ieee14()
        pop_w = build_pop_model(net14; loss = :l2, ωV = 5.0, ωP = 2.0, ωQ = 3.0)
        pt = flat_start(pop_w)
        pt["VL1_0", :dV] = 2.0
        pt["VL4_0", :dP] = -3.0
        pt["VL4_0", :dQ] = 4.0
        @test pop_w.objective(pop_w.variables => pt.values) ≈ 5.0 * 4.0 + 2.0 * 9.0 + 3.0 * 16.0

        pop_zero = build_pop_model(net14; loss = :l2, ωV = 0.0, ωP = 0.0, ωQ = 0.0)
        @test pop_zero.objective(pop_zero.variables => flat_start(pop_zero).values) == 0.0
    end

    @testset "Slack resolution fallback & explicit override (Eurostag)" begin
        net = Powsybl.Network.create_eurostag_tutorial_example1()
        pop_auto = build_pop_model(net) # no slackTerminal extension -> generator fallback
        @test pop_auto.var_map.slack_idx == pop_auto.var_map.bus_to_idx["VLGEN_0"]

        pop_ovr = build_pop_model(net; slack_bus_id = "VLHV1_0") # non-PV bus -> v_mag_init fallback
        @test pop_ovr.var_map.slack_idx == pop_ovr.var_map.bus_to_idx["VLHV1_0"]
        @test pop_ovr.var_map["VLHV1_0", :dV] !== nothing
        @test_throws ArgumentError build_pop_model(net; slack_bus_id = "MISSING_BUS")
    end

    @testset "Low-impedance modes (:error, :ignore, :merge)" begin
        net14 = Powsybl.Network.create_ieee14()
        @test_throws ArgumentError build_pop_model(net14; low_impedance_mode = :error, low_impedance_threshold = 0.05)

        pop_ign = build_pop_model(net14; low_impedance_mode = :ignore, low_impedance_threshold = 0.05)
        @test pop_ign.var_map.n_buses == 14

        pop_m = build_pop_model(net14; loss = :l1, low_impedance_mode = :merge, low_impedance_threshold = 0.05)
        @test pop_m.var_map.n_buses == 13
        @test length(pop_m.var_map.bus_to_idx) == 14
        groups = filter(g -> length(g) > 1, pop_m.var_map.idx_to_buses)
        @test length(groups) == 1
        b1, b2 = groups[1]
        @test pop_m.var_map[b1, :VR] == pop_m.var_map[b2, :VR]

        # Non-unit-tap low-impedance 2WT cannot be merged
        @test_throws ArgumentError build_pop_model(net14; low_impedance_mode = :merge, low_impedance_threshold = 0.3)
    end

    @testset "VariableMap & Point API (unmerged + merged)" begin
        net14 = Powsybl.Network.create_ieee14()
        pop_l2 = build_pop_model(net14; loss = :l2)
        m2 = pop_l2.var_map
        @test string(pop_l2.variables[m2["VL1_0", :VR]]) == "VR[$(m2.bus_to_idx["VL1_0"])]"
        @test m2["VL1_0", :dV] !== nothing && m2["VL1_0", :dP] === nothing && m2["VL1_0", :dQ] === nothing
        @test m2["VL2_0", :dV] !== nothing && m2["VL2_0", :dP] !== nothing && m2["VL2_0", :dQ] === nothing
        @test m2["VL4_0", :dV] === nothing && m2["VL4_0", :dP] !== nothing && m2["VL4_0", :dQ] !== nothing
        @test m2["VL4_0", :tP] === nothing

        pop_m = build_pop_model(net14; loss = :l1, low_impedance_mode = :merge, low_impedance_threshold = 0.05)
        b1, b2 = only(filter(g -> length(g) > 1, pop_m.var_map.idx_to_buses))
        pt = flat_start(pop_m)
        @test pt[b1, :VR] == 1.0 && pt[b1, :VI] == 0.0 && pt["VL1_0", :dQ] === nothing
        pt[b1, :VR] = 1.05
        pt[b1, :VI] = -0.1
        @test voltage(pt, b2) == 1.05 - 0.1im
        @test Vector(pt) === pt.values
    end

    @testset "Input validation guards" begin
        net9 = Powsybl.Network.create_ieee9()
        pop = build_pop_model(net9; loss = :l2)
        pt = flat_start(pop)
        @test_throws ArgumentError build_pop_model(net9; loss = :l3)
        @test_throws ArgumentError build_pop_model(net9; low_impedance_mode = :bad)
        @test_throws ArgumentError build_pop_model(net9; low_impedance_threshold = -1e-6)
        @test_throws KeyError var_index(pop.var_map, "NO_SUCH_BUS", :VR)
        @test_throws ArgumentError var_index(pop.var_map, "VL1_0", :bad_kind)
        @test_throws KeyError (pt["VL1_0", :dQ] = 1.0)
        @test_throws DimensionMismatch Point(pop, zeros(2))
    end
end
