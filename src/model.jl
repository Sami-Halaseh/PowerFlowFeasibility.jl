using DynamicPolynomials
using Powsybl

const _MP = DynamicPolynomials.MP
const _SA = DynamicPolynomials.SA

"""
    VariableMap

Compact bidirectional mapping between physical PowSyBl `BusView` bus IDs and variable
indices `1:length(pop.variables)` in a `POPModel`. Supports merged super-buses produced
by `low_impedance_mode = :merge` (multiple physical bus IDs map to the same model bus index `i`).

Voltage variables always occupy `VR[i] => i` and `VI[i] => n_buses + i`.
Per-bus delta and L1 slack index vectors store `0` when that variable is absent on bus `i`.
"""
struct VariableMap
    n_buses::Int
    slack_idx::Int
    loss::Symbol
    bus_to_idx::Dict{String, Int}
    idx_to_buses::Vector{Vector{String}}
    dV_idx::Vector{Int}
    dP_idx::Vector{Int}
    dQ_idx::Vector{Int}
    tV_idx::Vector{Int}
    tP_idx::Vector{Int}
    tQ_idx::Vector{Int}
end

"""
    var_index(m::VariableMap, bus_id::AbstractString, kind::Symbol) -> Union{Int, Nothing}

Return the 1-based index into `pop.variables` for `(bus_id, kind)` where
`kind ∈ (:VR, :VI, :dV, :dP, :dQ, :tV, :tP, :tQ)`, or `nothing` if that variable does not
exist on `bus_id` (e.g., `:dQ` on a PV/slack bus or `:t*` under `:l2`).
Throws `KeyError` if `bus_id` is not in the network, or `ArgumentError` for an unknown `kind`.
"""
function var_index(m::VariableMap, bus_id::AbstractString, kind::Symbol)::Union{Int, Nothing}
    i = m.bus_to_idx[bus_id]
    idx = if kind === :VR
        i
    elseif kind === :VI
        m.n_buses + i
    elseif kind === :dV
        m.dV_idx[i]
    elseif kind === :dP
        m.dP_idx[i]
    elseif kind === :dQ
        m.dQ_idx[i]
    elseif kind === :tV
        m.tV_idx[i]
    elseif kind === :tP
        m.tP_idx[i]
    elseif kind === :tQ
        m.tQ_idx[i]
    else
        throw(ArgumentError("Unknown variable kind :$kind; expected one of :VR, :VI, :dV, :dP, :dQ, :tV, :tP, :tQ."))
    end
    return idx == 0 ? nothing : idx
end

Base.getindex(m::VariableMap, bus_id::AbstractString, kind::Symbol) = var_index(m, bus_id, kind)

"""
    POPModel{P <: AbstractPolynomialLike, V <: AbstractVariable}

Lightweight container for a Polynomial Optimization Problem (POP):
```math
\\min_{x \\in \\mathbb{R}^N} \\; f(x) \\quad \\text{s.t.} \\quad g_j(x) \\ge 0, \\quad h_k(x) = 0.
```

# Fields
- `objective::P`: Polynomial objective function `f(x)` to minimize.
- `equality_constraints::Vector{P}`: Polynomials `h_k(x)` constrained to `h_k(x) = 0`.
- `inequality_constraints::Vector{P}`: Polynomials `g_j(x)` constrained to `g_j(x) >= 0`.
- `variables::Vector{V}`: Ordered vector of all polynomial decision variables `x`.
- `var_map::VariableMap`: Bidirectional map between PowSyBl bus IDs and `variables` indices.
"""
struct POPModel{P <: AbstractPolynomialLike, V <: AbstractVariable}
    objective::P
    equality_constraints::Vector{P}
    inequality_constraints::Vector{P}
    variables::Vector{V}
    var_map::VariableMap
end

"""
    Point{M <: POPModel}

A full numerical assignment `values::Vector{Float64}` to all decision variables of `model::M`,
supporting O(1) semantic lookup and mutation by PowSyBl bus ID and variable kind.
"""
struct Point{M <: POPModel}
    model::M
    values::Vector{Float64}
    function Point(model::M, values::Vector{Float64}) where {M <: POPModel}
        length(values) == length(model.variables) ||
            throw(DimensionMismatch("Expected $(length(model.variables)) values, got $(length(values))."))
        return new{M}(model, values)
    end
end

Point(model::POPModel, values::AbstractVector{<:Real}) = Point(model, Float64.(values))

"""
    flat_start(model::POPModel; VR::Real = 1.0, VI::Real = 0.0) -> Point

Construct a standard power-flow flat-start `Point` with `VR_i = VR` (default `1.0` p.u.),
`VI_i = VI` (default `0.0` p.u.), and all mismatch/epigraph variables (`dV, dP, dQ, tV, tP, tQ`)
initialized to `0.0`.
"""
function flat_start(model::POPModel; VR::Real = 1.0, VI::Real = 0.0)
    n = model.var_map.n_buses
    vals = zeros(Float64, length(model.variables))
    vals[1:n] .= Float64(VR)
    VI != 0.0 && (vals[(n + 1):(2n)] .= Float64(VI))
    return Point(model, vals)
end

function Base.getindex(pt::Point, bus_id::AbstractString, kind::Symbol)::Union{Float64, Nothing}
    idx = var_index(pt.model.var_map, bus_id, kind)
    return idx === nothing ? nothing : pt.values[idx]
end

function Base.setindex!(pt::Point, val::Real, bus_id::AbstractString, kind::Symbol)
    idx = var_index(pt.model.var_map, bus_id, kind)
    idx === nothing && throw(KeyError((bus_id, kind)))
    pt.values[idx] = Float64(val)
    return pt
end

"""
    voltage(pt::Point, bus_id::AbstractString) -> ComplexF64

Return the complex per-unit voltage `VR + im * VI` at `bus_id` in `pt`.
"""
function voltage(pt::Point, bus_id::AbstractString)::ComplexF64
    m = pt.model.var_map
    i = m.bus_to_idx[bus_id]
    return ComplexF64(pt.values[i], pt.values[m.n_buses + i])
end

Base.Vector(pt::Point) = pt.values

# Execute `f()` with `Powsybl.Network` temporarily configured for `s_base` MVA per-unit queries,
# restoring previous module globals afterwards.
function _with_per_unit(f::Function; s_base::Real = 100.0)
    prev_pu = Powsybl.Network.per_unit
    prev_sn = Powsybl.Network.nominal_apparent_power
    Powsybl.Network.per_unit = true
    Powsybl.Network.nominal_apparent_power = Float64(s_base)
    try
        return f()
    finally
        Powsybl.Network.per_unit = prev_pu
        Powsybl.Network.nominal_apparent_power = prev_sn
    end
end

# Construct a separable sum `∑_k weights[k] * vars[k]^deg` directly in pre-sorted Graded{LexOrder}
# without O(N^2) variable-ring merging in DynamicPolynomials v0.7 (`vars` must be in CreationOrder).
function _separable_polynomial(vars::Vector{V}, weights::Vector{Float64}, deg::Int, fallback_var::V) where {V <: AbstractVariable}
    keep = findall(!iszero, weights)
    isempty(keep) && return polynomial(0.0 * fallback_var, Float64)
    active_vars = vars[keep]
    active_w = weights[keep]
    m = length(active_vars)
    exps = Vector{Vector{Int}}(undef, m)
    vals = Vector{Float64}(undef, m)
    for k in 1:m
        idx = m - k + 1
        e = zeros(Int, m)
        e[idx] = deg
        exps[k] = e
        vals[k] = active_w[idx]
    end
    alg = _MP.algebra(_MP.FullBasis{_MP.Monomial}(active_vars))
    sc = _SA.SparseCoefficients(exps, vals, _MP.Graded{_MP.LexOrder}())
    return _SA.AlgebraElement(sc, alg)
end

# Resolve the slack bus index in `bus_to_idx` (main connected/synchronous component):
# 1. Explicit `slack_bus_id` override if provided by the caller.
# 2. PowSyBl `slackTerminal` network extension (standard in IEEE & XIIDM exports).
# 3. Fallback: bus of a connected voltage-regulating generator with highest nominal voltage
#    (ties broken by `max_p`), so `Vsp[slack_idx]` is guaranteed to be defined.
function _resolve_slack_idx(
    network::Powsybl.Network.NetworkHandle,
    bus_to_idx::Dict{String, Int},
    gens,
    slack_bus_id::Union{Nothing, AbstractString},
)
    if slack_bus_id !== nothing
        haskey(bus_to_idx, slack_bus_id) ||
            throw(ArgumentError("Specified slack_bus_id '$slack_bus_id' is not in the main network component."))
        return bus_to_idx[slack_bus_id]
    end

    st = Powsybl.Network.get_extensions(network, "slackTerminal")
    for bus_id in (st.bus_id::Vector{String})
        idx = get(bus_to_idx, bus_id, 0)
        idx > 0 && return idx
    end

    vls = Powsybl.Network.get_voltage_levels(network, false, ["nominal_v"])
    vl_ids = vls.id::Vector{String}
    vl_nom = Dict{String, Float64}(vl_ids[i] => vls.nominal_v[i] for i in eachindex(vl_ids))

    g_conn = gens.connected::Vector{Bool}
    g_reg = gens.voltage_regulator_on::Vector{Bool}
    g_bus = gens.bus_id::Vector{String}
    g_vl = gens.voltage_level_id::Vector{String}
    g_maxp = gens.max_p

    best_idx, best_key = 0, (-Inf, -Inf)
    for r in eachindex(g_conn)
        (g_conn[r] && g_reg[r]) || continue
        idx = get(bus_to_idx, g_bus[r], 0)
        idx == 0 && continue
        mp = g_maxp[r]
        key = (get(vl_nom, g_vl[r], 0.0), isfinite(mp) ? mp : 0.0)
        if best_idx == 0 || key > best_key
            best_idx, best_key = idx, key
        end
    end
    best_idx > 0 && return best_idx
    throw(ArgumentError("Could not determine a slack bus with a voltage setpoint; pass `slack_bus_id` explicitly."))
end

# Union-Find helper used when merging buses connected by low-impedance branches.
function _uf_find!(parent::Vector{Int}, x::Int)
    while parent[x] != x
        parent[x] = parent[parent[x]]
        x = parent[x]
    end
    return x
end

# Extract per-unit sparse bus admittances (`Y_diag`, `Y_off`), neighbor lists, and bus setpoints
# from PowSyBl DataFrames (querying only required columns for speed on multi-thousand-bus grids).
function _extract_network_data(
    network::Powsybl.Network.NetworkHandle;
    s_base::Real = 100.0,
    slack_bus_id::Union{Nothing, AbstractString} = nothing,
    low_impedance_mode::Symbol = :error,
    low_impedance_threshold::Float64 = 1e-8,
)
    return _with_per_unit(; s_base) do
        # 1. Energized BusView buses in the main connected & synchronous component (0, 0).
        buses_df = Powsybl.Network.get_buses(
            network, false, ["v_mag", "connected_component", "synchronous_component"]
        )
        b_ids = buses_df.id::Vector{String}
        b_vmag, b_cc, b_sc = buses_df.v_mag, buses_df.connected_component, buses_df.synchronous_component
        keep = [i for i in eachindex(b_ids) if b_cc[i] == 0 && b_sc[i] == 0]
        n_raw = length(keep)
        n_raw > 0 || throw(ArgumentError("Network contains no energized buses in main component (0, 0)."))
        bus_to_idx = Dict{String, Int}(b_ids[keep[idx]] => idx for idx in 1:n_raw)
        v_mag_init_raw = Float64[b_vmag[keep[idx]] for idx in 1:n_raw]

        # Fetch branch tables early so `:error` / `:merge` low-impedance handling runs before
        # building `Y_diag`, `Y_off`, and bus injections.
        lines = Powsybl.Network.get_lines(
            network, false, ["r", "x", "g1", "b1", "g2", "b2", "bus1_id", "bus2_id", "connected1", "connected2"]
        )
        l_ids = lines.id::Vector{String}
        l_c1, l_c2 = lines.connected1::Vector{Bool}, lines.connected2::Vector{Bool}
        l_b1, l_b2 = lines.bus1_id::Vector{String}, lines.bus2_id::Vector{String}
        l_r, l_x, l_g1, l_sh_b1, l_g2, l_sh_b2 = lines.r, lines.x, lines.g1, lines.b1, lines.g2, lines.b2

        twts = Powsybl.Network.get_2_windings_transformers(
            network, false,
            ["rho", "alpha", "r_at_current_tap", "x_at_current_tap", "g_at_current_tap", "b_at_current_tap",
             "bus1_id", "bus2_id", "connected1", "connected2"],
        )
        t_ids = twts.id::Vector{String}
        t_c1, t_c2 = twts.connected1::Vector{Bool}, twts.connected2::Vector{Bool}
        t_b1, t_b2 = twts.bus1_id::Vector{String}, twts.bus2_id::Vector{String}
        t_rho, t_alpha = twts.rho, twts.alpha
        t_r, t_x, t_g, t_b = twts.r_at_current_tap, twts.x_at_current_tap, twts.g_at_current_tap, twts.b_at_current_tap

        # 2. Low-impedance branch handling (`|z| <= low_impedance_threshold`):
        #    - `:ignore`: keep all branches as-is (still rejects exact `r == 0 && x == 0` -> Inf/NaN).
        #    - `:error`: throw an `ArgumentError` if any connected branch has `|z| <= low_impedance_threshold`.
        #    - `:merge`: contract zero/low-impedance branches (lines and unit-tap 2WTs with `rho ≈ 1, alpha ≈ 0`)
        #      by merging their endpoint buses via union-find; their per-end shunts remain attached to the merged bus.
        line_merged = falses(length(l_c1))
        twt_merged = falses(length(t_c1))
        n = n_raw
        v_mag_init = v_mag_init_raw

        if low_impedance_mode == :error || low_impedance_mode == :merge
            parent = collect(1:n_raw)
            any_merged = false

            for r in eachindex(l_c1)
                (l_c1[r] && l_c2[r]) || continue
                i, k = get(bus_to_idx, l_b1[r], 0), get(bus_to_idx, l_b2[r], 0)
                (i == 0 || k == 0) && continue
                z = hypot(l_r[r], l_x[r])
                if z <= low_impedance_threshold
                    if low_impedance_mode == :error
                        throw(ArgumentError(
                            "Line '$(l_ids[r])' has |z| = $z <= low_impedance_threshold ($low_impedance_threshold). " *
                            "Use `low_impedance_mode = :merge` or `:ignore`."
                        ))
                    else
                        ri, rk = _uf_find!(parent, i), _uf_find!(parent, k)
                        ri != rk && (parent[max(ri, rk)] = min(ri, rk); any_merged = true)
                        line_merged[r] = true
                    end
                end
            end

            for r in eachindex(t_c1)
                (t_c1[r] && t_c2[r]) || continue
                i, k = get(bus_to_idx, t_b1[r], 0), get(bus_to_idx, t_b2[r], 0)
                (i == 0 || k == 0) && continue
                z = hypot(t_r[r], t_x[r])
                if z <= low_impedance_threshold
                    if low_impedance_mode == :error
                        throw(ArgumentError(
                            "2WT '$(t_ids[r])' has |z| = $z <= low_impedance_threshold ($low_impedance_threshold). " *
                            "Use `low_impedance_mode = :merge` or `:ignore`."
                        ))
                    else
                        # Merging endpoints directly is only physically valid when V1 == V2 (rho ≈ 1, alpha ≈ 0).
                        if !(isapprox(t_rho[r], 1.0; atol = 1e-6) && isapprox(t_alpha[r], 0.0; atol = 1e-6))
                            throw(ArgumentError(
                                "Cannot merge low-impedance 2WT '$(t_ids[r])' with non-unit complex tap " *
                                "(rho=$(t_rho[r]), alpha=$(t_alpha[r]) rad)."
                            ))
                        end
                        ri, rk = _uf_find!(parent, i), _uf_find!(parent, k)
                        ri != rk && (parent[max(ri, rk)] = min(ri, rk); any_merged = true)
                        twt_merged[r] = true
                    end
                end
            end

            if any_merged
                # Relabel union-find roots to contiguous indices 1..n and remap `bus_to_idx` so
                # all downstream shunts, branches, loads, generators, and slack lookups automatically
                # aggregate onto the merged super-buses.
                root_to_new = Dict{Int, Int}()
                v_mag_merged = Float64[]
                for raw_i in 1:n_raw
                    root = _uf_find!(parent, raw_i)
                    new_i = get!(root_to_new, root) do
                        push!(v_mag_merged, v_mag_init_raw[root])
                        return length(v_mag_merged)
                    end
                    if !isfinite(v_mag_merged[new_i]) && isfinite(v_mag_init_raw[raw_i])
                        v_mag_merged[new_i] = v_mag_init_raw[raw_i]
                    end
                end
                for (bid, raw_i) in bus_to_idx
                    bus_to_idx[bid] = root_to_new[_uf_find!(parent, raw_i)]
                end
                n = length(v_mag_merged)
                v_mag_init = v_mag_merged
            end
        end

        Y_diag = zeros(ComplexF64, n)
        Y_off = Dict{Tuple{Int, Int}, ComplexF64}()
        nbrs = [Int[] for _ in 1:n]

        # 3. Fixed bus shunts: PowSyBl supplies both shunt conductance g and susceptance b in p.u.
        shunts = Powsybl.Network.get_shunt_compensators(network, false, ["g", "b", "bus_id", "connected"])
        sh_conn, sh_bus = shunts.connected::Vector{Bool}, shunts.bus_id::Vector{String}
        sh_g, sh_b = shunts.g, shunts.b
        for r in eachindex(sh_conn)
            sh_conn[r] || continue
            i = get(bus_to_idx, sh_bus[r], 0)
            i > 0 && (Y_diag[i] += ComplexF64(sh_g[r], sh_b[r]))
        end

        # 4. AC transmission lines: standard π-section with series impedance r + jx
        #    and per-end shunt admittances g1 + jb1 (side 1) and g2 + jb2 (side 2).
        sizehint!(Y_off, 2 * (length(l_c1) + 512))
        for r in eachindex(l_c1)
            (l_c1[r] && l_c2[r]) || continue
            i, k = get(bus_to_idx, l_b1[r], 0), get(bus_to_idx, l_b2[r], 0)
            (i == 0 || k == 0) && continue
            if line_merged[r]
                # Series element was contracted into the merged bus; retain charging shunts.
                Y_diag[i] += ComplexF64(l_g1[r] + l_g2[r], l_sh_b1[r] + l_sh_b2[r])
                continue
            end
            z_c = ComplexF64(l_r[r], l_x[r])
            iszero(z_c) && throw(ArgumentError("Line '$(l_ids[r])' has exact zero impedance (r = x = 0)."))
            y = inv(z_c)
            Y_diag[i] += y + ComplexF64(l_g1[r], l_sh_b1[r])
            Y_diag[k] += y + ComplexF64(l_g2[r], l_sh_b2[r])
            if i != k
                Y_off[(i, k)] = get(Y_off, (i, k), zero(ComplexF64)) - y
                Y_off[(k, i)] = get(Y_off, (k, i), zero(ComplexF64)) - y
                push!(nbrs[i], k); push!(nbrs[k], i)
            else
                Y_diag[i] -= 2y
            end
        end

        # 5. Two-winding transformers: in PowSyBl IIDM per-unit mode, series impedance
        #    `r_at_current_tap + j*x_at_current_tap` and magnetizing shunt
        #    `g_at_current_tap + j*b_at_current_tap` sit behind an ideal complex tap
        #    `t = rho * exp(j*alpha)` on side 1 (`alpha` in radians, `rho` in p.u. ratio):
        #      I1 = (y + y_sh)*|t|^2 * V1 - y*conj(t) * V2,   I2 = -y*t * V1 + y * V2.
        for r in eachindex(t_c1)
            (t_c1[r] && t_c2[r]) || continue
            i, k = get(bus_to_idx, t_b1[r], 0), get(bus_to_idx, t_b2[r], 0)
            (i == 0 || k == 0) && continue
            y_sh = ComplexF64(t_g[r], t_b[r])
            if twt_merged[r]
                # Unit-tap series impedance contracted; retain magnetizing shunt admittance.
                Y_diag[i] += y_sh
                continue
            end
            z_c = ComplexF64(t_r[r], t_x[r])
            iszero(z_c) && throw(ArgumentError("2WT '$(t_ids[r])' has exact zero impedance (r = x = 0)."))
            y = inv(z_c)
            tap = t_rho[r] * cis(t_alpha[r])
            Y_diag[i] += (y + y_sh) * abs2(tap)
            Y_diag[k] += y
            if i != k
                Y_off[(i, k)] = get(Y_off, (i, k), zero(ComplexF64)) - y * conj(tap)
                Y_off[(k, i)] = get(Y_off, (k, i), zero(ComplexF64)) - y * tap
                push!(nbrs[i], k); push!(nbrs[k], i)
            else
                Y_diag[i] -= y * (tap + conj(tap))
            end
        end

        for i in 1:n; sort!(unique!(nbrs[i])); end

        # 6. Loads & generators -> net bus active/reactive injections and voltage setpoints.
        P_sp, Q_sp, V_sp = zeros(Float64, n), zeros(Float64, n), fill(NaN, n)
        has_local_v_reg = falses(n)

        loads = Powsybl.Network.get_loads(network, false, ["p0", "q0", "bus_id", "connected"])
        ld_conn, ld_bus, ld_p0, ld_q0 = loads.connected::Vector{Bool}, loads.bus_id::Vector{String}, loads.p0, loads.q0
        for r in eachindex(ld_conn)
            ld_conn[r] || continue
            i = get(bus_to_idx, ld_bus[r], 0)
            if i > 0
                P_sp[i] -= ld_p0[r]
                Q_sp[i] -= ld_q0[r]
            end
        end

        # Note: `regulated_bus_id` is a non-default PowSyBl attribute, so we request it
        # explicitly to support both local and remote generator voltage regulation.
        gens = Powsybl.Network.get_generators(
            network, false,
            ["target_p", "target_q", "target_v", "max_p", "voltage_regulator_on",
             "regulated_bus_id", "voltage_level_id", "bus_id", "connected"],
        )
        g_conn, g_reg = gens.connected::Vector{Bool}, gens.voltage_regulator_on::Vector{Bool}
        g_bus, g_reg_bus = gens.bus_id::Vector{String}, gens.regulated_bus_id::Vector{String}
        g_tp, g_tq, g_tv = gens.target_p, gens.target_q, gens.target_v
        for r in eachindex(g_conn)
            g_conn[r] || continue
            i = get(bus_to_idx, g_bus[r], 0)
            i == 0 && continue
            P_sp[i] += g_tp[r]
            if g_reg[r]
                # A regulating generator absorbs reactive power mismatch locally at `bus_id`
                # (making `i` a PV/slack bus with no Q balance equation), while enforcing
                # its voltage setpoint `target_v` on `regulated_bus_id` (which may differ).
                has_local_v_reg[i] = true
                reg_id = isempty(g_reg_bus[r]) ? g_bus[r] : g_reg_bus[r]
                j = get(bus_to_idx, reg_id, 0)
                if j > 0
                    tv = g_tv[r]
                    if !isnan(V_sp[j]) && !isapprox(V_sp[j], tv; atol = 1e-6)
                        throw(ArgumentError("Conflicting voltage setpoints at bus '$reg_id': $(V_sp[j]) vs $tv p.u."))
                    end
                    V_sp[j] = tv
                end
            else
                # Non-regulating generator: contributes fixed `target_p` and `target_q`
                # without turning the bus into a PV bus (bus remains PQ unless another unit regulates).
                tq = g_tq[r]
                isfinite(tq) && (Q_sp[i] += tq)
            end
        end

        s_idx = _resolve_slack_idx(network, bus_to_idx, gens, slack_bus_id)
        if isnan(V_sp[s_idx])
            v_init = v_mag_init[s_idx]
            V_sp[s_idx] = (isfinite(v_init) && v_init > 0.0) ? v_init : 1.0
        end

        v_idxs = findall(!isnan, V_sp)
        nonslack_idxs = [i for i in 1:n if i != s_idx]
        pq_idxs = [i for i in 1:n if i != s_idx && !has_local_v_reg[i]]

        # Build reverse mapping `idx_to_buses[i]` in original PowSyBl BusView order.
        idx_to_buses = [String[] for _ in 1:n]
        for raw_i in 1:n_raw
            bid = b_ids[keep[raw_i]]
            push!(idx_to_buses[bus_to_idx[bid]], bid)
        end

        return n, s_idx, v_idxs, nonslack_idxs, pq_idxs, P_sp, Q_sp, V_sp, Y_diag, Y_off, nbrs, bus_to_idx, idx_to_buses
    end
end

"""
    build_pop_model(network::Powsybl.Network.NetworkHandle;
                    loss::Union{Symbol, AbstractString} = :l2,
                    ωP::Real = 1.0, ωQ::Real = 1.0, ωV::Real = 1.0,
                    lV::Real = 0.5, uV::Real = 0.5,
                    s_base::Real = 100.0,
                    slack_bus_id::Union{Nothing, AbstractString} = nothing,
                    low_impedance_mode::Union{Symbol, AbstractString} = :error,
                    low_impedance_threshold::Real = 1e-8) -> POPModel

Build the AC Power Flow Feasibility (PFF) Polynomial Optimization Problem directly
from a `Powsybl.Network.NetworkHandle` in rectangular voltage coordinates `V_i = VR_i + im * VI_i`.

# Keywords
- `loss`: `:l2` (default) or `:l1` (also accepts `"l2"` / `"l1"`).
- `ωP, ωQ, ωV`: objective weights for active power, reactive power, and squared voltage mismatches.
- `lV, uV`: uniform voltage-magnitude deviation bounds `(1 - lV)^2 <= |V_i|^2 <= (1 + uV)^2`.
- `s_base`: per-unit apparent power base in MVA (default `100.0`).
- `slack_bus_id`: optional BusView bus ID overriding automatic slack bus detection.
- `low_impedance_mode`: how to handle connected branches with `|z| <= low_impedance_threshold`:
  - `:error` (default): throw an `ArgumentError`.
  - `:merge`: contract low-impedance branches (lines and unit-tap 2WTs) by merging endpoint buses
    via union-find, retaining branch shunts and aggregating bus loads/generators/setpoints.
  - `:ignore`: build the model as-is (still rejects exact `r == x == 0` division by zero).
- `low_impedance_threshold`: per-unit impedance magnitude threshold `|z| = hypot(r, x)` (default `1e-8`).

# Formulation
- **Variables**:
  - `VR[1:n], VI[1:n]`: real and imaginary bus voltage coordinates for all `n` buses.
  - `dV`: squared voltage-magnitude mismatch at regulated buses `i ∈ V_sp`.
  - `dP`: active power mismatch at non-slack buses `i ∈ N \\ {slack}`.
  - `dQ`: reactive power mismatch at PQ buses `i ∈ N_PQ` (non-slack buses with no local
    voltage-regulating generator).
  - `tV, tP, tQ` (`loss = :l1` only): non-negative epigraph slack variables bounding `|d| <= t`.
- **Objective**:
  - `loss = :l2`: `∑ ωV dV_i^2 + ∑ ωP dP_i^2 + ∑ ωQ dQ_i^2`
  - `loss = :l1`: `∑ ωV tV_i + ∑ ωP tP_i + ∑ ωQ tQ_i`
- **Equality constraints (`h(x) = 0`)**:
  - Slack phase reference: `VI[slack] = 0`
  - Voltage magnitude setpoints (`i ∈ V_sp`): `VR_i^2 + VI_i^2 + dV_i - Vsp_i^2 = 0`
  - Active power balance (`i ∈ N \\ {slack}`): `Pspec_i + dP_i - P_i(VR, VI) = 0`
  - Reactive power balance (`i ∈ N_PQ`): `Qspec_i + dQ_i - Q_i(VR, VI) = 0`
  - Squared L1 epigraph complementarity (`loss = :l1` only): `t_r^2 - d_r^2 = 0`
- **Inequality constraints (`g(x) >= 0`)**:
  - Slack reference pin: `VR[slack] >= 0`
  - Operational voltage limits (`∀ i`): `|V_i|^2 - (1 - lV)^2 >= 0` and `(1 + uV)^2 - |V_i|^2 >= 0`
  - Linear L1 epigraph bounds (`loss = :l1` only): `t_r - d_r >= 0` and `t_r + d_r >= 0`
"""
function build_pop_model(
    network::Powsybl.Network.NetworkHandle;
    loss::Union{Symbol, AbstractString} = :l2,
    ωP::Real = 1.0,
    ωQ::Real = 1.0,
    ωV::Real = 1.0,
    lV::Real = 0.5,
    uV::Real = 0.5,
    s_base::Real = 100.0,
    slack_bus_id::Union{Nothing, AbstractString} = nothing,
    low_impedance_mode::Union{Symbol, AbstractString} = :error,
    low_impedance_threshold::Real = 1e-8,
)::POPModel
    loss_sym = Symbol(lowercase(String(loss)))
    loss_sym in (:l1, :l2) || throw(ArgumentError("Unsupported loss '$loss'; expected :l1 or :l2."))

    low_imp_sym = Symbol(lowercase(String(low_impedance_mode)))
    low_imp_sym in (:ignore, :error, :merge) ||
        throw(ArgumentError("Unsupported low_impedance_mode '$low_impedance_mode'; expected :ignore, :error, or :merge."))
    low_imp_tol = Float64(low_impedance_threshold)
    low_imp_tol >= 0.0 || throw(ArgumentError("low_impedance_threshold must be non-negative (got $low_impedance_threshold)."))

    # 1. Extract per-unit network topology, sparse bus admittances, and bus specifications.
    n, slack_idx, v_idxs, nonslack_idxs, pq_idxs, Pspec, Qspec, Vsp, Y_diag, Y_off, neighbours, bus_to_idx, idx_to_buses =
        _extract_network_data(
            network;
            s_base,
            slack_bus_id,
            low_impedance_mode = low_imp_sym,
            low_impedance_threshold = low_imp_tol,
        )

    # 2. Decision variables (DynamicPolynomials).
    #    Rectangular voltage coordinates V_i = VR[i] + im * VI[i] for every bus i = 1..n.
    @polyvar VR[1:n] VI[1:n]

    #    Signed physical mismatch (delta) variables on active equations:
    #    - dV[k] for regulated voltage buses v_idxs[k]
    #    - dP[k] for non-slack buses nonslack_idxs[k]
    #    - dQ[k] for PQ buses pq_idxs[k]
    @polyvar dV[1:length(v_idxs)]
    @polyvar dP[1:length(nonslack_idxs)]
    @polyvar dQ[1:length(pq_idxs)]

    if loss_sym == :l1
        # Distributed L1 epigraph slack variables (t >= |d|) paired 1-to-1 with (dV, dP, dQ).
        @polyvar tV[1:length(v_idxs)]
        @polyvar tP[1:length(nonslack_idxs)]
        @polyvar tQ[1:length(pq_idxs)]
        variables = vcat(VR, VI, dV, dP, dQ, tV, tP, tQ)
    else
        variables = vcat(VR, VI, dV, dP, dQ)
    end

    # 3. Objective function.
    Poly = DynamicPolynomials.Polynomial{Float64}
    Term = typeof(1.0 * VR[1]^2)
    weights = vcat(
        fill(Float64(ωV), length(v_idxs)),
        fill(Float64(ωP), length(nonslack_idxs)),
        fill(Float64(ωQ), length(pq_idxs)),
    )
    objective::Poly = if loss_sym == :l2
        # Smooth quadratic L2 penalty over squared voltage, active power, and reactive power mismatches.
        _separable_polynomial(vcat(dV, dP, dQ), weights, 2, VR[1])
    else
        # Smooth linear L1 objective over epigraph slack variables (coupled to |d| via inequalities below).
        _separable_polynomial(vcat(tV, tP, tQ), weights, 1, VR[1])
    end

    # 4. Equality constraints: h(x) = 0.
    n_deltas = length(v_idxs) + length(nonslack_idxs) + length(pq_idxs)
    eqs = Poly[]
    sizehint!(eqs, 1 + (loss_sym == :l1 ? 2 * n_deltas : n_deltas))
    terms_buf = Term[]
    const_one = one(VR[1]^2)

    # (a) Slack bus phase angle reference: VI[slack] = 0 (paired with VR[slack] >= 0 below => θ_slack = 0).
    push!(eqs, polynomial(1.0 * VI[slack_idx], Float64))

    # (b) Squared voltage-magnitude setpoint equations for all regulated buses i ∈ v_idxs:
    #     VR_i^2 + VI_i^2 + dV_k - Vsp_i^2 = 0.
    for (k, i) in enumerate(v_idxs)
        push!(eqs, polynomial([-Vsp[i]^2 * const_one, 1.0 * VR[i]^2, 1.0 * VI[i]^2, 1.0 * dV[k]]))
    end

    # (c) Active power balance equations for all non-slack buses i ∈ nonslack_idxs:
    #     P_i(VR, VI) = |V_i|^2 G_ii + ∑_{m ∈ N(i)} [VR_i(G_im VR_m - B_im VI_m) + VI_i(B_im VR_m + G_im VI_m)]
    #     Enforced as: Pspec_i + dP_k - P_i(VR, VI) = 0.
    for (k, i) in enumerate(nonslack_idxs)
        empty!(terms_buf)
        Pspec[i] != 0.0 && push!(terms_buf, Pspec[i] * const_one)
        push!(terms_buf, 1.0 * dP[k])
        g_ii = real(Y_diag[i])
        if g_ii != 0.0
            push!(terms_buf, -g_ii * VR[i]^2)
            push!(terms_buf, -g_ii * VI[i]^2)
        end
        vr_i, vi_i = VR[i], VI[i]
        for m in neighbours[i]
            y_im = Y_off[(i, m)]
            g_im, b_im = real(y_im), imag(y_im)
            if g_im != 0.0
                push!(terms_buf, -g_im * vr_i * VR[m])
                push!(terms_buf, -g_im * vi_i * VI[m])
            end
            if b_im != 0.0
                push!(terms_buf,  b_im * vr_i * VI[m])
                push!(terms_buf, -b_im * vi_i * VR[m])
            end
        end
        push!(eqs, polynomial(terms_buf))
    end

    # (d) Reactive power balance equations for all PQ buses i ∈ pq_idxs:
    #     Q_i(VR, VI) = -|V_i|^2 B_ii + ∑_{m ∈ N(i)} [VR_i(-B_im VR_m - G_im VI_m) + VI_i(G_im VR_m - B_im VI_m)]
    #     Enforced as: Qspec_i + dQ_k - Q_i(VR, VI) = 0.
    for (k, i) in enumerate(pq_idxs)
        empty!(terms_buf)
        Qspec[i] != 0.0 && push!(terms_buf, Qspec[i] * const_one)
        push!(terms_buf, 1.0 * dQ[k])
        b_ii = imag(Y_diag[i])
        if b_ii != 0.0
            push!(terms_buf, b_ii * VR[i]^2)
            push!(terms_buf, b_ii * VI[i]^2)
        end
        vr_i, vi_i = VR[i], VI[i]
        for m in neighbours[i]
            y_im = Y_off[(i, m)]
            g_im, b_im = real(y_im), imag(y_im)
            if b_im != 0.0
                push!(terms_buf, b_im * vr_i * VR[m])
                push!(terms_buf, b_im * vi_i * VI[m])
            end
            if g_im != 0.0
                push!(terms_buf,  g_im * vr_i * VI[m])
                push!(terms_buf, -g_im * vi_i * VR[m])
            end
        end
        push!(eqs, polynomial(terms_buf))
    end

    # (e) Squared L1 epigraph complementarity equalities (`loss = :l1` only):
    #     At any true L1 optimum with positive weights (ω > 0), each epigraph slack is tight:
    #     t_r = |d_r|, which implies (t_r - d_r)(t_r + d_r) = t_r^2 - d_r^2 = 0.
    #     Why include this redundant polynomial equality?
    #     In an Order-1 Moment-SOS SDP relaxation, the linear inequalities `t_r ± d_r >= 0`
    #     only constrain first-order pseudo-moments `L_y(t_r) >= |L_y(d_r)|`. When a mismatch has
    #     zero mean (`L_y(d_r) = 0`) but positive pseudo-variance (`L_y(d_r^2) > 0`), the SDP can
    #     set `L_y(t_r) = 0` and collapse the L1 lower bound to zero. Enforcing `t_r^2 - d_r^2 = 0`
    #     forces `L_y(t_r^2) = L_y(d_r^2)` inside the existing 2-variable clique `{d_r, t_r}`
    #     (zero correlative sparsity fill-in!), tightening the relaxation via Cauchy-Schwarz.
    if loss_sym == :l1
        for k in eachindex(v_idxs); push!(eqs, polynomial([1.0 * tV[k]^2, -1.0 * dV[k]^2])); end
        for k in eachindex(nonslack_idxs); push!(eqs, polynomial([1.0 * tP[k]^2, -1.0 * dP[k]^2])); end
        for k in eachindex(pq_idxs); push!(eqs, polynomial([1.0 * tQ[k]^2, -1.0 * dQ[k]^2])); end
    end

    # 5. Inequality constraints: g(x) >= 0.
    ineqs = Poly[]
    sizehint!(ineqs, 1 + 2n + (loss_sym == :l1 ? 2 * n_deltas : 0))

    # (a) Slack bus real-axis half-space pin: VR[slack] >= 0 (eliminates the θ = π ambiguity of VI[slack] = 0).
    push!(ineqs, polynomial(1.0 * VR[slack_idx], Float64))

    # (b) Operational squared voltage-magnitude box bounds for every bus i = 1..n:
    #     (1 - lV)^2 <= VR[i]^2 + VI[i]^2 <= (1 + uV)^2.
    vmin2 = (1.0 - Float64(lV))^2
    vmax2 = (1.0 + Float64(uV))^2
    for i in 1:n
        push!(ineqs, polynomial([-vmin2 * const_one,  1.0 * VR[i]^2,  1.0 * VI[i]^2]))
        push!(ineqs, polynomial([ vmax2 * const_one, -1.0 * VR[i]^2, -1.0 * VI[i]^2]))
    end

    # (c) Decoupled linear L1 epigraph inequalities (`loss = :l1` only):
    #     t_r - d_r >= 0 and t_r + d_r >= 0  <=>  t_r >= |d_r|  (automatically implying t_r >= 0).
    #     Each pair touches only `{d_r, t_r}`, forming isolated size-2 cliques with no bus coupling.
    if loss_sym == :l1
        for k in eachindex(v_idxs)
            push!(ineqs, polynomial([1.0 * tV[k], -1.0 * dV[k]]))
            push!(ineqs, polynomial([1.0 * tV[k],  1.0 * dV[k]]))
        end
        for k in eachindex(nonslack_idxs)
            push!(ineqs, polynomial([1.0 * tP[k], -1.0 * dP[k]]))
            push!(ineqs, polynomial([1.0 * tP[k],  1.0 * dP[k]]))
        end
        for k in eachindex(pq_idxs)
            push!(ineqs, polynomial([1.0 * tQ[k], -1.0 * dQ[k]]))
            push!(ineqs, polynomial([1.0 * tQ[k],  1.0 * dQ[k]]))
        end
    end

    # Build bidirectional VariableMap directly from the contiguous variable block offsets.
    nv, np, nq = length(v_idxs), length(nonslack_idxs), length(pq_idxs)
    off_dV, off_dP, off_dQ = 2n, 2n + nv, 2n + nv + np
    dV_idx, dP_idx, dQ_idx = zeros(Int, n), zeros(Int, n), zeros(Int, n)
    tV_idx, tP_idx, tQ_idx = zeros(Int, n), zeros(Int, n), zeros(Int, n)
    for (k, i) in enumerate(v_idxs); dV_idx[i] = off_dV + k; end
    for (k, i) in enumerate(nonslack_idxs); dP_idx[i] = off_dP + k; end
    for (k, i) in enumerate(pq_idxs); dQ_idx[i] = off_dQ + k; end
    if loss_sym == :l1
        off_tV = off_dQ + nq
        off_tP, off_tQ = off_tV + nv, off_tV + nv + np
        for (k, i) in enumerate(v_idxs); tV_idx[i] = off_tV + k; end
        for (k, i) in enumerate(nonslack_idxs); tP_idx[i] = off_tP + k; end
        for (k, i) in enumerate(pq_idxs); tQ_idx[i] = off_tQ + k; end
    end
    var_map = VariableMap(n, slack_idx, loss_sym, bus_to_idx, idx_to_buses, dV_idx, dP_idx, dQ_idx, tV_idx, tP_idx, tQ_idx)

    return POPModel(objective, eqs, ineqs, variables, var_map)
end
