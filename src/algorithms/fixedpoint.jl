# wrapper around KrylovKit.jl's eigsolve function

"""
    fixedpoint(A, x₀, which::Symbol; kwargs...) -> val, vec
    fixedpoint(A, x₀, which::Symbol, alg) -> val, vec

Compute the fixedpoint of a linear operator `A` using the specified eigensolver `alg`. The
fixedpoint is assumed to be unique.
"""
function fixedpoint(A, x₀, which::Symbol, alg::Lanczos)
    vals, vecs, info = eigsolve(A, x₀, 1, which, alg)

    if info.converged == 0
        @warnv 1 "fixedpoint not converged after $(info.numiter) iterations: normres = $(info.normres[1])"
    end

    return vals[1], vecs[1]
end

function fixedpoint(A, x₀, which::Symbol, alg::Arnoldi)
    TT, vecs, vals, info = schursolve(A, x₀, 1, which, alg)

    if info.converged == 0
        @warnv 1 "fixedpoint not converged after $(info.numiter) iterations: normres = $(info.normres[1])"
    end
    if size(TT, 2) > 1 && TT[2, 1] != 0
        @warnv 1 "non-unique fixedpoint detected"
    end

    return vals[1], vecs[1]
end

function fixedpoint(A, A_adj, x₀, y₀, which::Symbol, alg::BiArnoldi)
    vals, V, info = bieigsolve((A, A_adj), x₀, y₀, 1, which, alg)

    if info[1].converged == 0
        @warnv 1 "Right fixedpoint not converged after right: $(info[1].numiter) iterations: normres = $(info[1].normres[1])"
    end
    if info[2].converged == 0
        @warnv 1 "Left fixedpoint not converged after right: $(info[2].numiter) iterations: normres = $(info[2].normres[1])"
    end
    return vals[1], V[1][1],V[2][1]
end

### fix it so it also works for dmrg/gives error
function fixedpoint(A, A_adj, x₀, which::Symbol, alg::BiArnoldi)
    return fixedpoint(A, A_adj, x₀, copy(x₀), which::Symbol, alg::BiArnoldi)
end

function fixedpoint(A, x₀, which::Symbol; kwargs...)
    alg = KrylovKit.eigselector(A, scalartype(x₀); kwargs...)
    return fixedpoint(A, x₀, which, alg)
end
