"""
$(TYPEDEF)

Single-site Biorthonormal DMRG algorithm for finding the dominant left and right eigenvector for non-Hermitian systems.
based on https://doi.org/10.48550/arXiv.2401.15000

## Fields

$(TYPEDFIELDS)
"""
struct NH_DMRG{A, F} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64

    "maximal amount of iterations"
    maxiter::Int

    "setting for how much information is displayed"
    verbosity::Int

    "algorithm used for the eigenvalue solvers"
    alg_eigsolve::A

    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs`"
    finalize::F
end
function NH_DMRG(;
        tol = Defaults.tol, maxiter = Defaults.maxiter, alg_eigsolve = (;),
        verbosity = Defaults.verbosity, finalize = Defaults._finalize
    )
    alg_eigsolve′ = alg_eigsolve isa NamedTuple ? Defaults.alg_eigsolve(; alg_eigsolve...) :
        alg_eigsolve
    return NH_DMRG(tol, maxiter, verbosity, alg_eigsolve′, finalize)
end


### Conjugates MPO but does not introduce adjoint spaces so that we can act like its an original hamiltonian. This is justified by the fact that bottom and top virtual spaces need to be equal for H.
## We are also kind of assuming that the virtual degreees of freedom are the same in dual and original Hamitlonian
function _conj_mpo_no(W::JordanMPOTensor)
    V = left_virtualspace(W) ⊗ physicalspace(W) ← physicalspace(W) ⊗ right_virtualspace(W)
    Ux = isomorphism(space(W.A)[1],space(W.A)[1]')
    Uy = isomorphism(space(W.A)[4],space(W.A)[4]')
    @plansor A[-1 -2; -3 -4] := conj(W.A[-1 -3; -2 -4])
    @plansor A[-1 -2;-3 -4]:= A[1 -2;-3  4] * Ux[-1;1]* Uy[-4;4]
    @plansor B[-1 -2; -3] ≔ conj(W.B[1 -3; -2]) * Ux[-1;1]
    @plansor C[-1; -2 -3] ≔ conj(W.C[-2; -1 3]) * Uy[-3;3]
    D = copy(adjoint(W.D))
    return JordanMPOTensor(V, A, B, C, D)
end



function find_groundstate!(ψR::AbstractFiniteMPS,ψL::AbstractFiniteMPS, H, alg::NH_DMRG, envs = environments(ψR, H, ψL))
    ϵs = map(pos -> calc_galerkin(pos, ψL, H, ψR, envs), 1:length(ψL))
    ϵ = maximum(ϵs)
    log = IterLog("NH_DMRG")
    H_adj = MPO([_conj_mpo_no(H[i]) for i in 1:length(H)])
    envs1 = environments(ψR, H, ψR)
    envs2 = environments(ψL, H_adj, ψL)
    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, expectation_value(ψL, H, envs1))
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)
            zerovector!(ϵs)
            for pos in [1:(length(ψL) - 1); length(ψL):-1:2]
                h = AC_hamiltonian(pos, ψR, H, ψR, envs1)
                h_adj = AC_hamiltonian(pos, ψL, H_adj, ψL, envs2)
                #### TODO: put warning for Lancos (default) somewhere + verify the results
                if alg_eigsolve isa BiArnoldi
                _, vec1,vec2 = fixedpoint(h, h_adj, ψR.AC[pos],  ψL.AC[pos], :SR, alg_eigsolve)
                ψR.AC[pos] = vec1
                ψL.AC[pos] = vec2
                envs = environments(ψR, H, ψL)
                ϵs[pos] = max(ϵs[pos], calc_galerkin(pos, ψR, H, ψL, envs)) 
                else
                 _, vec = fixedpoint(h, ψR.AC[pos], :SR, alg_eigsolve)
                ψR.AC[pos] = vec
                _, vec = fixedpoint(h_adj, ψL.AC[pos], :SR, alg_eigsolve)
                ψL.AC[pos] = vec
                ϵs[pos] = max(ϵs[pos], calc_galerkin(pos, ψR, H, ψR, envs1),calc_galerkin(pos, ψL, H_adj, ψL, envs2)) ### check if error is proper! should it be both or only one?
                end
                
            end
            ϵ = maximum(ϵs)
             #### TODO: write some two way finaliser for environments
            ψR, envs1 = alg.finalize(iter, ψR, H, envs1)::Tuple{typeof(ψL), typeof(envs1)}
            ψL, envs2 = alg.finalize(iter, ψL, H_adj, envs2)::Tuple{typeof(ψR), typeof(envs2)}

            if ϵ <= alg.tol
                @infov 2 logfinish!(log, iter, ϵ, expectation_value(ψR, H, envs1))
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ, expectation_value(ψR, H, envs1))
            else
                @infov 3 logiter!(log, iter, ϵ, expectation_value(ψR, H, envs1))
            end
        end
    end
    return ψR, ψL, envs, ϵ
end


"""
$(TYPEDEF)

Two-site Non Hermitian DMRG algorithm for finding the dominant left and right eigenvector for non-Hermitian systems.

## Fields

$(TYPEDFIELDS)
"""
struct NH_DMRG2{A, S, F} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64

    "maximal amount of iterations"
    maxiter::Int

    "setting for how much information is displayed"
    verbosity::Int

    "algorithm used for the eigenvalue solvers"
    alg_eigsolve::A

    "algorithm used for the singular value decomposition"
    alg_svd::S

    "algorithm used for [truncation](@extref MatrixAlgebraKit.TruncationStrategy) of the two-site update"
    trscheme::TruncationStrategy

    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs`"
    finalize::F
end
# TODO: find better default truncation
function NH_DMRG2(;
        tol = Defaults.tol, maxiter = Defaults.maxiter, verbosity = Defaults.verbosity,
        alg_eigsolve = (;), alg_svd = Defaults.alg_svd(), trscheme,
        finalize = Defaults._finalize
    )
    alg_eigsolve′ = alg_eigsolve isa NamedTuple ? Defaults.alg_eigsolve(; alg_eigsolve...) :
        alg_eigsolve
    return NH_DMRG2(tol, maxiter, verbosity, alg_eigsolve′, alg_svd, trscheme, finalize)
end

function find_groundstate!(ψ::AbstractFiniteMPS, H, alg::NH_DMRG2, envs = environments(ψ, H))
    ϵs = map(pos -> calc_galerkin(pos, ψ, H, ψ, envs), 1:length(ψ))
    ϵ = maximum(ϵs)
    log = IterLog("NH_DMRG2")

    LoggingExtras.withlevel(; alg.verbosity) do
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)
            zerovector!(ϵs)

            # left to right sweep
            for pos in 1:(length(ψ) - 1)
                @plansor ac2[-1 -2; -3 -4] := ψ.AC[pos][-1 -2; 1] * ψ.AR[pos + 1][1 -4; -3]
                Hac2 = AC2_hamiltonian(pos, ψ, H, ψ, envs)
                _, newA2center = fixedpoint(Hac2, ac2, :SR, alg_eigsolve)

                al, c, ar = svd_trunc!(newA2center; trunc = alg.trscheme, alg = alg.alg_svd)
                normalize!(c)
                v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) * conj(ar[6; 3 4])
                ϵs[pos] = max(ϵs[pos], abs(1 - abs(v)))

                ψ.AC[pos] = (al, complex(c))
                ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
            end

            # right to left sweep
            for pos in (length(ψ) - 2):-1:1
                @plansor ac2[-1 -2; -3 -4] := ψ.AL[pos][-1 -2; 1] * ψ.AC[pos + 1][1 -4; -3]
                Hac2 = AC2_hamiltonian(pos, ψ, H, ψ, envs)
                _, newA2center = fixedpoint(Hac2, ac2, :SR, alg_eigsolve)

                al, c, ar = svd_trunc!(newA2center; trunc = alg.trscheme, alg = alg.alg_svd)
                normalize!(c)
                v = @plansor ac2[1 2; 3 4] * conj(al[1 2; 5]) * conj(c[5; 6]) * conj(ar[6; 3 4])
                ϵs[pos] = max(ϵs[pos], abs(1 - abs(v)))

                ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
                ψ.AC[pos] = (al, complex(c))
            end

            ϵ = maximum(ϵs)
            ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ), typeof(envs)}

            if ϵ <= alg.tol
                @infov 2 logfinish!(log, iter, ϵ, expectation_value(ψ, H, envs))
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ, expectation_value(ψ, H, envs))
            else
                @infov 3 logiter!(log, iter, ϵ, expectation_value(ψ, H, envs))
            end
        end
    end
    return ψ, envs, ϵ
end


function find_groundstate(ψL,ψR,H, alg::Union{NH_DMRG, NH_DMRG2}, envs...; kwargs...)
    return find_groundstate!(copy(ψL),copy(ψR),H, alg, envs...; kwargs...)
end
function find_groundstate(ψ,H, alg::Union{NH_DMRG, NH_DMRG2}, envs...; kwargs...)
    return find_groundstate!(copy(ψ),copy(ψ),H, alg, envs...; kwargs...)
end