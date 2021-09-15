using Distributions
using LinearAlgebra
using PolynomialRoots
using Dierckx
using SparseArrays
using IterativeSolvers
using IncompleteLU
using Plots

struct BandedMatrix{T <: Real} #rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju
    rij::Vector{T}
    A1ilj::Vector{T}
    A1ij::Vector{T}
    A1iuj::Vector{T}
    A2ijl::Vector{T}
    A2ij::Vector{T}
    A2iju::Vector{T}
    L::Int
    M::Int
end
function LinearAlgebra.size(A::BandedMatrix{T},d::Int) where {T}
    return A.L*A.M
end

function Base.:*(A::BandedMatrix{T}, B::Vector{T}) where {T}
Y = zeros(T, size(A,1))
LinearAlgebra.mul!(Y,A,B)
return Y
end

function LinearAlgebra.mul!(Y::Vector{T},A::BandedMatrix{T},B::Vector{T}) where {T}
    L = A.L
    M = A.M
        @inbounds for j = 1:L
            jm = (j - 1) * M
        i = 1
        index = i + jm
        Y[index] = B[index] +(A.A1ij[index]+A.A2ij[index])* B[index] + A.A1iuj[index] * B[index + 1]
        i = M
        index = i + jm
        Y[index] = B[index] +( A.A1ij[index]+A.A2ij[index]) * B[index] + A.A1ilj[index] * B[index - 1]
        @inbounds @simd for i = 2:M - 1
            index = i + jm
            if j == 1
                Y0ij = B[index] +((A.A1ij[index] + A.A2ij[index]) * B[index] + A.A1iuj[index] * B[index + 1] + A.A1ilj[index] * B[index - 1] + A.A2iju[index] * B[index+ M])
                Y[index] = Y0ij
            elseif j == L
                Y0ij = B[index] +((A.A1ij[index] + A.A2ij[index]) * B[index] + A.A1iuj[index] * B[index + 1] + A.A1ilj[index] * B[index - 1] + A.A2ijl[index] * B[index - M])
                Y[index] = Y0ij
            else
                Y0ij = B[index] +((A.A1ij[index] + A.A2ij[index]) * B[index] + A.A1iuj[index] * B[index + 1] + A.A1ilj[index] * B[index - 1] + A.A2iju[index] * B[index+ M] + A.A2ijl[index] * B[index - M])
                Y0ij += A.rij[index] * (B[index+1+M] - B[index + 1 - M] + B[index - 1 - M] - B[index - 1 + M])
                Y[index] = Y0ij
            end
        end
    end
end

function solveGS2(
    tol::Real,
    maxIter::Int,
    a::Real,
    rij::Vector{T},
    A1ilj::Vector{T},
    A1ij::Vector{T},
    A1iuj::Vector{T},
    A2ijl::Vector{T},
    A2ij::Vector{T},
    A2iju::Vector{T},
    F::Vector{T},
    Y::Vector{T},
    useDirichlet::Bool,
    lbValue::Real,
    M::Int,
    L::Int) where {T}
    #tol=1e-20
    useAcceleration = true
    error =0
    sorLoops = 0
    X0 = copy(Y)
    X1 = copy(Y)
    Y0 = copy(Y)
    X2 = copy(Y)
    Y1 = copy(Y)
    while true
        error = 0
        @inbounds for j = 1:L
            i = 1
            index = i + (j - 1) * M
            back =  (A1iuj[index] * Y[index+1])
            y =  (F[index] - back) / (1 + (A1ij[index] + A2ij[index]))
            Y[index] = y
            if useDirichlet
                Y[index] = lbValue #update boundary
            end
            error += (y - X2[index])^2
            i = M
            index = i + (j - 1) * M
            back =  (A1ilj[index] * Y[index-1])
            y = (F[index] - back) / (1 + (A1ij[index] + A2ij[index]))
            x1 = Y[index]
            Y[index] = y
            error += (y - X2[index])^2
            jm = (j - 1) * M
            @inbounds @simd for i = 2:M-1
                index = i + jm
                if j == 1
                    back =  (A1iuj[index] * Y[index+1] + A1ilj[index] * Y[index-1] + A2iju[index] * Y[index+M])
                elseif j == L
                    back =  (A1iuj[index] * Y[index+1] + A1ilj[index] * Y[index-1] + A2ijl[index] * Y[index-M])
                else
                    back =  (A1iuj[index] * Y[index+1] + A1ilj[index] * Y[index-1] + A2iju[index] * Y[index+M] +
                            A2ijl[index] * Y[index-M])
                    back +=  rij[index] * (Y[index + 1 + M] - Y[index+1-M] + Y[index-1-M] - Y[index-1+M])
                end
                y = (F[index] - back) / (1 + (A1ij[index] + A2ij[index]))
                Y[index] = y
                error += (y - X2[index])^2
            end
        end
 if error <= tol || sorLoops == maxIter
     break
end
#Anderson acceleration m=1
        if sorLoops == 0 || !useAcceleration
            Y0[1:end] = Y
            X1[1:end] = Y
            X2[1:end] = Y
        elseif sorLoops == 1
             num = 0.0
             denom = 0.0
            @inbounds @simd for i = 1:M*L
                f1i = Y[i]-X1[i]
                f0i = Y0[i]-X0[i]
                num += f1i * (f0i - f1i)
                 denom += (f0i - f1i)^2
            end
            # f0mf1 = (Y0-X0-Y+X1)
            # num = dot(Y-X1, f0mf1)
            # denom = dot(f0mf1,f0mf1)
            theta = -num / denom
            @inbounds @simd for i = 1:M*L
                 yy = Y[i] + theta*(Y0[i]-Y[i])
                 X0[i] = X1[i]
                 Y0[i] = Y[i]
                 Y[i] = yy
                 X1[i] = yy
            end
        #    Y .= Y + theta * (Y0-Y)
            # X0 = copy(X1)
            # X1 = copy(Y)
            Y1[1:end] = Y
            X2[1:end] = X1
        else
            a1 = 0.0
            a2 = 0.0
            b1 = 0.0
            b2 = 0.0
            c1 = 0.0
            c2 = 0.0
            @inbounds @simd for i = 1:M*L
                f1i = Y1[i]-X1[i]
                f0i = Y0[i]-X0[i]
                f2i = Y[i]-X2[i]
                a1 += (f2i) * (f1i - f2i)
                b1 += (f1i - f2i)^2
                c1 += (f1i - f2i)^2
                a2 += f2i * (f0i - f2i)
                c2 += (f0i - f2i)^2
            end
            b2 = c1
            theta2 = (a1 - b1/b2*a2) / (-c1 + b1/b2*c2)
            theta1 = -(a1 + theta2*c1) / b1
            @inbounds @simd for i = 1:M*L
                 yy = Y[i] + theta1*(Y1[i]-Y[i])+theta2*(Y0[i]-Y[i])
                 X0[i] = X1[i]
                 Y0[i] = Y1[i]
                 X1[i] = X2[i]
                 Y1[i] = Y[i]
                 Y[i] = yy
                 X2[i] = yy
            end
        end

        sorLoops += 1
    end
    if sorLoops == maxIter
        println("Anderson did not converge, error= ", error)
    end
    #    println(sorLoops, " ",error)
    #return Y
    return sorLoops
end

function solveGS(
    tol::Real,
    maxIter::Int,
    a::Real,
    rij::Vector{T},
    A1ilj::Vector{T},
    A1ij::Vector{T},
    A1iuj::Vector{T},
    A2ijl::Vector{T},
    A2ij::Vector{T},
    A2iju::Vector{T},
    F::Vector{T},
    Y::Vector{T},
    useDirichlet::Bool,
    lbValue::Real,
    M::Int,
    L::Int) where {T}
    #tol=1e-20
    useAcceleration = true
    error =0
    sorLoops = 0
    X0 = copy(Y)
    X1 = copy(Y)
    Y0 = copy(Y)
    while true
        error = 0
        @inbounds for j = 1:L
            i = 1
            index = i + (j - 1) * M
            back =  (A1iuj[index] * Y[index+1])
            y =  (F[index] - back) / (1 + (A1ij[index] + A2ij[index]))
            Y[index] = y
            if useDirichlet
                Y[index] = lbValue #update boundary
            end
            error += (y - X1[index])^2
            i = M
            index = i + (j - 1) * M
            back =  (A1ilj[index] * Y[index-1])
            y = (F[index] - back) / (1 + (A1ij[index] + A2ij[index]))
            x1 = Y[index]
            Y[index] = y
            error += (y - X1[index])^2
            jm = (j - 1) * M
            @inbounds @simd for i = 2:M-1
                index = i + jm
                if j == 1
                    back =  (A1iuj[index] * Y[index+1] + A1ilj[index] * Y[index-1] + A2iju[index] * Y[index+M])
                elseif j == L
                    back =  (A1iuj[index] * Y[index+1] + A1ilj[index] * Y[index-1] + A2ijl[index] * Y[index-M])
                else
                    back =  (A1iuj[index] * Y[index+1] + A1ilj[index] * Y[index-1] + A2iju[index] * Y[index+M] +
                            A2ijl[index] * Y[index-M])
                    back +=  rij[index] * (Y[index + 1 + M] - Y[index+1-M] + Y[index-1-M] - Y[index-1+M])
                end
                y = (F[index] - back) / (1 + (A1ij[index] + A2ij[index]))
                Y[index] = y
                error += (y - X1[index])^2
            end
        end
 if error <= tol || sorLoops == maxIter
     break
end
#Anderson acceleration m=1
        if sorLoops == 0 || !useAcceleration
            Y0[1:end] = Y
            X1[1:end] = Y
         else
             num = 0.0
             denom = 0.0
            @inbounds @simd for i = 1:M*L
                f1i = Y[i]-X1[i]
                f0i = Y0[i]-X0[i]
                num += f1i * (f0i - f1i)
                 denom += (f0i - f1i)^2
            end
            # f0mf1 = (Y0-X0-Y+X1)
            # num = dot(Y-X1, f0mf1)
            # denom = dot(f0mf1,f0mf1)
            theta = -num / denom
            @inbounds @simd for i = 1:M*L
                 yy = Y[i] + theta*(Y0[i]-Y[i])
                 X0[i] = X1[i]
                 Y0[i] = Y[i]
                 Y[i] = yy
                 X1[i] = yy
            end
        #    Y .= Y + theta * (Y0-Y)
            # X0 = copy(X1)
            # X1 = copy(Y)
        end

        sorLoops += 1
    end
    if sorLoops == maxIter
        println("Anderson did not converge, error= ", error)
    end
    #    println(sorLoops, " ",error)
    #return Y
    return sorLoops
end
function solveJacobi(
    tol::Real,
    maxIter::Int,
    a::Real,
    rij::Vector{T},
    A1ilj::Vector{T},
    A1ij::Vector{T},
    A1iuj::Vector{T},
    A2ijl::Vector{T},
    A2ij::Vector{T},
    A2iju::Vector{T},
    F::Vector{T},
    Y::Vector{T},
    useDirichlet::Bool,
    lbValue::Real,
    M::Int,
    L::Int) where {T}
    tol=1e-20
    useAcceleration = true
    error =0
    sorLoops = 0
    X0 = copy(Y)
    X1 = copy(Y)
    Y0 = copy(Y)
    while true
        error = 0
        @inbounds for j = 1:L
            i = 1
            index = i + (j - 1) * M
            back =  (A1iuj[index] * X1[index+1])
            y =  (F[index] - back) / (1 + (A1ij[index] + A2ij[index]))
            Y[index] = y
            if useDirichlet
                Y[index] = lbValue #update boundary
            end
            error += (y - X1[index])^2
            i = M
            index = i + (j - 1) * M
            back =  (A1ilj[index] * X1[index-1])
            y = (F[index] - back) / (1 + (A1ij[index] + A2ij[index]))
            x1 = Y[index]
            Y[index] = y
            error += (y - X1[index])^2
            jm = (j - 1) * M
            @inbounds @simd for i = 2:M-1
                index = i + jm
                if j == 1
                    back =  (A1iuj[index] * X1[index+1] + A1ilj[index] * X1[index-1] + A2iju[index] * X1[index+M])
                elseif j == L
                    back =  (A1iuj[index] * X1[index+1] + A1ilj[index] * X1[index-1] + A2ijl[index] * X1[index-M])
                else
                    back =  (A1iuj[index] * X1[index+1] + A1ilj[index] * X1[index-1] + A2iju[index] * X1[index+M] +
                            A2ijl[index] * X1[index-M])
                    back +=  rij[index] * (X1[index + 1 + M] - X1[index+1-M] + X1[index-1-M] - X1[index-1+M])
                end
                y = (F[index] - back) / (1 + (A1ij[index] + A2ij[index]))
                Y[index] = y
                error += (y - X1[index])^2
            end
        end
 if error <= tol || sorLoops == maxIter
     break
end
#Anderson acceleration m=1
        if sorLoops == 0 || !useAcceleration
            Y0[1:end] = Y
            X1[1:end] = Y
         else
             num = 0.0
             denom = 0.0
            @inbounds @simd for i = 1:M*L
                f1i = Y[i]-X1[i]
                f0i = Y0[i]-X0[i]
                num += f1i * (f0i - f1i)
                 denom += (f0i - f1i)^2
            end
            # f0mf1 = (Y0-X0-Y+X1)
            # num = dot(Y-X1, f0mf1)
            # denom = dot(f0mf1,f0mf1) 0.013609815396568692
            theta = -num / denom
            #println("theta ",theta)
            @inbounds @simd for i = 1:M*L
                 yy = Y[i] + theta*(Y0[i]-Y[i])
                 X0[i] = X1[i]
                 Y0[i] = Y[i]
                 Y[i] = yy
                 X1[i] = yy
            end
        #    Y .= Y + theta * (Y0-Y)
            # X0 = copy(X1)
            # X1 = copy(Y)
        end

        sorLoops += 1
    end
    if sorLoops == maxIter
        println("Anderson did not converge, error= ", error)
    end
    #println(sorLoops, " ",error)
    #return Y
end


function solveSOR(
    tol::Real,
    maxIter::Int,
    a::Real,
    rij::Vector{T},
    A1ilj::Vector{T},
    A1ij::Vector{T},
    A1iuj::Vector{T},
    A2ijl::Vector{T},
    A2ij::Vector{T},
    A2iju::Vector{T},
    F::Vector{T},
    Y::Vector{T},
    useDirichlet::Bool,
    lbValue::Real,
    M::Int,
    L::Int) where {T}
    useAcceleration = false
tol=1e-20
    cnOmega =0.0
	if cnOmega == 0.0
		rhoG = 0.0
		j = floor(Int,L/2)
        i = floor(Int,M/2)
        index = i + (j - 1) * M
			back =  abs(A1iuj[index]) + abs(A1ilj[index]) + abs(A2iju[index]) +	abs(A2ijl[index])
			back +=  abs(rij[index])*4
			estimate = back / (abs(1+A1ij[index] + A2ij[index]))
			if estimate > rhoG
				rhoG = estimate
		    end
		# println("rhoG ",rhoG)
		rhoG = max(0.0, min(1.0, rhoG))
		cnOmega = 2.0 / (1 + sqrt(1-rhoG^2))
#		println("omega ",cnOmega)
	end
    error =0
    sorLoops = 0
    X0 = copy(Y)
    X1 = copy(Y)
    Y0 = copy(Y)
    while true
        error = 0
        @inbounds for j = 1:L
            i = 1
            index = i + (j - 1) * M
            back =  (A1iuj[index] * Y[index+1])
            y =  (F[index] - back) / (1 + (A1ij[index] + A2ij[index]))
            Y[index] = (1-cnOmega)*X1[index] + cnOmega*y
            if useDirichlet
                Y[index] = lbValue #update boundary
            end
            error += (y - X1[index])^2
            i = M
            index = i + (j - 1) * M
            back =  (A1ilj[index] * Y[index-1])
            y = (F[index] - back) / (1 + (A1ij[index] + A2ij[index]))
            x1 = Y[index]
            Y[index] = (1-cnOmega)*X1[index] + cnOmega*y
            error += (Y[index] - X1[index])^2
            jm = (j - 1) * M
            @inbounds @simd for i = 2:M-1
                index = i + jm
                if j == 1
                    back =  (A1iuj[index] * Y[index+1] + A1ilj[index] * Y[index-1] + A2iju[index] * Y[index+M])
                elseif j == L
                    back =  (A1iuj[index] * Y[index+1] + A1ilj[index] * Y[index-1] + A2ijl[index] * Y[index-M])
                else
                    back =  (A1iuj[index] * Y[index+1] + A1ilj[index] * Y[index-1] + A2iju[index] * Y[index+M] +
                            A2ijl[index] * Y[index-M])
                    back +=  rij[index] * (Y[index + 1 + M] - Y[index+1-M] + Y[index-1-M] - Y[index-1+M])
                end
                y = (1-cnOmega)*X1[index] + cnOmega*(F[index] - back) / (1 + (A1ij[index] + A2ij[index]))
                Y[index] = y
                error += (y - X1[index])^2
            end
        end
 if error <= tol || sorLoops == maxIter
     break
end
#Anderson acceleration m=1
        if sorLoops == 0 || !useAcceleration
            Y0[1:end] = Y
            X1[1:end] = Y
         else
             num = 0.0
             denom = 0.0
            @inbounds @simd for i = 1:M*L
                f1i = Y[i]-X1[i]
                f0i = Y0[i]-X0[i]
                num += f1i * (f0i - f1i)
                 denom += (f0i - f1i)^2
            end
            # f0mf1 = (Y0-X0-Y+X1)
            # num = dot(Y-X1, f0mf1)
            # denom = dot(f0mf1,f0mf1)
            theta = -num / denom
            @inbounds @simd for i = 1:M*L
                 yy = Y[i] + theta*(Y0[i]-Y[i])
                 X0[i] = X1[i]
                 Y0[i] = Y[i]
                 Y[i] = yy
                 X1[i] = yy
            end
        #    Y .= Y + theta * (Y0-Y)
            # X0 = copy(X1)
            # X1 = copy(Y)
        end

        sorLoops += 1
    end
    if sorLoops == maxIter
        println("Anderson did not converge, error= ", error)
    end
    return sorLoops
    #return Y
end

mutable struct ASORIterable{T,solT,vecT,rhsT,numT <: Real}
    U::IterativeSolvers.StrictlyUpperTriangular
    L::IterativeSolvers.FastLowerTriangular
    ω::numT

    x::solT
    next::vecT
    b::rhsT

    maxiter::Int
    tolerance::numT
end

start(::ASORIterable) = 1
done(s::ASORIterable, iteration::Int) = iteration > s.maxiter || norm(s.x - s.next) < s.tolerance
function iterate(s::ASORIterable{T,solT,vecT,rhsT,numT}) where {T,solT,vecT,rhsT,numT}
	return iterate(s, start(s))
end
function Base.iterate(s::ASORIterable{T,solT,vecT,rhsT,numT}, iteration::Int = start(s)) where {T,solT,vecT,rhsT,numT}
    if done(s, iteration) return nothing end

    # next = b - U * x
    IterativeSolvers.gauss_seidel_multiply!(-one(T), s.U, s.x, one(T), s.b, s.next)

    # next = ω * inv(L) * next + (1 - ω) * x
    IterativeSolvers.forward_sub!(s.ω, s.L, s.next, one(T) - s.ω, s.x)

    # Switch current and next iterate
    s.x, s.next = s.next, s.x

    nothing, iteration + 1
end

function asor_iterable(x::AbstractVector, A::SparseMatrixCSC, b::AbstractVector, ω::Real; maxiter::Int = 10, tolerance::Real = 0.0)
    D = IterativeSolvers.DiagonalIndices(A)
    T = eltype(x)
    ASORIterable{T,typeof(x),typeof(x),typeof(b),eltype(ω)}(IterativeSolvers.StrictlyUpperTriangular(A, D), IterativeSolvers.FastLowerTriangular(A, D), ω,
        x, similar(x), b, maxiter, tolerance)
end

"""
    asor!(x, A::SparseMatrixCSC, b, ω::Real; maxiter=10, tolerance=0)
Performs exactly `maxiter` SOR iterations with relaxation parameter `ω`.
Allocates a temporary vector and precomputes the diagonal indices.
Throws `LinearAlgebra.SingularException` when the diagonal has a zero. This check
is performed once beforehand.
"""
function asor!(x::AbstractVector, A::SparseMatrixCSC, b::AbstractVector, ω::Real; maxiter::Int = 10, tolerance::Real = 0.0)
    iterable = asor_iterable(x, A, b, ω, maxiter = maxiter, tolerance = tolerance)
    for item = iterable end
    iterable.x
end

function computeGamma(x::Vector{T}, y::Vector{T}) where {T}
    @views h = x[2:end] - x[1:end - 1]
    gamma = zeros(length(x))
    @views gamma[2:end - 1] = 2 * (h[1:end - 1] .* y[3:end] + h[2:end] .* y[1:end - 2] - (h[1:end - 1] + h[2:end]) .* y[2:end - 1] ) ./ (h[1:end - 1].^2 .* h[2:end] + h[1:end - 1] .* h[2:end].^2)
    gamma[1] = 2 * (h[1] * y[3] + h[2] * y[1] - (h[1] + h[2]) * y[2]) / (h[1]^2 * h[2] + h[1] * h[2]^2)
    gamma[end] = 2 * (h[end - 1] * y[end] + h[end] * y[end - 2] - (h[end] + h[end - 1]) * y[end - 1]) / (h[end]^2 * h[end - 1] + h[end] * h[end - 1]^2)
    return gamma
end

function computeDeltaGamma(x::Vector{T}, y::Vector{T}) where {T}
   	h0 = (x[2] - x[1])
    h1 = (x[3] - x[2])
    delta = zeros(length(x))
    gamma = zeros(length(x))
   	delta[1] = (y[2] - y[1]) / h0
   	gamma[1] = 2 * (h0 * y[3] + h1 * y[1] - (h0 + h1) * y[2]) / (h0 * h1 * (h0 + h1))
   	@simd for i = 2:length(x) - 1
      		him = x[i] - x[i - 1]
      		hi = x[i + 1] - x[i]
      		denom = him * hi * (him + hi)
      		delta[i] = (him * him * y[i + 1] - hi * hi * y[i - 1] + (hi * hi - him * him) * y[i]) / denom
      		gamma[i] = 2 * (him * y[i + 1] + hi * y[i - 1] - (hi + him) * y[i]) / denom
    end
   	hnm = (x[end] - x[end - 1])
   	hnm2 = (x[end - 1] - x[end - 2])
   	delta[end] = (y[end] - y[end - 1]) / hnm
    gamma[end] = 2 * (hnm2 * y[end] + hnm * y[end - 2] - (hnm + hnm2) * y[end - 1]) / (hnm * hnm2 * (hnm + hnm2))
    return delta, gamma
end
struct CollocationFunction{T <: Real}
    a::Vector{T}
    b::Vector{T}
    c::Vector{T}
    x::Vector{T}
    leftSlope::T
    rightSlope::T
    T::T
end

function evaluateSlice(self::CollocationFunction, z::T) where {T}
    if z <= self.x[1]
        return self.leftSlope * (z - self.x[1]) + self.a[1]
    elseif z >= self.x[end]
        return self.rightSlope * (z - self.x[end]) + self.a[end]
    end
    i = searchsortedfirst(self.x, z)  # x[i-1]<z<=x[i]
    if i > 1
        i -= 1
    end
    h = z - self.x[i]
    return self.a[i] + h * (self.b[i] + h * self.c[i])
end

function evaluate(self::CollocationFunction, t::T, z::T) where {T}
    return t / self.T * evaluateSlice(self, z) + (1.0 - t / self.T) * z #linear interpolation between slice at t=0 and slice T.
end

function evaluateSliceDerivative(self::CollocationFunction, z::T) where {T}
    if z <= self.x[1]
        return self.leftSlope
    elseif z >= self.x[end]
        return self.rightSlope
    end
    i = searchsortedfirst(self.x, z)  # x[i-1]<z<=x[i]
    if i > 1
        i -= 1
    end
    h = z - self.x[i]
    return self.b[i] + 2 * h * self.c[i]
end

function evaluateTimeDerivative(self::CollocationFunction, z::T) where {T}
    return 1.0 / self.T * evaluateSlice(self, z) - z / self.T #linear interpolation between slice at t=0 and slice T.
end

function makeSlice(self::CollocationFunction, t::T) where {T}
    A2 = [self.a[i] * t / self.T + (1.0 - t / self.T) * self.x[i] for i = 1:length(self.a)]
    B2 = [Bi * t / self.T + 1.0 - t / self.T for Bi in self.b]
    C2 = [Ci * t / self.T for Ci in self.c]
    l2 =  self.leftSlope * t / self.T + (1.0 - t / self.T)
    r2 =  self.rightSlope * t / self.T + (1.0 - t / self.T)
    return CollocationFunction(A2, B2, C2, self.x, l2, r2, self.T)
end
function solve(self::CollocationFunction, strike::T) where {T}
    if strike <= self.a[1]
        sn = self.leftSlope
        return (strike - self.a[1]) / sn + self.x[1]
    elseif strike >= self.a[end]
        sn = self.rightSlope
        return (strike - self.a[end]) / sn + self.x[end]
    end
    i = searchsortedfirst(self.a, strike)  # a[i-1]<strike<=a[i]
    if i == 1
        i += 1
    end
    if abs(self.a[i] - strike) < 1e-10
        return self.x[i]
    elseif abs(self.a[i - 1] - strike) < 1e-10
        return self.x[i - 1]
    end
    x0 = self.x[i - 1]
    c = self.c[i - 1]
    b = self.b[i - 1]
    a = self.a[i - 1]
    d = 0
    cc = a + x0 * (-b + x0 * (c - d * x0)) - strike
    bb = b + x0 * (-2 * c + x0 * 3 * d)
    aa = -3 * d * x0 + c

    allck = PolynomialRoots.roots([cc,bb,aa])
    for ck in allck
        if abs(imag(ck)) < 1e-10 && real(ck) >= self.x[i - 1] - 1e-10 && real(ck) <= self.x[i] + 1e-10
            return real(ck)
        end
    end
    println("ERRROR !!! no roots found in range", allck, " ", strike, " ", aa, bb, cc, " ", i, " ", self.x[i - 1], " ", self.x[i])
    return self.x[1]
end


function updatePayoffExplicitTrans(F::Vector{T}, useDirichlet::Bool, lbValue::Real, M::Int, L::Int) where T
    if useDirichlet
        @simd for j = 1:L
            F[1 + (j - 1) * M] =  lbValue
        end
    end
end

function makeJacobians(ti::Real, cFunc::CollocationFunction{T}, S::Vector{T}) where T
    cFunci = makeSlice(cFunc, ti)
    Sc = [solve(cFunci, Si) for Si in S]
    Jc  = [evaluateSliceDerivative(cFunci, Sci) for Sci in Sc]
    @views Jch = 0.5 * (Jc[2:M] + Jc[1:M - 1])
    Jct = [-evaluateTimeDerivative(cFunc, Sci) for Sci in Sc]
    return Sc, Jc, Jch, Jct
end

function makeSystem(useExponentialFitting::Bool, upwindingThreshold::Real, dt::Real, Sc::Vector{T}, Jc::Vector{T}, Jch::Vector{T}, Jct::Vector{T}, S::Vector{T}, J::Vector{T}, Jm::Vector{T}, V::Vector{T}, JV::Vector{T}, JVm::Vector{T}, hm::Real, hl::Real, kappa::Real, theta::Real, sigma::Real, rho::Real, r::Real, q::Real, useDirichlet::Bool, M::Int, L::Int) where T
    rij = zeros(T, L * M)
    A1ilj = zeros(T, L * M)
    A1ij = zeros(T, L * M)
    A1iuj = zeros(T, L * M)
    A2ij = zeros(T, L * M)
    A2ijl = zeros(T, L * M)
    A2iju = zeros(T, L * M)
    for j = 1:L
        jm =  (j - 1) * M
        i = 1
        index = i + jm
        if useDirichlet
            A1ij[index] = 0
            A2ij[index] = 0
        else
            drifti = (r - q) * Sc[i] * Jc[i] - Jct[i]
            A1iuj[index] = -dt * drifti / (Jm[i + 1] * hm)
            A1ij[index] = -dt * (-r * 0.5) - A1iuj[index]
            A2ij[index] = -dt * (-r * 0.5)
        end
        i = M
        index = i + jm
        drifti = (r - q) * Sc[i] * Jc[i] - Jct[i]
        A1ilj[index] = dt * drifti / (Jm[i] * hm)
        A1ij[index] = -dt * (-r * 0.5) - A1ilj[index]
        A2ij[index] = -dt * (-r * 0.5)

        @simd for i = 2:M - 1
            index = i + jm
            svi = Sc[i]^2 * V[j] * Jc[i] / J[i]
            drifti = (r - q) * Sc[i] - Jct[i] / Jc[i]
            if useExponentialFitting
                if abs(drifti * hm / svi) > upwindingThreshold
                    svi = drifti * hm / tanh(drifti * hm / svi)
                    # svi = svi + 0.5 * abs(drifti) * hm
                end
            end
            svi /= (hm * hm)
            drifti = drifti * Jc[i] / (2 * J[i] * hm)
            A1iuj[index] = -dt * (0.5 * svi * Jch[i] / (Jm[i + 1]) + drifti )
            A1ij[index] = -dt * (-svi * 0.5  * (Jch[i] / Jm[i + 1] + Jch[i - 1] / Jm[i]) - r * 0.5)
            A1ilj[index] = -dt * (0.5 * svi * Jch[i - 1] / (Jm[i]) - drifti)
        end
    end

    j = 1
    jm = (j - 1) * M
    driftj = kappa * (theta - V[j])
    for i = 2:M - 1
        index = i + jm
        A2iju[index] = -dt * (driftj / (JVm[j + 1] * hl))
        A2ij[index] = -dt * (-r * 0.5) - A2iju[index]
    end
    j = L
    jm = (j - 1) * M
    driftj = kappa * (theta - V[j])
    @simd for i = 2:M - 1
        index = i + jm
        A2ijl[index] = dt * (driftj / (JVm[j] * hl))
        A2ij[index] = -dt * (-r * 0.5) - A2ijl[index]
    end
    for j = 2:L - 1
        driftj = kappa * (theta - V[j])
        svj = sigma^2 * V[j] / JV[j]
        if useExponentialFitting
            if driftj != 0 && abs(driftj * hl / svj) > 1.0
                # svj = svj + 0.5 * abs(driftj) * hl
                svj = driftj * hl / tanh(driftj * hl / svj)
            end
        end
        svj /= (hl * hl)
        driftj /= (2 * JV[j] * hl)
        jm =  (j - 1) * M
        rCoeff = -dt * 0.25 * rho * sigma * V[j] / (JV[j] * hl * hm)
        a2ijuCoeff = -dt * (0.5 * svj / (JVm[j + 1]) + driftj )
        a2ijCoeff  = -dt * (-r * 0.5 - svj * 0.5  * (1.0 / JVm[j + 1] + 1.0 / JVm[j]))
        a2ijlCoeff = -dt * (svj * 0.5 / (JVm[j]) - driftj)
        @simd for i = 2:M - 1
            index = i + jm
            A2iju[index] = a2ijuCoeff
            A2ij[index] = a2ijCoeff
            A2ijl[index] = a2ijlCoeff
            rij[index] = rCoeff*Sc[i] * Jc[i] / ( J[i])
        end
    end
    return rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju
end

function solveTrapezoidal1(aLeft::Real, aRight::Real, A1ilj::Vector{T}, A1ij::Vector{T}, A1iuj::Vector{T}, A1iljNew::Vector{T}, A1ijNew::Vector{T}, A1iujNew::Vector{T}, useDirichlet::Bool, Y0::Vector{T}, F::Vector{T}, Y1::Vector{T}, M::Int, L::Int) where T
    lhs1d = zeros(T, M)
    lhs1dl = zeros(T, M - 1)
    lhs1du = zeros(T, M - 1)
    rhs1d = zeros(T, M)
    rhs1dl = zeros(T, M - 1)
    rhs1du = zeros(T, M - 1)
    Y0r = zeros(T, M)
    @inbounds for j = 1:L
        @views rhs1d[1:M] = A1ij[1 + (j - 1) * M:j * M]
        @views rhs1du[1:M - 1] = A1iuj[1 + (j - 1) * M:j * M - 1]
        @views  rhs1dl[1:M - 1] = A1ilj[2 + (j - 1) * M:j * M]
        @views lhs1d[1:M] = A1ijNew[1 + (j - 1) * M:j * M]
        @views lhs1du[1:M - 1] = A1iujNew[1 + (j - 1) * M:j * M - 1]
        @views lhs1dl[1:M - 1] = A1iljNew[2 + (j - 1) * M:j * M]
        A1 = Tridiagonal(rhs1dl, rhs1d, rhs1du)
        A1New = Tridiagonal(lhs1dl, lhs1d, lhs1du)
        @views Y0r .= Y0[1 + (j - 1) * M:j * M] .+ aRight .* (A1 * F[1 + (j - 1) * M:j * M])
        if useDirichlet
            Y0r[1] =  0 #update boundary
        end
        Y1[1 + (j - 1) * M:j * M] = (I + aLeft .* A1New) \  Y0r
    end
end

#solve  Y2 = (I+a*A2New) \ (Y1+a*A2*F)
function solveTrapezoidal2(aLeft::Real, aRight::Real, A2ijl::Vector{T}, A2ij::Vector{T}, A2iju::Vector{T}, A2ijlNew::Vector{T}, A2ijNew::Vector{T}, A2ijuNew::Vector{T}, useDirichlet::Bool, Y1::Vector{T}, F::Vector{T}, Y2::Vector{T}, M::Int, L::Int) where T
    lhs2d = zeros(T, L)
    lhs2dl = zeros(T, L - 1)
    lhs2du = zeros(T, L - 1)
    rhs2d = zeros(T, L)
    rhs2dl = zeros(T, L - 1)
    rhs2du = zeros(T, L - 1)
    indices = Array{Int}(undef, L)
    Y2r = zeros(T, L)
    @inbounds for i = 1:M
      @inbounds  @simd for j = 2:L - 1
            index = i + (j - 1) * M
            indices[j] = index
            rhs2d[j] = A2ij[index]
            rhs2dl[j - 1] = A2ijl[index]
            rhs2du[j] = A2iju[index]
            lhs2d[j] = A2ijNew[index]
            lhs2dl[j - 1] = A2ijlNew[index]
            lhs2du[j] = A2ijuNew[index]
        end
        indices[1] = i
        indices[L] = i + (L - 1) * M
        rhs2d[1] = A2ij[i]
        rhs2d[L] = A2ij[i + (L - 1) * M]
        rhs2du[1] = A2iju[i]
        rhs2dl[L - 1] = A2ijl[i + (L - 1) * M]
        lhs2d[1] = A2ijNew[i]
        lhs2d[L] = A2ijNew[i + (L - 1) * M]
        lhs2du[1] = A2ijuNew[i]
        lhs2dl[L - 1] = A2ijlNew[i + (L - 1) * M]
        A2 = Tridiagonal(rhs2dl, rhs2d, rhs2du)
        A2New = Tridiagonal(lhs2dl, lhs2d, lhs2du)
        @views Y2r .= Y1[indices] .+ aRight .* (A2 * F[indices])
        if useDirichlet && i == 1
            Y2r[1:L] .=  0 #update boundary
        end
        Y2[indices] = (I + aLeft .* A2New) \  Y2r
    end
end

function implicitStep1(a::Real, A1iljNew::Vector{T}, A1ijNew::Vector{T}, A1iujNew::Vector{T}, useDirichlet::Bool, Y0::Vector{T}, Y1::Vector{T}, M::Int, L::Int) where T
    lhs1d = zeros(T, M)
    lhs1dl = zeros(T, M - 1)
    lhs1du = zeros(T, M - 1)
    for j = 1:L
        @views lhs1d[1:M] = A1ijNew[1 + (j - 1) * M:j * M]
        @views lhs1du[1:M - 1] = A1iujNew[1 + (j - 1) * M:j * M - 1]
        @views lhs1dl[1:M - 1] = A1iljNew[2 + (j - 1) * M:j * M]
        A1New = Tridiagonal(lhs1dl, lhs1d, lhs1du)
        if useDirichlet
            Y0[1 + (j - 1) * M] =  0 #update boundary
        end
        @views Y1[1 + (j - 1) * M:j * M] = (I + a .* A1New) \  Y0[1 + (j - 1) * M:j * M]
    end
end

#solve  Y2 = (I+a*A2New) \ (Y1)
function implicitStep2(a::Real, A2ijlNew::Vector{T}, A2ijNew::Vector{T}, A2ijuNew::Vector{T}, useDirichlet::Bool, Y1::Vector{T}, Y2::Vector{T}, M::Int, L::Int) where T
    lhs2d = zeros(T, L)
    lhs2dl = zeros(T, L - 1)
    lhs2du = zeros(T, L - 1)
    indices = Array{Int}(undef, L)
    for i = 1:M
        @inbounds @simd for j = 2:L - 1
            indices[j] = i + (j - 1) * M
            lhs2d[j] = A2ijNew[i + (j - 1) * M]
            lhs2dl[j - 1] = A2ijlNew[i + (j - 1) * M]
            lhs2du[j] = A2ijuNew[i + (j - 1) * M]
        end
        indices[1] = i
        indices[L] = i + (L - 1) * M
        lhs2d[1] = A2ijNew[i]
        lhs2d[L] = A2ijNew[i + (L - 1) * M]
        lhs2du[1] = A2ijuNew[i]
        lhs2dl[L - 1] = A2ijlNew[i + (L - 1) * M]
        A2New = Tridiagonal(lhs2dl, lhs2d, lhs2du)
        if useDirichlet && i == 1
            Y1[indices] .=  0 #update boundary
        end
        Y2[indices] = (I + a .* A2New) \  Y1[indices]
    end
end

function explicitStep(rij::Vector{T}, A1ilj::Vector{T}, A1ij::Vector{T}, A1iuj::Vector{T}, A2ijl::Vector{T}, A2ij::Vector{T}, A2iju::Vector{T}, F::Vector{T}, Y0::Vector{T}, M::Int, L::Int) where T
    explicitStep(1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Y0, M, L)
end
# function explicitStep(a::T, rij::Array{T,1}, A1ilj::Array{T,1}, A1ij::Array{T,1}, A1iuj::Array{T,1}, A2ijl::Array{T,1}, A2ij::Array{T,1}, A2iju::Array{T,1}, F::Array{T,1}, Y0::Array{T,1}, Y1::Array{T,1}, M::Int, L::Int)
#     for j = 1:L
#     i = 1
#     index = i + (j - 1) * M
#     Y1[index] = Y0[index] - a*(A1ij[index]* F[index] + A1iuj[index] * F[index + 1])
#     i = M
#     index = i + (j - 1) * M
#     Y1[index] = Y0[index] - a*( A1ij[index] * F[index] + A1ilj[index] * F[index - 1])
#     for i = 2:M - 1
#         index = i + (j - 1) * M
#         if j == 1
#             Y0ij = Y0[index] - a *((A1ij[index] + A2ij[index]) * F[index] + A1iuj[index] * F[index + 1] + A1ilj[index] * F[index - 1] + A2iju[index] * F[index+ M])
#             Y1[index] = Y0ij
#         elseif j == Lexplicit
#             Y0ij = Y0[index] - a*((A1ij[index] + A2ij[index]) * F[index] + A1iuj[index] * F[index + 1] + A1ilj[index] * F[index - 1] + A2ijl[index] * F[index - M])
#             Y1[index] = Y0ij
#         else
#             Y0ij = Y0[index] - a*((A1ij[index] + A2ij[index]) * F[index] + A1iuj[index] * F[index + 1] + A1ilj[index] * F[index - 1] + A2iju[index] * F[index+ M] + A2ijl[index] * F[index - M])
#             Y0ij -= a*rij[index] * (F[index+1+M] - F[index + 1 - M] + F[index - 1 - M] - F[index - 1 + M])
#             Y1[index] = Y0ij
#         end
#     end
# end
# end

function explicitStep1(a::Real, A1ilj::Vector{T}, A1ij::Vector{T}, A1iuj::Vector{T}, F::Vector{T}, Y0::Vector{T}, Y1::Vector{T}, M::Int, L::Int) where {T}
    @inbounds for j = 1:L
        jm = (j - 1) * M
        i = 1
        index = i + jm
        Y1[index] = Y0[index] - a * ( A1ij[index] * F[index] + A1iuj[index] * F[index + 1])
        i = M
        index = i + jm
        Y1[index] = Y0[index] - a * (A1ij[index] * F[index] + A1ilj[index] * F[index - 1])
        @inbounds @simd for i = 2:M - 1
            index = i + jm
            Y1[index]  = Y0[index] - a * ( A1ij[index] * F[index] + A1iuj[index] * F[index + 1] + A1ilj[index] * F[index - 1])
        end
    end
end


function explicitStep2(a::Real, A2ijl::Vector{T}, A2ij::Vector{T}, A2iju::Vector{T}, F::Vector{T}, Y0::Vector{T}, Y1::Vector{T}, M::Int, L::Int) where {T}
    @inbounds for j = 1:L
        jm = (j - 1) * M
        i = 1
        index = i + jm
        Y1[index] = Y0[index]
        i = M
        index = i + jm
        Y1[index] = Y0[index]
    end
             j = 1
             jm = (j - 1) * M
             @inbounds @simd for i = 2:M - 1
                index = i + jm
                    Y0ij = Y0[index] - a * (A2ij[index] * F[index] + A2iju[index] * F[index + M])
                Y1[index] = Y0ij
             end
                j = L
                jm = (j - 1) * M
              @inbounds   @simd for i = 2:M - 1
                    index = i + jm
         Y0ij =  Y0[index] - a * (A2ij[index] * F[index] + A2ijl[index] * F[index - M])
                Y1[index] = Y0ij
                end
                @inbounds for j = 2:L - 1
                    jm = (j - 1) * M
                  @inbounds   @simd for i = 2:M - 1
                        index = i + jm

                Y0ij =  Y0[index] - a * (A2ij[index] * F[index] + A2iju[index] * F[index + M] + A2ijl[index] * F[index - M])
                Y1[index] = Y0ij
            end
        end
    end


    function explicitStep(a::Real, rij::Vector{T}, A1ilj::Vector{T}, A1ij::Vector{T}, A1iuj::Vector{T}, A2ijl::Vector{T}, A2ij::Vector{T}, A2iju::Vector{T}, F::Vector{T}, Y0::Vector{T}, Y1::Vector{T}, M::Int, L::Int) where T
        explicitStep1(a, A1ilj, A1ij, A1iuj, F, Y0, Y1, M, L)
        explicitStep2(a, A2ijl, A2ij, A2iju, F, Y1, Y1, M, L)
        @inbounds for j = 2:L - 1
            jm = (j - 1) * M
            @inbounds @simd for i = 2:M - 1
                index = i + jm
                    Y0ij = a * rij[index] * (F[index + 1 +  M] - F[index + 1 - M] + F[index - 1 - M] - F[index - 1 +  M])
                    Y1[index] -= Y0ij
                end
            end
        end

function RKLStep(s::Int, a::Vector{T}, b::Vector{T}, w1::Real, rij::Vector{T}, A1ilj::Vector{T}, A1ij::Vector{T}, A1iuj::Vector{T}, A2ijl::Vector{T}, A2ij::Vector{T}, A2iju::Vector{T}, F::Vector{T}, Y::Vector{Vector{T}}, useDirichlet::Bool, lbValue::Real, M::Int, L::Int) where T
    mu1b = b[1] * w1
    explicitStep(mu1b, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Y[1], M, L)
    updatePayoffExplicitTrans(Y[1], useDirichlet, lbValue, M, L)
    MY0 = (Y[1] - F) / mu1b
    for j = 2:s
        muj = (2 * j - 1) * b[j] / (j * b[j - 1])
        mujb = muj * w1
        gammajb = -a[j - 1] * mujb
        nuj = - 1.0 * b[2] / (2.0 * b[1]) #b0 = b[1]
        Yj = Y[j]
        Yjm = Y[j - 1]
        Yjm2 = F
        if j > 2
            Yjm2 = Y[j - 2]
            nuj = -(j - 1) * b[j] / (j * b[j - 2])
        end
       @. Yj = muj * Yjm + nuj * Yjm2 + (1 - nuj - muj) * F + gammajb * MY0 # + mujb*MYjm
       explicitStep(mujb, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, Yjm, Yj, Yj, M, L)
       updatePayoffExplicitTrans(Yj, useDirichlet, lbValue, M, L)
       Y[j] = Yj
    end
end


function RKLStep(s::Int, a::Vector{T}, b::Vector{T}, w1::Real, rij::Vector{T}, A1ilj::Vector{T}, A1ij::Vector{T}, A1iuj::Vector{T}, A2ijl::Vector{T}, A2ij::Vector{T}, A2iju::Vector{T}, F::Vector{T}, Yjm2::Vector{T}, Yjm::Vector{T}, Yj::Vector{T}, useDirichlet::Bool, lbValue::Real, M::Int, L::Int) where T
    mu1b = b[1] * w1
    explicitStep(mu1b, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Yjm, M, L)
    updatePayoffExplicitTrans(Yjm, useDirichlet, lbValue, M, L)
    MY0 = (Yjm - F) / mu1b
    Yjm2 .= F
    for j = 2:s
        muj = (2 * j - 1) * b[j] / (j * b[j - 1])
        mujb = muj * w1
        gammajb = -a[j - 1] * mujb
        nuj = - 1.0 * b[2] / (2.0 * b[1]) #b0 = b[1]
        if j > 2
            nuj = -(j - 1) * b[j] / (j * b[j - 2])
        end
       @. Yj = muj * Yjm + nuj * Yjm2 + (1 - nuj - muj) * F + gammajb * MY0 # + mujb*MYjm
       explicitStep(mujb, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, Yjm, Yj, Yj, M, L)
       updatePayoffExplicitTrans(Yj, useDirichlet, lbValue, M, L)
       Yjm2, Yjm = Yjm, Yjm2
       Yjm, Yj = Yj, Yjm
    end
    return Yjm
end


function RKCStep(s::Int, a::Vector{T}, b::Vector{T}, w0::Real, w1::Real, rij::Vector{T}, A1ilj::Vector{T}, A1ij::Vector{T}, A1iuj::Vector{T}, A2ijl::Vector{T}, A2ij::Vector{T}, A2iju::Vector{T}, F::Vector{T}, Yjm2::Vector{T}, Yjm::Vector{T}, Yj::Vector{T}, useDirichlet::Bool, lbValue::Real, M::Int, L::Int) where T
    mu1b = b[1] * w1
    explicitStep(mu1b, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Yjm, M, L)
    updatePayoffExplicitTrans(Yjm, useDirichlet, lbValue, M, L)
    MY0 = (Yjm - F) / mu1b
    Yjm2 .= F
    for j = 2:s
        muj = 2* b[j] *w0/ (b[j - 1])
        mujb = 2* b[j] *w1/ (b[j - 1])
        gammajb = -a[j - 1] * mujb
        nuj = - 1.0  #b0 = b[1]
        if j > 2
            nuj = -b[j] / (b[j - 2])
        end
       @. Yj = muj * Yjm + nuj * Yjm2 + (1 - nuj - muj) * F + gammajb * MY0  #+ mujb*MYjm
       explicitStep(mujb, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, Yjm, Yj, Yj, M, L)
       updatePayoffExplicitTrans(Yj, useDirichlet, lbValue, M, L)
       Yjm2, Yjm = Yjm, Yjm2
       Yjm, Yj = Yj, Yjm
    end
    return Yjm
end

function makeSparseSystemLU(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
    indicesI = zeros(Int, L * M)
    indicesJ = zeros(Int, L * M)
  @inbounds for j = 1:L, i = 1:M
     index = i + (j - 1) * M
        indicesI[index] = index
        indicesJ[index] = index
    end
    A1d = sparse(indicesI, indicesJ, A1ij, M * L, M * L)
    A2d = sparse(indicesI, indicesJ, A2ij, M * L, M * L)
    indicesI = zeros(Int, L * M)
    indicesJ = zeros(Int, L * M)
    @inbounds for j = 1:L
      @inbounds @simd  for i = 2:M
        index = i + (j - 1) * M
             indicesI[index] = index
            indicesJ[index] = index - 1
        end
        indicesI[1 + (j - 1) * M] = 1 + (j - 1) * M
        indicesJ[1 + (j - 1) * M] = 1 + (j - 1) * M
    end
    A1dl = sparse(indicesI, indicesJ, A1ilj, M * L, M * L)
    indicesI = zeros(Int, L * M)
    indicesJ = zeros(Int, L * M)
    @inbounds for j = 1:L
    @inbounds   @simd  for i = 1:M - 1
        index = i + (j - 1) * M
             indicesI[index] = index
            indicesJ[index] = index + 1
        end
        indicesI[M + (j - 1) * M] = 1 + (j - 1) * M
        indicesJ[M + (j - 1) * M] = 1 + (j - 1) * M
    end
    A1du = sparse(indicesI, indicesJ, A1iuj, M * L, M * L)
    indicesI = zeros(Int, L * M)
    indicesJ = zeros(Int, L * M)
    @inbounds for i = 1:M
    @inbounds   @simd  for j = 2:L
        index = i + (j - 1) * M
             indicesI[index] = index
            indicesJ[index] = index - M
        end
        indicesI[i] = i
        indicesJ[i] = i
    end
    A2dl = sparse(indicesI, indicesJ, A2ijl, M * L, M * L)
    indicesI = zeros(Int, L * M)
    indicesJ = zeros(Int, L * M)
  @inbounds   for i = 1:M
    @inbounds   @simd  for j = 1:L - 1
        index = i + (j - 1) * M
             indicesI[index] = index
            indicesJ[index] = index + M
        end
        indicesI[i + (L - 1) * M] = i + (L - 1) * M
        indicesJ[i + (L - 1) * M] = i + (L - 1) * M
    end
    A2du = sparse(indicesI, indicesJ, A2iju, M * L, M * L)

    indicesI = zeros(Int, L * M)
    indicesJ = zeros(Int, L * M)
  @inbounds   for i = 1:M - 1
    @inbounds   @simd  for j = 1:L - 1
        index = i + (j - 1) * M
           indicesI[index] = index
            indicesJ[index] = index + 1 + M
        end
        indicesI[i + (L - 1) * M] = i + (L - 1) * M
        indicesJ[i + (L - 1) * M] = i + (L - 1) * M
    end
  @inbounds   for j = 1:L
        indicesI[M + (j - 1) * M] = M + (j - 1) * M
        indicesJ[M + (j - 1) * M] = M + (j - 1) * M
    end
    A0uu = sparse(indicesI, indicesJ, rij, M * L, M * L)
    indicesI = zeros(Int, L * M)
    indicesJ = zeros(Int, L * M)
  @inbounds   for i = 2:M
    @inbounds     @simd for j = 2:L
        index = i + (j - 1) * M
           indicesI[index] = index
            indicesJ[index] = index - 1 - M
        end
        indicesI[i] = i
        indicesJ[i] = i
    end
    for j = 1:L
       indicesI[1 + (j - 1) * M] = 1 + (j - 1) * M
        indicesJ[1 + (j - 1) * M] = 1 + (j - 1) * M
    end
    A0ll = sparse(indicesI, indicesJ, rij, M * L, M * L)
    indicesI = zeros(Int, L * M)
    indicesJ = zeros(Int, L * M)
  @inbounds   for i = 2:M
    @inbounds   @simd  for j = 1:L - 1
        index = i + (j - 1) * M
           indicesI[index] = index
            indicesJ[index] = index - 1 + M
        end
        indicesI[i + (L - 1) * M] = i + (L - 1) * M
        indicesJ[i + (L - 1) * M] = i + (L - 1) * M
    end
    for j = 1:L
        indicesI[1 + (j - 1) * M] = 1 + (j - 1) * M
        indicesJ[1 + (j - 1) * M] = 1 + (j - 1) * M
    end
    A0lu = sparse(indicesI, indicesJ, -rij, M * L, M * L)
    indicesI = zeros(Int, L * M)
    indicesJ = zeros(Int, L * M)
  @inbounds   for i = 1:M - 1
    @inbounds     @simd for j = 2:L
        index = i + (j - 1) * M
           indicesI[index] = index
            indicesJ[index] = index + 1 - M
        end
        indicesI[i] = i
        indicesJ[i] = i
    end
  @inbounds   for j = 1:L
        indicesI[M + (j - 1) * M] = M + (j - 1) * M
        indicesJ[M + (j - 1) * M] = M + (j - 1) * M
    end
    A0ul = sparse(indicesI, indicesJ, -rij, M * L, M * L)
return A0ll, A0uu, A0ul, A0lu, A2d, A2dl, A2du, A1d, A1dl, A1du
end
function makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
    A0ll, A0uu, A0ul, A0lu, A2d, A2dl, A2du, A1d, A1dl, A1du = makeSparseSystemLU(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
    A0 = A0ll + A0uu + A0ul + A0lu
    A2 = A2d + A2dl + A2du
    A1 = A1d + A1dl + A1du
    return A0, A1, A2
end

function computeLowerBoundary(isCall::Bool, useDirichlet::Bool, B::Real, Smin::Real, r::Real, q::Real, ti::Real)
    lbValue = 0.0
    if useDirichlet && B == 0
        if !isCall
            lbValue = (K - Smin * exp(-q * ti))
        end
    end
    return lbValue
end

function linearInterpolate!(rij::Vector{T}, rij0::Vector{T}, rij1::Vector{T},t0::T,t1::T,t::T) where {T}
    w = (t-t0)/(t1-t0)
    # @. rij = w*rij1 + (1-w)*rij0
    @inbounds @simd for i = 1:length(rij)
        rij[i]= (1-w)*rij0[i]+w*rij1[i]
    end
end

function chebPoly(s::Int, w0::Real)
	tjm = 1.0
	tj = w0
    if s == 1
        return tj, 1.0, 0.0
    end
	ujm = 1.0
	uj = 2 * w0
	for j = 2:s
		tjp = 2*w0*tj - tjm
		ujp = 2*w0*uj - ujm
		tjm = tj
		tj = tjp
		ujm = uj
		uj = ujp
	end
	return tj, s * ujm, s* ((s+1)*tj - uj) / (w0^2 - 1)
end
function computeRKCStages(dtexplicit,dt,ep)
    # delta = 1 + 4 * (2 + 4 * dt / dtexplicit)
    # s = ceil(Int, (-1 + sqrt(delta)) / 2)
    # if s % 2 == 0
    #     s += 1
    # end
    # s+= Int(floor(s/5))
    #dtexplicit=2/lambdamax. we want beta = (w0+1)*tw0p2/tw0p > lambdamax*dt => dtexplicit*beta < 2dt.
    dtexplicit *= 0.9
    s = 1
    betaFunc = function (s::Int)
    w0 = 1 + ep/s^2
    _,tw0p, tw0p2 = chebPoly(s, w0)
    beta = (w0+1)*tw0p2/tw0p
    return beta - 2*dt/(dtexplicit)
end
while s < 10000 && betaFunc(s) < 0
    s += 1
end
    #s += Int(ceil(s/10))
    return s
end

function price(isCall, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L; damping = "None", method = "RKL", useVLinear = false, sDisc = "Sinh", useExponentialFitting = true, smoothing = "Kreiss", lambdaS = 0.25 , lambdaV = 2.0, rklStages = 0, printGamma=false, lsSolver = "SOR")
    epsilon = 1e-3

    #sDisc "Sinh" Linear", "Exp", "Collocation"
    upwindingThreshold = 1.0
    isConstant = false
    if length(cFunc.x) == 1
        isConstant = true
    end
    dChi = 4 * kappa * theta / (sigma * sigma)
    chiN = 4 * kappa * exp(-kappa * T) / (sigma * sigma * (1 - exp(-kappa * T)))
    ncx2 = NoncentralChisq(dChi, v0 * chiN)
    vmax = quantile(ncx2, 1 - epsilon) * exp(-kappa * T) / chiN
    vmin = quantile(ncx2, epsilon) * exp(-kappa * T) / chiN
    vmin = max(vmin, 1e-3)
    vmin = 0.0
    #println("vmin ",vmin)
    V = collect(range(0.0, stop = vmax, length = L))
    hl = V[2] - V[1]
    JV = ones(L)
    JVm = ones(L)

    if !useVLinear
        vscale = v0 * lambdaV
        u = collect(range(0.0, stop = 1.0, length = L))
        c1 = asinh((vmin - v0) / vscale)
        c2 = asinh((vmax - v0) / vscale)
        V = @. v0 + vscale * sinh((c2 - c1) * u + c1)
        hl = u[2] - u[1]
        JV = @. vscale * (c2 - c1) * cosh((c2 - c1) * u + c1)
        JVm = @. vscale * (c2 - c1) * cosh((c2 - c1) * (u - hl / 2) + c1)
    end
    Xspan = 4 * sqrt(theta * T)
    logK = log(K)
    Xmin = logK - Xspan + (r - q) * T - 0.5 * v0 * T
    if B != 0
        Xmin = log(B)
    end
    Xmax = logK + Xspan + (r - q) * T - 0.5 * v0 * T
    Smin = exp(Xmin)
    Smax = exp(Xmax)
    X = collect(range(Xmin, stop = Xmax, length = M))
    hm = X[2] - X[1]
    Sscale = K * lambdaS
    if sDisc == "Exp"
    S = exp.(X)
    J = exp.(X)
    Jm = @. exp(X - hm / 2)
    elseif sDisc == "Sinh"
        u = collect(range(0.0, stop = 1.0, length = M))
        c1 = asinh((Smin - K) / Sscale)
        c2 = asinh((Smax - K) / Sscale)
        S = @. K + Sscale * sinh((c2 - c1) * u + c1)
        hm = u[2] - u[1]
        J = @. Sscale * (c2 - c1) * cosh((c2 - c1) * u + c1)
        Jm = @. Sscale * (c2 - c1) * cosh((c2 - c1) * (u - hm / 2) + c1)
    elseif sDisc == "Collocation"
        Smin = exp(Xmin)
        Smax = exp(Xmax)
        Xmin = solve(cFunc, Smin)
        Xmax = solve(cFunc, Smax)
        X = collect(range(Xmin, stop = Xmax, length = M))
        hm = X[2] - X[1]
        J  = [evaluateSliceDerivative(cFunc, Sci) for Sci in X]
        Jm = [evaluateSliceDerivative(cFunc, Sci - hm / 2) for Sci in X]
        S = [evaluateSlice(cFunc, Xi) for Xi in X]
    else #if sDisc == "Linear"
        S = collect(range(B, stop = exp(Xmax), length = M))
        X = S
        hm = X[2] - X[1]
        J = ones(M)
        Jm = ones(M)
    end
    #Sc = [solve(cFunc, Si) for Si in S]

    Smin = S[1]
    sign = 1
    if !isCall
        sign = -1
    end

    F0 = zeros(M)
    if isCall
        F0 = @. max(S - K, 0.0)
    else
        F0 = @. max(K - S, 0.0)
    end

    iStrike = searchsortedfirst(S, K)
    if smoothing == "Averaging"
        if K < (S[iStrike] + S[iStrike - 1]) / 2
            iStrike -= 1
        end
        a = (S[iStrike] + S[iStrike + 1]) / 2
        if !isCall
            a = (S[iStrike] + S[iStrike - 1]) / 2   # int a,lnK K-eX dX = K(a-lnK)+ea-K
        end
        value = (a - K) * (a - K) * 0.5
        F0[iStrike] = value / (S[iStrike + 1] - S[iStrike - 1]) * 2
    elseif smoothing == "Kreiss"
        xmk = S[iStrike] - K
        h = (S[iStrike + 1] - S[iStrike - 1]) / 2
        # F0smooth[iStrike] = 0.5*xmk + h/6 + 0.5*xmk*xmk/h*(1-xmk/(3*h))
        F0[iStrike] = 0.5 * xmk + h / 6 + 0.5 * xmk * xmk / h * (1 - xmk / (3 * h))
        if !isCall
            F0[iStrike] -= xmk # C-P = f-K
        end
        iStrike -= 1
        xmk = S[iStrike] - K
        h = (S[iStrike + 1] - S[iStrike - 1]) / 2
        F0[iStrike] = 0.5 * xmk + h / 6 + 0.5 * xmk * xmk / h * (1 + xmk / (3 * h))
        if !isCall
            F0[iStrike] -= xmk # C-P = f-K
        end
    end

    F = zeros(M * L)
    for j = 1:L
        F[1 + (j - 1) * M:j * M] = F0
    end
    useDirichlet = (B != 0)
    ti = T
    dt = T / N
    lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
    updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
    etime = @elapsed begin

        if damping == "EU"
            a = 0.5
            ti -= a * dt
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
            lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
            updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
            F = (I + A0 + A1 + A2) \ F
            ti -= a * dt
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
            lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
            updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
            F = (I + A0 + A1 + A2) \ F
            N -= 1
        elseif damping == "EU2"
                a = 0.25
                ti -= a * dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
                updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
                F = (I + A0 + A1 + A2) \ F
                ti -= a * dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
                updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
                F = (I + A0 + A1 + A2) \ F
                ti -= a * dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
                updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
                F = (I + A0 + A1 + A2) \ F
                ti -= a * dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
                updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
                F = (I + A0 + A1 + A2) \ F
                N -= 1
            elseif damping == "DO"
            a = 0.5
            Y0 = zeros(M * L)
            Y2 = zeros(M * L)
            Y1 = zeros(M * L)
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                #explicit
            explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
            ti -= a * dt
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            solveTrapezoidal2(1, 1, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0, F, Y1, M, L)
            solveTrapezoidal1(1, 1, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1, F, Y2, M, L)
            F = Y2
            lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
            updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew
            explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
            ti -= a * dt
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            solveTrapezoidal2(1, 1, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0, F, Y1, M, L)
            solveTrapezoidal1(1, 1, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1, F, Y2, M, L)
            F = Y2
            lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
            updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
            N -= 1
        elseif damping == "Strang"
            a = 0.5
            Y2 = zeros(M * L)
            Y1 = zeros(M * L)
            Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            #expl/2
            Y0 = copy(F)
            for j = 2:L - 1, i = 2:M - 1
                index = i + (j - 1) * M
                Y0[index] -= rij[index] * (F[i + 1 + (j) * M] - F[i + 1 + (j - 2) * M] + F[i - 1 + (j - 2) * M] - F[i - 1 + (j) * M])
            end
            #imp = imp1 imp2 imp2 imp1
            implicitStep1(a, A1ilj, A1ij, A1iuj, useDirichlet, Y0, Y1, M, L)
            implicitStep2(a, A2ijl, A2ij, A2iju, useDirichlet, Y1, Y2, M, L)
            implicitStep2(a, A2ijl, A2ij, A2iju, useDirichlet, Y2, Y1, M, L)
            implicitStep1(a, A1ilj, A1ij, A1iuj, useDirichlet, Y1, Y0, M, L)
            #expl/2
            F = copy(Y0)
            for j = 2:L - 1, i = 2:M - 1
                index = i + (j - 1) * M
                Y0[index] -= rij[index] * (F[i + 1 + (j) * M] - F[i + 1 + (j - 2) * M] + F[i - 1 + (j - 2) * M] - F[i - 1 + (j) * M])
            end
            F = Y0
            ti -= dt
            N -= 1
        end
        if method == "EU"
            for n = 1:N
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
                updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
                F = (I + A0 + A1 + A2) \ F
            end
        elseif method == "Explicit"
            Y0 = zeros(M * L)
            for n = 1:N
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti - dt)
                updatePayoffExplicitTrans(Y0, useDirichlet, lbValue, M, L)
                F, Y0 = Y0, F
                ti -= dt
            end
        elseif method == "PC"
            Y0 = zeros(M * L)
            Y1 = zeros(M * L)
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            for n = 1:N
                explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
                ti -= dt
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
                updatePayoffExplicitTrans(Y0, useDirichlet, lbValue, M, L)
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                explicitStep(0.5, rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew, Y0, 0.5 * (Y0 + F), Y1, M, L)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti + 0.5 * dt)
               updatePayoffExplicitTrans(Y1, useDirichlet, lbValue, M, L)
                F = copy(Y1)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew
            end
        elseif method == "RK2"
            Y0 = zeros(M * L)
            Y1 = zeros(M * L)
            # Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            # rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            # dtexplicit = dt / maximum(A1ij + A2ij)
            # println("Nmin=",T/dtexplicit)
            for n = 1:N
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                explicitStep(0.5, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Y0, M, L)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti - 0.5 * dt)
                updatePayoffExplicitTrans(Y0, useDirichlet, lbValue, M, L)
                Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S)
                rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                explicitStep(1.0, rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew, Y0, F, Y1, M, L)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti - dt)
                updatePayoffExplicitTrans(Y1, useDirichlet, lbValue, M, L)
                F = copy(Y1)
                ti -= dt
            end
        elseif method == "RKL"
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, 1.0, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            dtexplicit = 1.0 / maximum(A1ij + A2ij)
            delta = 1 + 4 * (2 + 4 * dt / dtexplicit)
            s = ceil(Int, (-1 + sqrt(delta)) / 2)
            if s % 2 == 0
                s += 1
            end
            println("s ",s)
            # s = 10
            w1 = 4 / (s^2 + s - 2)
            a = zeros(s)
            b = zeros(s)
            Y0 = zeros(L * M)
            Y1 = zeros(L * M)
            Y2 = zeros(L * M)
            b[1] = 1.0 / 3
            b[2] = 1.0 / 3
            a[1] = 1.0 - b[1]
            a[2] = 1.0 - b[2]
            for i = 3:s
                b[i] = (i^2 + i - 2.0) / (2 * i * (i + 1.0))
                a[i] = 1.0 - b[i]
            end
            for n = 1:N
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                mu1b = b[1] * w1
                thjm2 = 0.0
                thjm1 = mu1b
                explicitStep(mu1b, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Y1, M, L)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti - dt * thjm1)
                updatePayoffExplicitTrans(Y1, useDirichlet, lbValue, M, L)
                MY0 = (Y1 - F) / mu1b
                Y0 .= F
                for j = 2:s
                    tj = ti - dt * thjm1 #* (j^2 + j - 2) / (s^2 + s - 2)
                    Sc, Jc, Jch, Jct = makeJacobians(tj, cFunc, S)
                    rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                    muj = (2 * j - 1) * b[j] / (j * b[j - 1])
                    mujb = muj * w1
                    gammajb = -a[j - 1] * mujb
                    if j > 2
                        nuj = -(j - 1) * b[j] / (j * b[j - 2])
                    else
                        nuj = - 1.0 * b[2] / (2.0 * b[1]) #b0 = b[1]
                    end
                    @. Y2 = muj * Y1 + nuj * Y0 + (1 - nuj - muj) * F + gammajb * MY0 # + mujb*MYjm
                    explicitStep(mujb, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, Y1, Y2, Y2, M, L)
                    # println(j, " ",time, " ",time1, " ",time2)
                    thj = muj * thjm1 + nuj * thjm2 + mujb * (1 - a[j - 1])
                    thjm2 = thjm1
                    thjm1 = thj
                    lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti - dt * thjm1)
                    updatePayoffExplicitTrans(Y2, useDirichlet, lbValue, M, L)
                    Y0 .= Y1
                    Y1 .= Y2
                end
                ti -= dt
                F .= Y2
            end
        elseif method == "RKL2I"
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            rij0, A1ilj0, A1ij0, A1iuj0, A2ijl0, A2ij0, A2iju0 = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            rij = copy(rij0); A1ilj = copy(A1ilj0); A1ij= copy(A1ij0); A1iuj = copy(A1iuj0); A2ijl=copy(A2ijl0); A2ij = copy(A2ij0); A2iju = copy(A2iju0)
            Sc, Jc, Jch, Jct = makeJacobians(ti-dt, cFunc, S)
            rij1, A1ilj1, A1ij1, A1iuj1, A2ijl1, A2ij1, A2iju1 = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            dtexplicit = dt /maximum(A1ij0 + A2ij0)
            if sDisc == "Linear" || (sDisc == "Sinh" && lambdaS >= 1)
                dtexplicit /= 2 #lambdaS
            else
                dtexplicit *= 0.5 #don't be too close to boundary
            end

            delta = 1 + 4 * (2 + 4 * dt / dtexplicit)
            s = ceil(Int, (-1 + sqrt(delta)) / 2)
            if s % 2 == 0
                s += 1
            end
            if rklStages > 0
                s = rklStages
            end
            println("s ",s," ",1.0/dtexplicit," ",T/dtexplicit)
            # s = 10
            w1 = 4 / (s^2 + s - 2)
            a = zeros(s)
            b = zeros(s)
            Y0 = zeros(L * M)
            Y1 = zeros(L * M)
            Y2 = zeros(L * M)
            b[1] = 1.0 / 3
            b[2] = 1.0 / 3
            a[1] = 1.0 - b[1]
            a[2] = 1.0 - b[2]
            for i = 3:s
                b[i] = (i^2 + i - 2.0) / (2 * i * (i + 1.0))
                a[i] = 1.0 - b[i]
            end
            for n = 1:N
                tj = ti
                t0 = ti
                t1 = ti-dt
                linearInterpolate!(rij, rij0, rij1,t0,t1,tj)
                linearInterpolate!(A1ilj,A1ilj0,A1ilj1,t0,t1,tj)
                linearInterpolate!(A1ij,A1ij0,A1ij1,t0,t1,tj)
                linearInterpolate!(A1iuj,A1iuj0,A1iuj1,t0,t1,tj)
                linearInterpolate!(A2ijl,A2ijl0,A2ijl1,t0,t1,tj)
                linearInterpolate!(A2ij,A2ij0,A2ij1,t0,t1,tj)
                linearInterpolate!(A2iju,A2iju0,A2iju1,t0,t1,tj)
                mu1b = b[1] * w1
                thjm2 = 0.0
                thjm1 = mu1b
                explicitStep(mu1b, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Y1, M, L)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti - dt * thjm1)
                updatePayoffExplicitTrans(Y1, useDirichlet, lbValue, M, L)
                MY0 = (Y1 - F) / mu1b
                Y0 .= F
                for j = 2:s
                    tj = ti - dt * thjm1 #* (j^2 + j - 2) / (s^2 + s - 2)
                    linearInterpolate!(rij, rij0, rij1,t0,t1,tj)
                    linearInterpolate!(A1ilj,A1ilj0,A1ilj1,t0,t1,tj)
                    linearInterpolate!(A1ij,A1ij0,A1ij1,t0,t1,tj)
                    linearInterpolate!(A1iuj,A1iuj0,A1iuj1,t0,t1,tj)
                    linearInterpolate!(A2ijl,A2ijl0,A2ijl1,t0,t1,tj)
                    linearInterpolate!(A2ij,A2ij0,A2ij1,t0,t1,tj)
                    linearInterpolate!(A2iju,A2iju0,A2iju1,t0,t1,tj)
                    muj = (2 * j - 1) * b[j] / (j * b[j - 1])
                    mujb = muj * w1
                    gammajb = -a[j - 1] * mujb
                    if j > 2
                        nuj = -(j - 1) * b[j] / (j * b[j - 2])
                    else
                        nuj = - 1.0 * b[2] / (2.0 * b[1]) #b0 = b[1]
                    end
                    @. Y2 = muj * Y1 + nuj * Y0 + (1 - nuj - muj) * F + gammajb * MY0 # + mujb*MYjm
                    explicitStep(mujb, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, Y1, Y2, Y2, M, L)
                    # println(j, " ",time, " ",time1, " ",time2)
                    thj = muj * thjm1 + nuj * thjm2 + mujb * (1 - a[j - 1])
                    thjm2 = thjm1
                    thjm1 = thj
                    lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti - dt * thjm1)
                    updatePayoffExplicitTrans(Y2, useDirichlet, lbValue, M, L)
                    Y0 .= Y1
                    Y1 .= Y2
                end
                ti -= dt
                F .= Y2
                rij0, A1ilj0, A1ij0, A1iuj0, A2ijl0, A2ij0, A2iju0  = rij1, A1ilj1, A1ij1, A1iuj1, A2ijl1, A2ij1, A2iju1
                if ti-dt >= -1e-14
                Sc, Jc, Jch, Jct = makeJacobians(ti-dt, cFunc, S)
                rij1, A1ilj1, A1ij1, A1iuj1, A2ijl1, A2ij1, A2iju1 = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                end
            end
        elseif method == "RKL2"
            Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            dtexplicit = dt / max(maximum(A1ij + A2ij))
            if true || sDisc == "Linear" || (sDisc == "Sinh" && lambdaS >= 1)
                dtexplicit /= 2 #lambdaS
            else
                dtexplicit *= 0.9 #don't be too close to boundary
            end
            delta = 1 + 4 * (2 + 4 * dt / dtexplicit)
            s = ceil(Int, (-1 + sqrt(delta)) / 2)
            if s % 2 == 0
                s += 1
            end
            if rklStages > 0
                s = rklStages
            end
            # println(maximum(A1ij + A2ij), " ", 2 * minimum(A1ilj), " ", 2 * minimum(A2ijl), " ", maximum(A1ij + A2ij + A1iuj + A2iju + A1ilj + A2ijl), " DTE ", dtexplicit, " NE ", T / dtexplicit, " s ", s)
            # println("s=",s)
            w1 = 4 / (s^2 + s - 2)
            a = zeros(s)
            b = zeros(s)
            Y = Vector{Array{Float64,1}}(undef, s)
            b[1] = 1.0 / 3
            b[2] = 1.0 / 3
            a[1] = 1.0 - b[1]
            a[2] = 1.0 - b[2]
            for i = 3:s
                b[i] = (i^2 + i - 2.0) / (2 * i * (i + 1.0))
                a[i] = 1.0 - b[i]
            end
            Y0 = zeros(L * M)
            Y1 = zeros(L * M)
            Y2 = zeros(L * M)
            for n = 1:N
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti - dt * 0.5)
              #  RKLStep(s, a, b, w1, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y, useDirichlet, lbValue, M, L)
              F .= RKLStep(s, a, b, w1, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, Y1, Y2, useDirichlet, lbValue, M, L)
              ti -= dt
                if n < N
                    Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S)
                    rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                end
            end
        elseif method == "RKC2"
            Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            dtexplicit = dt / max(maximum(A1ij + A2ij))
            if sDisc == "Linear" || (sDisc == "Sinh" && lambdaS >= 1)
                dtexplicit /= 2 #lambdaS
            else

            end
            ep = 10.0
            s = computeRKCStages(dtexplicit, dt, ep)

            #s= Int(floor(sqrt(1+6*dt/dtexplicit)))  #Foulon rule as dtExplicit=lambda/2
            if rklStages > 0
                s = rklStages
            end
            #println(maximum(A1ij + A2ij), " ", 2 * minimum(A1ilj), " ", 2 * minimum(A2ijl), " ", maximum(A1ij + A2ij + A1iuj + A2iju + A1ilj + A2ijl), " DTE ", dtexplicit, " NE ", T / dtexplicit, " s ", s)
            # println("s=",s)
            w0 = 1 + ep/s^2
            _,tw0p, tw0p2 = chebPoly(s, w0)
            w1 = tw0p/tw0p2
            b = zeros(s)
            for jj = 2:s
                _, tw0p, tw0p2 = chebPoly(jj, w0)
                b[jj] = tw0p2 / tw0p^2
            end
            b[1] = b[2]
            a = zeros(s)
            for jj = 2:s
                tw0, _, _ = chebPoly(jj-1, w0)
                a[jj-1] = (1 - b[jj-1]*tw0)
            end
            Y = Vector{Array{Float64,1}}(undef, s)
            Y0 = zeros(L * M)
            Y1 = zeros(L * M)
            Y2 = zeros(L * M)
            for n = 1:N
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti - dt * 0.5)
              #  RKLStep(s, a, b, w1, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y, useDirichlet, lbValue, M, L)
              F .= RKCStep(s, a, b, w0, w1, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, Y1, Y2, useDirichlet, lbValue, M, L)
              ti -= dt
                if n < N
                    Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S)
                    rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                end
            end
        elseif method == "ADE0"
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            A0llr, A0uur, A0ulr, A0lur, A2dr, A2dlr, A2dur, A1dr, A1dlr, A1dur = makeSparseSystemLU(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
            # eigenvals, phi = Arpack.eigs(Ar,nev=2)
            # println("largest eigenvalue", eigenvals)

            for n = 1:N
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                A0ll, A0uu, A0ul, A0lu, A2d, A2dl, A2du, A1d, A1dl, A1du = makeSparseSystemLU(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
                # Al = A0 + A1 + A2
                # time1 = @elapsed begin
                # Ald = 0.5 * Diagonal(Al)
                # Ard= 0.5*Diagonal(A1r)
                # end
                # time2 = @elapsed begin
                # Aru = triu!(copy(Ar)) - Ard
                # All = tril!(copy(Al)) - Ald
                # end
                Ald = 0.5 * (A1d + A2d)
                Ard = 0.5 * (A1dr + A2dr)
                All = A0lu + A0uu + A1du + A2du + Ald
            Aru = A0ulr + A0llr + A1dlr + A2dlr + Ard
                rhs = I - Aru
                lhs = I + All
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
                updatePayoffExplicitTrans(F, useDirichlet,  lbValue, M, L)
                F1 = lhs \ (rhs * F)
                # println(n," ",time0, " ",time3)
                Arl = A0lur + A0uur + A1dur + A2dur + Ard
                Alu = A0ul + A0ll + A1dl + A2dl + Ald

                rhs = I - Arl
                lhs = I + Alu
                # println(N, " ", ti, " Ar ", maximum(A1ij + A2ij + 4 * (rij) + (A1ilj) + (A1iuj) + (A2ijl) + (A2iju)))
                F2 =  lhs \ (rhs * F)
                F = 0.5 * (F1 + F2)
                A0llr, A0uur, A0ulr, A0lur, A2dr, A2dlr, A2dur, A1dr, A1dlr, A1dur = A0ll, A0uu, A0ul, A0lu, A2d, A2dl, A2du, A1d, A1dl, A1du
            end
        elseif method == "ADE"
            Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            A0ll, A0uu, A0ul, A0lu, A2d, A2dl, A2du, A1d, A1dl, A1du = makeSparseSystemLU(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)

            for n = 1:N
                ti -= dt
                Ad = 0.5 * (A1d + A2d)
                Al = A0lu + A0uu + A1du + A2du + Ad
            Au = A0ul + A0ll + A1dl + A2dl + Ad
                rhs = I - Au
                lhs = I + Al
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
                updatePayoffExplicitTrans(F, useDirichlet,  lbValue, M, L)
                F1 = lhs \ (rhs * F)
                # println(n," ",time0, " ",time3)
                rhs = I - Al
                lhs = I + Au
                # println(N, " ", ti, " Ar ", maximum(A1ij + A2ij + 4 * (rij) + (A1ilj) + (A1iuj) + (A2ijl) + (A2iju)))
                F2 =  lhs \ (rhs * F)
                @. F = 0.5 * (F1 + F2)
                Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                A0ll, A0uu, A0ul, A0lu, A2d, A2dl, A2du, A1d, A1dl, A1du = makeSparseSystemLU(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
            end
        elseif method == "LS"
            a = 1 - sqrt(2) / 2
            F1 = zeros(L * M)
            F2 = zeros(L * M)
            nIter = 0.0
            # Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            # rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            # A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
            for n = 1:N
                ti1 = ti - a * dt
                Sc, Jc, Jch, Jct = makeJacobians(ti1, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)

                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti1)
                updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
				if lsSolver == "GS"
					F1 = copy(F); nIter+=solveGS(1e-16, 5000, 1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F1, useDirichlet, lbValue, M, L)
	                ti1 -= a * dt
	                Sc, Jc, Jch, Jct = makeJacobians(ti1, cFunc, S)
	                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
					lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti1)
	                updatePayoffExplicitTrans(F1, useDirichlet, lbValue, M, L)
					F2 .= F1; nIter+=solveGS(1e-16, 5000, 1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F1, F2, useDirichlet, lbValue, M, L)
                elseif lsSolver == "Jacobi"
                    F1 = copy(F); nIter+=solveJacobi(1e-16, 5000, 1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F1, useDirichlet, lbValue, M, L)
                    ti1 -= a * dt
                    Sc, Jc, Jch, Jct = makeJacobians(ti1, cFunc, S)
                    rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                    lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti1)
                    updatePayoffExplicitTrans(F1, useDirichlet, lbValue, M, L)
                    F2 .= F1; nIter+=solveSOR(1e-16, 5000, 1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F1, F2, useDirichlet, lbValue, M, L)
				elseif lsSolver == "SOR"
					F1 = copy(F); nIter+=solveSOR(1e-16, 5000, 1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F1, useDirichlet, lbValue, M, L)
	                ti1 -= a * dt
	                Sc, Jc, Jch, Jct = makeJacobians(ti1, cFunc, S)
	                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
					lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti1)
	                updatePayoffExplicitTrans(F1, useDirichlet, lbValue, M, L)
					F2 .= F1; nIter+=solveSOR(1e-16, 5000, 1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F1, F2, useDirichlet, lbValue, M, L)
				elseif lsSolver == "ASOR"
					A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
					F1 = copy(F); asor!(F1, I + A0 + A1 + A2,F,1.2,maxiter=5000,tolerance=1e-8)
					ti1 -= a * dt
					Sc, Jc, Jch, Jct = makeJacobians(ti1, cFunc, S)
					rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                    A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
					lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti1)
					updatePayoffExplicitTrans(F1, useDirichlet, lbValue, M, L)
					F2 = copy(F1); F2= asor!(F2, I + A0 + A1 + A2,F1,1.2,maxiter=5000,tolerance=1e-8)
				elseif lsSolver == "BICGSTABL"
					A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
					LU = IncompleteLU.ilu(I + A0 + A1 + A2, τ = 0.1)
					F1.= F; IterativeSolvers.bicgstabl!(F1, I + A0 + A1 + A2,F,1,Pl=LU)
					ti1 -= a * dt
					Sc, Jc, Jch, Jct = makeJacobians(ti1, cFunc, S)
					rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                    A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
					lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti1)
	                updatePayoffExplicitTrans(F1, useDirichlet, lbValue, M, L)
	                F2 .= F1; IterativeSolvers.bicgstabl!(F2, I + A0 + A1 + A2,F1,1,Pl=LU)
				elseif lsSolver == "IRDS"
                    # if !isConstant || n == 1
                #    A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
                #end
					# F1 .= F;  F1,log=IterativeSolvers.idrs!(F1, I + A0 + A1 + A2,F,s=4,log=true)
                    bMatrix = BandedMatrix{Float64}(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
                    F1 .= F;  F1,log=IterativeSolvers.idrs!(F1, bMatrix,F,s=4,log=true)
                    nIter += log.iters
					ti1 -= a * dt
					Sc, Jc, Jch, Jct = makeJacobians(ti1, cFunc, S)
                    #if !isConstant
                    rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                    bMatrix = BandedMatrix{Float64}(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
                    #A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
                    #end
                	lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti1)
					updatePayoffExplicitTrans(F1, useDirichlet, lbValue, M, L)
					F2 = copy(F); F2,log =IterativeSolvers.idrs!(F2, bMatrix,F1,s=4,log=true)
                    nIter += log.iters
				else
					A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
					F1 = (I + A0 + A1 + A2) \ F
					ti1 -= a * dt
					Sc, Jc, Jch, Jct = makeJacobians(ti1, cFunc, S)
					rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                    A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
					lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti1)
	                updatePayoffExplicitTrans(F1, useDirichlet, lbValue, M, L)
					F2 = (I + A0 + A1 + A2) \ F1
				end
				F = (1 + sqrt(2)) * F2 - sqrt(2) * F1
                ti -= dt
            end
            println("Iterations=",nIter/(2*N))
        elseif method == "SLSA"
            a = 0.5
            b = a * (1 - sqrt(2) / 2)
            for n = 1:N
                Y2 = zeros(M * L)
                Y1 = zeros(M * L)
                Y2t = zeros(M * L)
                Y1t = zeros(M * L)
        #imp = imp1 imp2 imp2 imp1
                Sc, Jc, Jch, Jct = makeJacobians(ti - 2 * b * dt, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                implicitStep1(b, A1ilj, A1ij, A1iuj, useDirichlet, F, Y1t, M, L)
                Sc, Jc, Jch, Jct = makeJacobians(ti - 4 * b * dt, cFunc, S)
                rijt, A1iljt, A1ijt, A1iujt, A2ijlt, A2ijt, A2ijut = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                implicitStep1(b, A1iljt, A1ijt, A1iujt, useDirichlet, Y1t, Y1, M, L)
                Y1 = (1 + sqrt(2)) * Y1 - sqrt(2) * Y1t
                implicitStep2(b, A2ijl, A2ij, A2iju, useDirichlet, Y1, Y2t, M, L)
                implicitStep2(b, A2ijlt, A2ijt, A2ijut, useDirichlet, Y2t, Y2, M, L)
                Y2 = (1 + sqrt(2)) * Y2 - sqrt(2) * Y2t

                Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S)
                rije, A1ilje, A1ije, A1iuje, A2ijle, A2ije, A2ijue = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                F = copy(Y2)
                for j = 2:L - 1, i = 2:M - 1
                    index = i + (j - 1) * M
                    Y2[index] -= 2 * rije[index] * (F[i + 1 + (j) * M] - F[i + 1 + (j - 2) * M] + F[i - 1 + (j - 2) * M] - F[i - 1 + (j) * M])
                end

            # Sc, Jc, Jch, Jct = makeJacobians(ti - a*dt-b*dt, cFunc, S)
            # rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold,  dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            # Sc, Jc, Jch, Jct = makeJacobians(ti - a*dt-2*b * dt, cFunc, S)
            # rijt, A1iljt, A1ijt, A1iujt, A2ijlt, A2ijt, A2ijut = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                implicitStep2(b, A2ijl, A2ij, A2iju, useDirichlet, Y2, Y1t, M, L)
                implicitStep2(b, A2ijlt, A2ijt, A2ijut, useDirichlet, Y1t, Y1, M, L)
                Y1 = (1 + sqrt(2)) * Y1 - sqrt(2) * Y1t
                implicitStep1(b, A1ilj, A1ij, A1iuj, useDirichlet, Y1, Y2t, M, L)
                implicitStep1(b, A1iljt, A1ijt, A1iujt, useDirichlet, Y2t, Y2, M, L)
                Y2 = (1 + sqrt(2)) * Y2 - sqrt(2) * Y2t
        #expl/2
                F = Y2
                ti -= dt
            end
        elseif method == "SLS"
            a = 0.5
            b = a * (1 - sqrt(2) / 2)
            for n = 1:N
                Y2 = zeros(M * L)
                Y1 = zeros(M * L)
                Y2t = zeros(M * L)
                Y1t = zeros(M * L)
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
        #expl/2
                Y0 = copy(F)
                for j = 2:L - 1, i = 2:M - 1
                    index = i + (j - 1) * M
                    Y0[index] -= rij[index] * (F[i + 1 + (j) * M] - F[i + 1 + (j - 2) * M] + F[i - 1 + (j - 2) * M] - F[i - 1 + (j) * M])
                end
        #imp = imp1 imp2 imp2 imp1
                Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * b * dt, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                implicitStep1(b, A1ilj, A1ij, A1iuj, useDirichlet, Y0, Y1t, M, L)
                Sc, Jc, Jch, Jct = makeJacobians(ti - b * dt, cFunc, S)
                rijt, A1iljt, A1ijt, A1iujt, A2ijlt, A2ijt, A2ijut = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                implicitStep1(b, A1iljt, A1ijt, A1iujt, useDirichlet, Y1t, Y1, M, L)
                Y1 = (1 + sqrt(2)) * Y1 - sqrt(2) * Y1t
                Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * b * dt, cFunc, S)
                rijt, A1iljt, A1ijt, A1iujt, A2ijlt, A2ijt, A2ijut = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                implicitStep2(b, A2ijl, A2ij, A2iju, useDirichlet, Y1, Y2t, M, L)
                Sc, Jc, Jch, Jct = makeJacobians(ti - b * dt, cFunc, S)
                rijt, A1iljt, A1ijt, A1iujt, A2ijlt, A2ijt, A2ijut = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                implicitStep2(b, A2ijlt, A2ijt, A2ijut, useDirichlet, Y2t, Y2, M, L)
                Y2 = (1 + sqrt(2)) * Y2 - sqrt(2) * Y2t
                Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * b * dt - 0.5 * dt, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold,  dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                Sc, Jc, Jch, Jct = makeJacobians(ti - b * dt - 0.5 * dt, cFunc, S)
                rijt, A1iljt, A1ijt, A1iujt, A2ijlt, A2ijt, A2ijut = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                implicitStep2(b, A2ijl, A2ij, A2iju, useDirichlet, Y2, Y1t, M, L)
                implicitStep2(b, A2ijl, A2ij, A2iju, useDirichlet, Y1t, Y1, M, L)
                Y1 = (1 + sqrt(2)) * Y1 - sqrt(2) * Y1t
                implicitStep1(b, A1ilj, A1ij, A1iuj, useDirichlet, Y1, Y2t, M, L)
                implicitStep1(b, A1ilj, A1ij, A1iuj, useDirichlet, Y2t, Y2, M, L)
                Y2 = (1 + sqrt(2)) * Y2 - sqrt(2) * Y2t
        #expl/2
                Sc, Jc, Jch, Jct = makeJacobians(ti - .5 * dt, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                F = copy(Y2)
                for j = 2:L - 1, i = 2:M - 1
                    index = i + (j - 1) * M
                    Y2[index] -= rij[index] * (F[i + 1 + (j) * M] - F[i + 1 + (j - 2) * M] + F[i - 1 + (j - 2) * M] - F[i - 1 + (j) * M])
                end
                F = Y2
                ti -= dt
            end
        elseif method == "DO"
            a = 0.5
            Y0 = zeros(M * L)
            Y2 = zeros(M * L)
            Y1 = zeros(M * L)
            for n = 1:N
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                #explicit
                explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                solveTrapezoidal2(a, a, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0, F, Y1, M, L)
                solveTrapezoidal1(a, a, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1, F, Y2, M, L)
                F = copy(Y2)
            end

        elseif method == "PR"
            a = 0.5
            Y0 = zeros(M * L)
            Y2 = zeros(M * L)
            Y1 = zeros(M * L)
            for n = 1:N
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                explicitStep2(a, A2ijl, A2ij, A2iju, F, F, Y0, M, L)
                for j = 2:L - 1, i = 2:M - 1
                    index = i + (j - 1) * M
                    Y0[index] -= rij[index] * (F[i + 1 + (j) * M] - F[i + 1 + (j - 2) * M] + F[i - 1 + (j - 2) * M] - F[i - 1 + (j) * M])
                end
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti - dt)
                updatePayoffExplicitTrans(Y0, useDirichlet, lbValue, M, L)
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                implicitStep1(a, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y0, Y1, M, L)
                explicitStep1(a,  A1ilj, A1ij, A1iuj, Y1, Y1, Y2, M, L)
                for j = 2:L - 1, i = 2:M - 1
                    index = i + (j - 1) * M
                    Y2[index] -=  rij[index] * (Y1[i + 1 + (j) * M] - Y1[i + 1 + (j - 2) * M] + Y1[i - 1 + (j - 2) * M] - Y1[i - 1 + (j) * M])
                end
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
                updatePayoffExplicitTrans(Y2, useDirichlet, lbValue, M, L)
                implicitStep2(a, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y2, F, M, L)
            end
        elseif method == "CS"
            a = 0.5
            Y0 = zeros(M * L)
            Y0t = zeros(M * L)
            Y2 = zeros(M * L)
            Y1 = zeros(M * L)
            Y1t = zeros(M * L)
            Y2t = zeros(M * L)
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            for n = 1:N
                #explicit
                explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                solveTrapezoidal2(a, a, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0, F, Y1, M, L)
                solveTrapezoidal1(a, a, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1, F, Y2, M, L)
                # Y0t = Y0 - .5*(A0*Y2+A0u*F)
                Y0t .= Y0
                for j = 2:L - 1, i = 2:M - 1
                    index = i + (j - 1) * M
                    Y0t[index] += 0.5 * rij[index] * (F[i + 1 + (j) * M] - F[i + 1 + (j - 2) * M] + F[i - 1 + (j - 2) * M] - F[i - 1 + (j) * M])
                    Y0t[index] -= 0.5 * rijNew[index] * (Y2[i + 1 + (j) * M] - Y2[i + 1 + (j - 2) * M] + Y2[i - 1 + (j - 2) * M] - Y2[i - 1 + (j) * M])
                end
                solveTrapezoidal2(a, a, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0t, F, Y1t, M, L)
                solveTrapezoidal1(a, a, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1t, F, Y2t, M, L)
                F .= Y2t
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew
            end
        elseif method == "CS1"
            a = 0.5
            Y0 = zeros(M * L)
            Y2 = zeros(M * L)
            Y1 = zeros(M * L)
            Y1t = zeros(M * L)
            Y2t = zeros(M * L)
            for n = 1:N
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                #explicit
                explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                solveTrapezoidal1(a, a, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y0, F, Y1, M, L)
                solveTrapezoidal2(a, a, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y1, F, Y2, M, L)
                # Y0t = Y0 - .5*(A0*Y2+A0u*F)
                Y0t = copy(Y0)
                for j = 2:L - 1, i = 2:M - 1
                    index = i + (j - 1) * M
                    Y0t[index] += 0.5 * rij[index] * (F[i + 1 + (j) * M] - F[i + 1 + (j - 2) * M] + F[i - 1 + (j - 2) * M] - F[i - 1 + (j) * M])
                    Y0t[index] -= 0.5 * rijNew[index] * (Y2[i + 1 + (j) * M] - Y2[i + 1 + (j - 2) * M] + Y2[i - 1 + (j - 2) * M] - Y2[i - 1 + (j) * M])
                end
                solveTrapezoidal1(a, a, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y0t, F, Y1t, M, L)
                solveTrapezoidal2(a, a, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y1t, F, Y2t, M, L)
                F = Y2t
            end
        elseif method == "MCS1"
            a = 0.3334
            Y0 = zeros(M * L)
            Y0t = zeros(M * L)
            Y2 = zeros(M * L)
            Y1 = zeros(M * L)
            Y1t = zeros(M * L)
            Y2t = zeros(M * L)
            Y0h = zeros(M * L)
            for n = 1:N
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                #explicit
                explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                solveTrapezoidal1(a, a, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y0, F, Y1, M, L)
                solveTrapezoidal2(a, a, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y1, F, Y2, M, L)
                # Y0t = Y0 - .5*(A0*Y2+A0u*F)
                Y0t .= Y0
                for j = 2:L - 1, i = 2:M - 1
                    index = i + (j - 1) * M
                    Y0t[index] += a * rij[index] * (F[i + 1 + (j) * M] - F[i + 1 + (j - 2) * M] + F[i - 1 + (j - 2) * M] - F[i - 1 + (j) * M])
                    Y0t[index] -= a * rijNew[index] * (Y2[i + 1 + (j) * M] - Y2[i + 1 + (j - 2) * M] + Y2[i - 1 + (j - 2) * M] - Y2[i - 1 + (j) * M])
                end
                explicitStep(a - 0.5, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0t, Y0h, M, L)
                explicitStep(0.5 - a, rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew, Y2, Y0h, Y0h, M, L)
                solveTrapezoidal1(a, a, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y0h, F, Y1t, M, L)
                solveTrapezoidal2(a, a, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y1t, F, Y2t, M, L)
                F = Y2t
            end
        elseif method == "MCS"
            a = 0.3334
            Y0 = zeros(M * L)
            Y0t = zeros(M * L)
            Y2 = zeros(M * L)
            Y1 = zeros(M * L)
            Y1t = zeros(M * L)
            Y2t = zeros(M * L)
            Y0h = zeros(M * L)
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
          for n = 1:N
                #explicit
                explicitStep(1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Y0, M, L)
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                solveTrapezoidal2(a, a, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0, F, Y1, M, L)
                solveTrapezoidal1(a, a, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1, F, Y2, M, L)
                # Y0t = Y0 - .5*(A0*Y2+A0u*F)
                Y0t .= Y0
                for j = 2:L - 1, i = 2:M - 1
                    index = i + (j - 1) * M
                    Y0t[index] += a * rij[index] * (F[i + 1 + (j) * M] - F[i + 1 + (j - 2) * M] + F[i - 1 + (j - 2) * M] - F[i - 1 + (j) * M])
                    Y0t[index] -= a * rijNew[index] * (Y2[i + 1 + (j) * M] - Y2[i + 1 + (j - 2) * M] + Y2[i - 1 + (j - 2) * M] - Y2[i - 1 + (j) * M])
                end
                explicitStep(a - 0.5, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0t, Y0h, M, L)
                explicitStep(0.5 - a, rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew, Y2, Y0h, Y0h, M, L)
                solveTrapezoidal2(a, a, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0h, F, Y1t, M, L)
                solveTrapezoidal1(a, a, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1t, F, Y2t, M, L)
                F = Y2t
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew
            end
        elseif method == "CSBDF2"
            a1 = 4.0 / 3.0
            a2 = -1.0 / 3.0
            #b=0
            b0 = 2.0 / 3.0
            bh1 = 4.0 / 3.0
            bh2 = -2.0 / 3.0
            bf1 = 4.0/3.0 - 2.0 / 3.0
                #bf2= 0
            Fprev = copy(F)
            Y0 = zeros(M * L)
            Y2 = zeros(M * L)
            Y1 = zeros(M * L)
            #first step= douglas with theta=1
            Y2t = zeros(M * L)
            Y1t = zeros(M * L)
            Y0t = zeros(M * L)
            dth = 0.5*dt
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dth, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
             #DO
            a=0.5
            for n = 1:N
                #CS
                explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
                ti -= dth
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dth, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                solveTrapezoidal2(a, a, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0, F, Y1, M, L)
                solveTrapezoidal1(a, a, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1, F, Y2, M, L)
                Y0t .= Y0
                for j = 2:L - 1, i = 2:M - 1
                    index = i + (j - 1) * M
                    Y0t[index] += 0.5 * rij[index] * (F[i + 1 + (j) * M] - F[i + 1 + (j - 2) * M] + F[i - 1 + (j - 2) * M] - F[i - 1 + (j) * M])
                    Y0t[index] -= 0.5 * rijNew[index] * (Y2[i + 1 + (j) * M] - Y2[i + 1 + (j - 2) * M] + Y2[i - 1 + (j - 2) * M] - Y2[i - 1 + (j) * M])
                end
                solveTrapezoidal2(a, a, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0t, F, Y1t, M, L)
                solveTrapezoidal1(a, a, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1t, F, Y2t, M, L)
                Fprev .= F
                F .= Y2t
                #BDF2
                rijPrev, A1iljPrev, A1ijPrev, A1iujPrev, A2ijlPrev, A2ijPrev, A2ijuPrev =  rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew
                #explicit
                Y0 = a1 .* F + a2 .* Fprev
                explicitStep2(bf1, A2ijl, A2ij, A2iju, F, Y0, Y0, M, L)
                explicitStep1(bf1, A1ilj, A1ij, A1iuj, F, Y0, Y0, M, L)
                for j = 2:L - 1, i = 2:M - 1
                    index = i + (j - 1) * M
                    Y0[index] -= bh1 * rij[index] * (F[i + 1 + (j) * M] - F[i + 1 + (j - 2) * M] + F[i - 1 + (j - 2) * M] - F[i - 1 + (j) * M])
                    Y0[index] -= bh2 * rijPrev[index] * (Fprev[i + 1 + (j) * M] - Fprev[i + 1 + (j - 2) * M] + Fprev[i - 1 + (j - 2) * M] - Fprev[i - 1 + (j) * M])
                end
                ti -= dth
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dth, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                solveTrapezoidal2(b0, b0, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0, F, Y1t, M, L)
                solveTrapezoidal1(b0, b0, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1t, F, Y2t, M, L)
                F .= Y2t
                rijPrev, A1iljPrev, A1ijPrev, A1iujPrev, A2ijlPrev, A2ijPrev, A2ijuPrev =  rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew
            end
        elseif method == "SC2B"
            a1 = 4.0 / 3.0
            a2 = -1.0 / 3.0
            #b=0
            b0 = 2.0 / 3.0
            bh1 = 4.0 / 3.0
            bh2 = -2.0 / 3.0
            bf1 = 4.0/3.0 - 2.0 / 3.0
                #bf2= 0
            Fprev = copy(F)
            Y0 = zeros(M * L)
            Y2 = zeros(M * L)
            Y1 = zeros(M * L)
            #first step= douglas with theta=1
            Y2t = zeros(M * L)
            Y1t = zeros(M * L)
            Y0t = zeros(M * L)
            Y0h = zeros(M * L)
                 Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                #DO
            explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
            ti -= dt
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            solveTrapezoidal2(0.5, 0.5, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0, F, Y1, M, L)
            solveTrapezoidal1(0.5, 0.5, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1, F, Y2, M, L)
            F = copy(Y2)
            # a=0.3334
            # explicitStep(1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Y0, M, L)
            # ti -= dt
            # Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            # rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            # solveTrapezoidal2(a, a, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0, F, Y1, M, L)
            # solveTrapezoidal1(a, a, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1, F, Y2, M, L)
            # # Y0t = Y0 - .5*(A0*Y2+A0u*F)
            # Y0t .= Y0
            # for j = 2:L - 1, i = 2:M - 1
            #     index = i + (j - 1) * M
            #     Y0t[index] += a * rij[index] * (F[i + 1 + (j) * M] - F[i + 1 + (j - 2) * M] + F[i - 1 + (j - 2) * M] - F[i - 1 + (j) * M])
            #     Y0t[index] -= a * rijNew[index] * (Y2[i + 1 + (j) * M] - Y2[i + 1 + (j - 2) * M] + Y2[i - 1 + (j - 2) * M] - Y2[i - 1 + (j) * M])
            # end
            # explicitStep(a - 0.5, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0t, Y0h, M, L)
            # explicitStep(0.5 - a, rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew, Y2, Y0h, Y0h, M, L)
            # solveTrapezoidal2(a, a, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0h, F, Y1t, M, L)
            # solveTrapezoidal1(a, a, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1t, F, Y2t, M, L)

            # F = copy(Y2t)
            for n = 2:N
                rijPrev, A1iljPrev, A1ijPrev, A1iujPrev, A2ijlPrev, A2ijPrev, A2ijuPrev =  rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew
                #explicit
                Y0 = a1 .* F + a2 .* Fprev
                explicitStep2(bf1, A2ijl, A2ij, A2iju, F, Y0, Y0, M, L)
                explicitStep1(bf1, A1ilj, A1ij, A1iuj, F, Y0, Y0, M, L)
                for j = 2:L - 1, i = 2:M - 1
                    index = i + (j - 1) * M
                    Y0[index] -= bh1 * rij[index] * (F[i + 1 + (j) * M] - F[i + 1 + (j - 2) * M] + F[i - 1 + (j - 2) * M] - F[i - 1 + (j) * M])
                    Y0[index] -= bh2 * rijPrev[index] * (Fprev[i + 1 + (j) * M] - Fprev[i + 1 + (j - 2) * M] + Fprev[i - 1 + (j - 2) * M] - Fprev[i - 1 + (j) * M])
                end
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                solveTrapezoidal2(b0, b0, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0, F, Y1t, M, L)
                solveTrapezoidal1(b0, b0, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1t, F, Y2t, M, L)
                Fprev = F
                F = copy(Y2t)
            end
        elseif method == "SC2A"
            #b=0
            b0 = 3.0/4.0
            bh1 = 1.5
            bh2 = -0.5
            bf1 = 1.5-b0
            bf2 = -0.5+b0
                #bf2= 0
            Fprev = copy(F)
            Y0 = zeros(M * L)
            Y2 = zeros(M * L)
            Y1 = zeros(M * L)
            #first step= douglas with theta=1
            Y2t = zeros(M * L)
            Y1t = zeros(M * L)
            Y0t = zeros(M * L)
            Y0h = zeros(M * L)
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                #DO
            explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
            ti -= dt
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            solveTrapezoidal2(1, 1, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0, F, Y1, M, L)
            solveTrapezoidal1(1, 1, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1, F, Y2, M, L)
            #MCS
            # a=0.3334
            # explicitStep(1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Y0, M, L)
            # ti -= dt
            # Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            # rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            # solveTrapezoidal2(a, a, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0, F, Y1, M, L)
            # solveTrapezoidal1(a, a, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1, F, Y2, M, L)
            # # Y0t = Y0 - .5*(A0*Y2+A0u*F)
            # Y0t .= Y0
            # for j = 2:L - 1, i = 2:M - 1
            #     index = i + (j - 1) * M
            #     Y0t[index] += a * rij[index] * (F[i + 1 + (j) * M] - F[i + 1 + (j - 2) * M] + F[i - 1 + (j - 2) * M] - F[i - 1 + (j) * M])
            #     Y0t[index] -= a * rijNew[index] * (Y2[i + 1 + (j) * M] - Y2[i + 1 + (j - 2) * M] + Y2[i - 1 + (j - 2) * M] - Y2[i - 1 + (j) * M])
            # end
            # explicitStep(a - 0.5, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0t, Y0h, M, L)
            # explicitStep(0.5 - a, rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew, Y2, Y0h, Y0h, M, L)
            # solveTrapezoidal2(a, a, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0h, F, Y1t, M, L)
            # solveTrapezoidal1(a, a, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1t, F, Y2t, M, L)

            F = copy(Y2)
            for n = 2:N
                rijPrev, A1iljPrev, A1ijPrev, A1iujPrev, A2ijlPrev, A2ijPrev, A2ijuPrev =  rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew
                #explicit
                Y0 = copy(F)
                explicitStep2(bf1, A2ijl, A2ij, A2iju, F, Y0, Y0, M, L)
                explicitStep1(bf1, A1ilj, A1ij, A1iuj, F, Y0, Y0, M, L)
                explicitStep2(bf2, A2ijlPrev, A2ijPrev, A2ijuPrev, Fprev, Y0, Y0, M, L)
                explicitStep1(bf2, A1iljPrev, A1ijPrev, A1iujPrev, Fprev, Y0, Y0, M, L)
                for j = 2:L - 1, i = 2:M - 1
                    index = i + (j - 1) * M
                    Y0[index] -= bh1 * rij[index] * (F[i + 1 + (j) * M] - F[i + 1 + (j - 2) * M] + F[i - 1 + (j - 2) * M] - F[i - 1 + (j) * M])
                    Y0[index] -= bh2 * rijPrev[index] * (Fprev[i + 1 + (j) * M] - Fprev[i + 1 + (j - 2) * M] + Fprev[i - 1 + (j - 2) * M] - Fprev[i - 1 + (j) * M])
                end
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                solveTrapezoidal2(b0, b0, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0, F, Y1t, M, L)
                solveTrapezoidal1(b0, b0, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1t, F, Y2t, M, L)
                Fprev = F
                F = copy(Y2t)
            end
        elseif method == "HV"
            a =  0.5 + sqrt(3) / 6 #1.0+sqrt(2)/2
            Y0 = zeros(M * L)
            Y2 = zeros(M * L)
            Y1 = zeros(M * L)
            Y0t = zeros(M * L)
            Y0t0 = zeros(M * L)
            Y1t = zeros(M * L)
            Y2t = zeros(M * L)
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
           for n = 1:N
                explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti - dt)
                updatePayoffExplicitTrans(Y0, useDirichlet, lbValue, M, L)
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                solveTrapezoidal2(a, a, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0, F, Y1, M, L)
                solveTrapezoidal1(a, a, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1, F, Y2, M, L)
            # Y0t =F + .5*(A0*Y2+A0u*F)
                explicitStep(0.5, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Y0t0, M, L)
                explicitStep(0.5, rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew, Y2, Y0t0, Y0t, M, L)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
                updatePayoffExplicitTrans(Y0t, useDirichlet, lbValue, M, L)
                solveTrapezoidal2(a, a, A2ijlNew, A2ijNew, A2ijuNew, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0t, Y2, Y1t, M, L)
                solveTrapezoidal1(a, a, A1iljNew, A1ijNew, A1iujNew, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1t, Y2, Y2t, M, L)
                F = Y2t
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew
            end
        elseif method == "HV1"
            a = 0.5 + sqrt(3) / 6
            Y0 = zeros(M * L)
            Y2 = zeros(M * L)
            Y1 = zeros(M * L)
            Y0t = zeros(M * L)
            Y0t0 = zeros(M * L)
            Y1t = zeros(M * L)
            Y2t = zeros(M * L)
            for n = 1:N
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S)
                rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                solveTrapezoidal1(a, a, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y0, F, Y1, M, L)
                solveTrapezoidal2(a, a, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y1, F, Y2, M, L)
                explicitStep(0.5, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Y0t0, M, L)
                explicitStep(0.5, rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew, Y2, Y0t0, Y0t, M, L)
                solveTrapezoidal1(a, a, A1iljNew, A1ijNew, A1iujNew, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y0t, Y2, Y1t, M, L)
                solveTrapezoidal2(a, a, A2ijlNew, A2ijNew, A2ijuNew, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y1t, Y2, Y2t, M, L)
                F = Y2t
            end
        end
    end  #elapsed
    Payoff = reshape(F, (M, L))
    spl = Spline2D(S, V, Payoff; s = 0.0)
    l2 = 0.0
    maxerr = 0.0
    for i = 1:length(spotArray)
        spot = spotArray[i]
        price = Dierckx.evaluate(spl, spot, v0)
        err = price-priceArray[i]
        l2 += err^2
        if abs(err) > maxerr && i > length(spotArray)/4 && i < length(spotArray)*3/4
            maxerr = abs(err)
        end
        if length(spotArray)==1
            println(B, " ", method, " ", N, " ", M, " ", L, " ", price, " ",price-priceArray[i]," ", etime)
        end
        #println(spot, " ", method, " ", N, " ", M, " ", L, " ", price, " ",price-priceArray[i]," ", etime)
    end
    l2 = sqrt(l2/length(spotArray))
    if length(spotArray) > 1
        println(B, " ", method, " ", N, " ", M, " ", L, " ", l2," ", etime)
    end
    if printGamma
         priceArray = [Dierckx.evaluate(spl, si, v0) for si in S]
         println("push!(B,",B,")")
         println("push!(spotArray2,",S,")")
         println("push!(priceArray2,",priceArray,")")
         # delta, gamma = computeDeltaGamma(S, priceArray)
         # for (i, si) in enumerate(S)
         #     println(si, " ", method, " ", damping, " ", priceArray[i], " ",delta[i], " ", gamma[i])
         # end
         # plot(S, gamma)
     end
end


A = [0.6266758553145932, 0.8838690008217314 ,0.9511741483703275, 0.9972169412308787 ,1.045230848712316, 1.0932361943842062, 1.1786839882076958, 1.2767419415280061]
B = [0.8329310535215612, 0.5486175716699259, 1.0783076034285555 ,1.1476195823811128 ,1.173600641673776, 1.1472056638621118, 0.918270335988941]
C = [-0.38180731761048253, 3.2009663415588276, 0.8377175268235754, 0.31401193651971954 ,-0.31901463307065175, -1.3834775717464938, -1.9682171790586938]
X = [0.5171192610665245, 0.8894451290344221, 0.972184210805066, 1.013553751690388, 1.05492329257571, 1.0962928334610318, 1.179031915231676, 1.3445100787729636]
leftSlope = 0.8329310535215612
rightSlope = 0.2668764075068484
kappa = 0.35
theta = 0.321
sigma = 1.388
rho = -0.63
r = 0.0
q = 0.0
v0 = 0.133
T = 0.4986301369863014
cFunc = CollocationFunction(A, B, C, X, leftSlope, rightSlope, T)
M = 128
L = 128
N = 32
spotArray = [1.0]
K = 1.0
priceArray = [0.07278065] #from cos
# K=0.7
# priceArray = [0.00960629]
# K=1.4
# priceArray = [0.40015225]
# K=0.8
# priceArray = [0.01942598]
# K=1.2
# priceArray = [0.20760077]
t = collect(range(0,length=100,stop=T))
for ti in t
    cFunci = makeSlice(cFunc, ti)
    println(ti," ",solve(cFunci, 0.9))
end

B = 0.0
isCall = false
Ns = [1024, 768,512, 384, 256, 192, 128, 96, 64, 56, 48, 32, 24, 16, 12, 8 ,6,4] #timesteps
Ns = reverse(Ns)
damping = "None"
method = "CS"
N = 16
# M = 512
schemes = ["LS"]
schemes = ["DOLS","CS", "MCS"]#,"HV","RKLM","LS"]
for method in schemes
    if method == "LS" || method == "RKL" || method == "RKLM"
        damping = "None"
    else
        damping = "EU"
    end
#B=0.90
#for B in range(0.5, stop = 0.9, length = 21)
for N in Ns
    price(isCall, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L, damping = damping, method = method)
end
end
end
Base.show(io::IO, x::Union{Float64,Float32}) = Base.Grisu._show(io, x, Base.Grisu.SHORTEST, 0, true, false)
for B in range(0.5, stop = 0.9, length = 21)
    N=40000; price(isCall, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L, damping = "None", method = "RK2", printGamma=true)
    end

    B=[]
    spotArray2=[]
    priceArray2=[]
        push!(B,0.0)
push!(spotArray2,[0.19525278129100765, 0.2297240479317374, 0.2628540364118287, 0.29470043583593486, 0.32531870020067577, 0.3547621449564414, 0.3830820398453534, 0.41032769817704295, 0.4365465626977032, 0.46178428820193573, 0.48608482103124606, 0.5094904755976166, 0.5320420080654074, 0.5537786873198884, 0.5747383633459783, 0.5949575331362578, 0.6144714042430244, 0.6333139560850498, 0.6515179991157933, 0.6691152319561027, 0.6861362965908833, 0.7026108317258537, 0.7185675243972945, 0.7340341599246605, 0.749037670293035, 0.7636041810496794, 0.7777590567963344, 0.7915269453564904, 0.8049318206945355, 0.8179970246615149, 0.8307453076401949, 0.8431988681602054, 0.8553793915522419, 0.8673080877086383, 0.8790057280160606, 0.8904926815246301, 0.9017889504164638, 0.9129142048353853, 0.9238878171384621, 0.9347288956290049, 0.9454563178297747, 0.9560887633543295, 0.966644746433755, 0.977142648155414, 0.9876007484698542, 0.998037258021606, 1.008470349859299, 1.018918191080312, 1.029398974465061, 1.0399309501560077, 1.0505324574365542, 1.0612219566651564, 1.0720180614202666, 1.0829395709120755, 1.094005502717496, 1.1052351258953865, 1.1166479945396797, 1.1282639818288407, 1.1401033146309463, 1.152186608724642, 1.1645349046973084, 1.1771697045829435, 1.1901130093035612, 1.2033873569793008, 1.2170158621739577, 1.231022256144273, 1.2454309281630662, 1.2602669679881717, 1.275556209551123, 1.2913252759416656, 1.3076016257664287, 1.3244136009624752, 1.3417904761489947, 1.35976250960307, 1.378360995948283, 1.3976183206479065, 1.4175680163975715, 1.4382448215156007, 1.459684740432696, 1.4819251063862922, 1.5050046464287632, 1.5289635488626692, 1.5538435332204732, 1.5796879229105802, 1.6065417206562005, 1.634451686858395, 1.6634664210197574, 1.6936364463705198, 1.7250142978444285, 1.7576546135575986, 1.7916142299496263, 1.826952280752635, 1.8637302999605847, 1.9020123289781452, 1.9418650281357206, 1.9833577927647874, 2.026562874035685, 2.0715555047682646, 2.118414030434469, 2.1672200455809634, 2.218058535909363, 2.2710180262614648, 2.326190734767171, 2.3836727334235146, 2.44356411538441, 2.5059691692524284, 2.5709965606760843, 2.638759521568855, 2.7093760472794273, 2.7829691020564677, 2.8596668331657504, 2.939602794032422, 3.0229161767970028, 3.109752054690054, 3.2002616346475676, 3.294602520606935, 3.3929389879420078, 3.4954422695150775, 3.6022908538439258, 3.713670795903103, 3.8297760411006343, 3.9508087629943223, 4.076979715335678, 4.208508599054516, 4.345624444823224, 4.4885660118668715, 4.637582203713615, 4.792932501609322])
push!(priceArray2,[0.6413804697751534, 0.612664704983235, 0.5850742852018089, 0.5585581290201791, 0.5330757764583522, 0.5085925596614621, 0.48508151301422003, 0.46252524981197873, 0.44091689244125515, 0.4202587902002297, 0.4005576740607188, 0.3818166451355511, 0.364026964235509, 0.34716522745251494, 0.3311924507090287, 0.316062742038169, 0.30172671320294786, 0.2881336328244414, 0.2752329183525332, 0.2629749787896162, 0.2513115911554122, 0.24019584514669606, 0.2295820143558661, 0.2194254616984951, 0.2096827102323313, 0.2003118186868873, 0.19127318816300223, 0.18253086740543556, 0.17405430489391133, 0.16582032862345453, 0.1578149674823217, 0.15003463801722217, 0.1424862725735131, 0.13518616569393166, 0.12815760076309662, 0.12142758991084088, 0.11502313154459068, 0.10896795823401126, 0.10327949719450273, 0.0979669272022377, 0.09303034542047793, 0.08846104256079365, 0.08424270883446222, 0.08035325705864557, 0.07676696909326293, 0.07345630496594509, 0.07039364473535069, 0.06755249405890654, 0.06490825965883516, 0.06243861476328359, 0.06012362775167572, 0.05794569865860538, 0.05588940547111498, 0.05394127030548036, 0.052089547755799166, 0.05032401016722961, 0.0486357633003123, 0.04701704794747027, 0.045461098370819714, 0.04396201063272201, 0.042514629532636676, 0.041114451766004784, 0.03975754298824242, 0.03844047269840643, 0.037160242320124666, 0.03591423700397697, 0.034700180272938565, 0.03351609508835234, 0.03236027063221266, 0.031231234112016917, 0.030127726779309395, 0.029048683230636852, 0.027993213055719435, 0.026960584034484057, 0.02595016797852072, 0.024961483945007613, 0.023994161741391504, 0.023047928853568468, 0.022122598049326803, 0.021218055621281446, 0.02033425024656312, 0.0194711824441715, 0.018628894609604697, 0.01780746160356751, 0.017006981871060225, 0.016227569076305664, 0.01546934427228767, 0.014732428697672512, 0.014016937389896773, 0.013322973790257693, 0.012650625165648856, 0.011999958068916775, 0.011371013158132201, 0.01076380017937079, 0.010178294609305123, 0.009614435497508816, 0.009072123652943779, 0.008551220273099026, 0.00805154602172598, 0.0075728805899915475, 0.007114962942804464, 0.0066774926533394695, 0.006260132493439551, 0.005862511220287602, 0.005484224510922266, 0.0051248347516054575, 0.00478387372989983, 0.004460846092094001, 0.004155232335768125, 0.0038664917324052264, 0.0035940653276043105, 0.003337379356824622, 0.003095849468026849, 0.002868885175017158, 0.0026558922483967333, 0.002456272703196329, 0.0022694277404496743, 0.002094760811441119, 0.0019316793623317904, 0.001779595731274034, 0.0016379274617734656, 0.0015060965101680714, 0.0013835046071627203, 0.0012693074489341163, 0.001161176371076999, 0.0010506605510814974, 0.0009111893075395297, 0.0006779494104840894])

#L=256
#M=256
#B=0.9
#B=[]
#spotArray2=[]
#priceArray2=[]
push!(B,0.9)
push!(spotArray2,[0.9, 0.9040054070017137, 0.9079894546979057, 0.9119530295602958, 0.9158970135052898, 0.9198222840902109, 0.9237297147085604, 0.9276201747843521, 0.9314945299655649, 0.9353536423167528, 0.9391983705108596, 0.9430295700202778, 0.946848093307196, 0.9506547900132761, 0.954450507148703, 0.9582360892806492, 0.962012378721195, 0.9657802157147477, 0.9695404386250004, 0.9732938841214723, 0.9770413873656721, 0.9807837821969261, 0.9845219013179112, 0.9882565764799366, 0.991988638668012, 0.9957189182857469, 0.9994482453401191, 1.0031774496261563, 1.0069073609115695, 1.010638809121381, 1.014372624522586, 1.0181096379088936, 1.021850680785581, 1.0255965855545082, 1.0293481856993318, 1.0331063159709606, 1.03687181257329, 1.0406455133492643, 1.0444282579672985, 1.0482208881081116, 1.0520242476520034, 1.0558391828666223, 1.0596665425952652, 1.0635071784457488, 1.0673619449798981, 1.0712316999036893, 1.0751173042580946, 1.0790196226106692, 1.0829395232479209, 1.0868778783685091, 1.0908355642773135, 1.0948134615804168, 1.0988124553810437, 1.1028334354765017, 1.106877296556165, 1.1109449384005485, 1.1150372660815127, 1.1191551901636476, 1.1232996269068787, 1.1274714984703385, 1.131671733117552, 1.1359012654229819, 1.1401610364799735, 1.1444519941101556, 1.1487750930743343, 1.1531312952849329, 1.1575215700200228, 1.1619468941389917, 1.1664082522999015, 1.1709066371785797, 1.1754430496894943, 1.1800184992084624, 1.184634003797243, 1.1892905904300601, 1.19398929522211, 1.1987311636601028, 1.2035172508348881, 1.2083486216762196, 1.2132263511897055, 1.218151524696005, 1.223125238072317, 1.2281485979962183, 1.2332227221919057, 1.2383487396788955, 1.2435277910232363, 1.248761028591291, 1.2540496168061437, 1.2593947324066916, 1.264797564709474, 1.2702593158733029, 1.2757812011667482, 1.2813644492385428, 1.2870103023909627, 1.2927200168562463, 1.2984948630761117, 1.3043361259844386, 1.3102451052931707, 1.31622311578151, 1.322271487588462, 1.3283915665087977, 1.3345847142925014, 1.340852308947766, 1.347195745047608, 1.3536164340401673, 1.3601158045627617, 1.3666953027597657, 1.373356392604387, 1.3801005562244062, 1.3869292942319589, 1.3938441260574312, 1.4008465902875389, 1.4079382450076716, 1.415120668148577, 1.4223954578374562, 1.4297642327535567, 1.437228632488337, 1.444790317910285, 1.4524509715344682, 1.4602122978969048, 1.4680760239338297, 1.4760438993659482, 1.4841176970877574, 1.4922992135620263, 1.5005902692195154, 1.5089927088640347, 1.517508402082921, 1.5261392436630326, 1.534887154012346, 1.5437540795872615, 1.5527419933256958, 1.5618528950860742, 1.5710888120923072, 1.580451799384862, 1.5899439402780149, 1.5995673468234046, 1.6093241602799722, 1.6192165515904033, 1.629246721864175, 1.6394169028673125, 1.649729357518971, 1.6601863803949457, 1.6707902982382277, 1.6815434704767172, 1.692448289748207, 1.7035071824327588, 1.7147226091925858, 1.7260970655195607, 1.7376330822904782, 1.7493332263301884, 1.7612001009827263, 1.7732363466905734, 1.7854446415821683, 1.797827702067803, 1.8103882834440417, 1.8231291805067866, 1.8360532281731354, 1.8491633021121656, 1.862462319384786, 1.8759532390927969, 1.8896390630373092, 1.9035228363866585, 1.91760764835397, 1.9318966328845293, 1.9463929693530968, 1.9610998832713382, 1.9760206470055182, 1.9911585805046164, 2.0065170520390394, 2.0220994789500715, 2.0379093284102554, 2.0539501181948583, 2.070225417464591, 2.0867388475597677, 2.103494082806071, 2.120494851332115, 2.1377449358989655, 2.1552481747418293, 2.1730084624240824, 2.191029750703828, 2.209316049413183, 2.227871427350496, 2.2467000131856585, 2.265805996378776, 2.2851936281123333, 2.304867222237104, 2.3248311562320145, 2.3450898721781464, 2.3656478777471266, 2.3865097472041157, 2.4076801224255884, 2.429163713932196, 2.450965301936867, 2.473089737408437, 2.4955419431510153, 2.5183269148993315, 2.541449722430312, 2.564915510691141, 2.5887295009440257, 2.6128969919279648, 2.6374233610377393, 2.6623140655204214, 2.6875746436896266, 2.7132107161578287, 2.739227987086971, 2.7656322454576765, 2.7924293663573216, 2.8196253122872785, 2.847226134489598, 2.8752379742934537, 2.9036670644816063, 2.932519730677239, 2.9618023927514447, 2.9915215662516728, 3.02168386385148, 3.052295996821882, 3.083364776524646, 3.1148971159278616, 3.1469000311441033, 3.1793806429915556, 3.2123461785784415, 3.2458039729110784, 3.279761470525965, 3.3142262271462255, 3.3492059113628057, 3.384708306340765, 3.4207413115510774, 3.457312944528303, 3.494431342654541, 3.532104764970024, 3.570341594010813, 3.609150337673945, 3.648539631110499, 3.688518238646946, 3.729095055735269, 3.7702791109322495, 3.8120795679083597, 3.854505727486737, 3.897567029712661, 3.9412730559540123, 3.9856335310331907, 4.03065832539092, 4.076357457282484, 4.122741095006848, 4.169819559169143, 4.217603324977068, 4.26610302457167, 4.315329449393068, 4.365293552581581, 4.41600645141488, 4.467479429781628, 4.5197239406922245, 4.572751608827133, 4.626574233123456, 4.681203789400247, 4.736652433023213, 4.792932501609323])
push!(priceArray2,[-1.610920761215018e-20, 0.0004274296227348393, 0.0007941223645368175, 0.0011051213938668755, 0.0013654809501453126, 0.0015801720934377694, 0.0017540070078237342, 0.0018915801035099, 0.0019972241651238227, 0.002074979804623437, 0.0021285765068837195, 0.0021614236054225385, 0.0021766095932926562, 0.0021769082581565777, 0.002164790229734525, 0.0021424386411084766, 0.002111767731295033, 0.0020744433527062943, 0.0020319044901976244, 0.001985385044069745, 0.0019359353026724728, 0.0018844424953035099, 0.0018316503380214014, 0.0017781771368805186, 0.0017245323759474932, 0.0016711317328512792, 0.001618310526157869, 0.0015663356449739931, 0.001515416043665356, 0.0014657119055347023, 0.001417342590932086, 0.0013703934953827185, 0.0013249219112475726, 0.0012809620493059429, 0.001238529295706452, 0.0011976238114021398, 0.001158233563002876, 0.0011203368645479807, 0.001083904500546275, 0.0010489014919990928, 0.0010152885591602208, 0.0009830233275666422, 0.000952061322441085, 0.000922356764301412, 0.0008938632297082156, 0.000866534177354646, 0.0008403233673537855, 0.0008151851916548979, 0.0007910749307301159, 0.0007679489492910223, 0.0007457648417668561, 0.0007244815365545229, 0.0007040593665919508, 0.0006844601199005887, 0.000665647044620532, 0.0006475848613650866, 0.0006302397526186499, 0.0006135793411323124, 0.0005975726598210207, 0.0005821901152249251, 0.0005674034462276416, 0.0005531856794146695, 0.0005395110821969767, 0.0005263551146098545, 0.0005136943805187369, 0.0005015065788159912, 0.0004897704550706774, 0.0004784657539926646, 0.0004675731729897313, 0.0004570743170282589, 0.0004469516549523936, 0.00043718847737098836, 0.0004277688580677703, 0.00041867761116905574, 0.0004099002562153694, 0.0004014229827705598, 0.00039323261671219464, 0.0003853165881743593, 0.0003776629011048227, 0.00037026010439171714, 0.0003630972645099401, 0.0003561639396341005, 0.00034945015516260397, 0.00034294638059612347, 0.00033664350771287865, 0.00033053282998263115, 0.0003246060231608302, 0.0003188551270037587, 0.0003132725280448436, 0.00030785094337148296, 0.0003025834053410959, 0.00029746324717484904, 0.0002924840893679995, 0.0002876398268573788, 0.000282924616889333, 0.0002783328675354978, 0.0002738592268088438, 0.000269498572338112, 0.00026524600156448987, 0.0002610968224296879, 0.0002570465445290392, 0.00025309087070667617, 0.0002492256890722257, 0.00024544705781206057, 0.0002417512139398, 0.00023813455825601594, 0.0002345936488517853, 0.00023112519494805276, 0.0002277260510510807, 0.0002243932114046762, 0.00022112380472038673, 0.0002179150891674709, 0.00021476444760515088, 0.00021166938304040364, 0.00020862751429536662, 0.00020563657186926443, 0.00020269439398061295, 0.0001997989227763146, 0.00019694820069509627, 0.00019414036697357233, 0.00019137365428402404, 0.00018864638549376435, 0.00018595697053670933, 0.00018330390338850815, 0.0001806857591372705, 0.00017810119114259384, 0.0001755489282762262, 0.00017302777223830203, 0.00017053659494365475, 0.0001680743359732545, 0.00016564000008632775, 0.00016323265478920102, 0.00016085142795736035, 0.00015849550550765034, 0.00015616412911792653, 0.0001538565939918569, 0.00015157224666690597, 0.00014931048286385696, 0.0001470707453765253, 0.00014485252200058474, 0.00014265534350068078, 0.0001404787816152372, 0.00013832244709858086, 0.0001361859878002221, 0.00013406908678134888, 0.00013197146046882674, 0.000129892856847275, 0.0001278330536901206, 0.00012579185683092418, 0.00012376909847673293, 0.0001217646355656348, 0.00011977834817097385, 0.00011781013795454136, 0.00011585992667017936, 0.00011392765471730366, 0.00011201327974084292, 0.00011011677527048874, 0.0001082381293892656, 0.0001063773434210899, 0.00010453443063078963, 0.00010270941493800826, 0.00010090232965593473, 9.911321627237785e-5, 9.734212329043589e-5, 9.558910513877111e-5, 9.38542211515718e-5, 9.213753461125529e-5, 9.043911184536795e-5, 8.875902137126803e-5, 8.709733308492123e-5, 8.5454117491946e-5, 8.382944497994826e-5, 8.222338513159326e-5, 8.063600607799766e-5, 7.906737389201906e-5, 7.751755202095871e-5, 7.598660075810051e-5, 7.447457675241767e-5, 7.298153255568777e-5, 7.150751620617795e-5, 7.00525708479878e-5, 6.861673438507748e-5, 6.720003916894959e-5, 6.58025117189072e-5, 6.442417247376776e-5, 6.306503557387427e-5, 6.172510867221522e-5, 6.0404392773445624e-5, 5.910288209958717e-5, 5.7820563981207146e-5, 5.655741877292253e-5, 5.531341979219786e-5, 5.408853328061382e-5, 5.288271838713388e-5, 5.1695927173386704e-5, 5.0528104641579135e-5, 4.937918878616501e-5, 4.8249110670424436e-5, 4.7137794528044125e-5, 4.604515788701976e-5, 4.497111170869513e-5, 4.391556052986729e-5, 4.287840259377592e-5, 4.185952996024466e-5, 4.0858828597359584e-5, 3.9876178471737554e-5, 3.891145366125563e-5, 3.796452250571693e-5, 3.703524779245242e-5, 3.612348696057673e-5, 3.5229092308892244e-5, 3.435191120171559e-5, 3.3491786272797666e-5, 3.2648555628194885e-5, 3.182205304842179e-5, 3.101210818979063e-5, 3.021854678453606e-5, 2.9441190839082948e-5, 2.8679858829611138e-5, 2.793436589386758e-5, 2.7204524017947123e-5, 2.6490142216477376e-5, 2.5791026704272675e-5, 2.5106981057022668e-5, 2.443780635790483e-5, 2.3783301326077736e-5, 2.3143262421722934e-5, 2.2517483920500556e-5, 2.1905757947747172e-5, 2.1307874459134773e-5, 2.072362114932705e-5, 2.015278326263613e-5, 1.959514326859646e-5, 1.9050480348823816e-5, 1.851856961643446e-5, 1.799918095059491e-5, 1.7492077267952017e-5, 1.6997011955214345e-5, 1.6513725027705075e-5, 1.6041937312566302e-5, 1.558134150281544e-5, 1.5131588148652607e-5, 1.4692263303690247e-5, 1.4262852244183474e-5, 1.3842679923835342e-5, 1.3430813214496234e-5, 1.3025902927320543e-5, 1.2625937763431685e-5, 1.2227884358565343e-5, 1.1827208216892081e-5, 1.1417319115010734e-5, 1.0989056748212604e-5, 1.0530395087984341e-5, 1.0026542150854541e-5, 9.460511679043205e-6, 8.814085203163183e-6])
B=[]
spotArray2=[]
priceArray2=[]

push!(B,0.5)
push!(spotArray2,[0.5, 0.5210198932030602, 0.5413390527524862, 0.5609872049745026, 0.579993094531567, 0.5983845264749559, 0.6161884069227277, 0.6334307824225743, 0.6501368780571453, 0.6663311343475931, 0.68203724300933, 0.6972781816123026, 0.7120762471964932, 0.7264530888918269, 0.7404297395902046, 0.7540266467159983, 0.767263702140025, 0.7801602712807625, 0.7927352214353807, 0.8050069493820358, 0.8169934082938107, 0.8287121340036725, 0.840180270658875, 0.8514145958023367, 0.8624315449176863, 0.8732472354738875, 0.8838774905046166, 0.8943378617568898, 0.904643652442807, 0.9148099396276962, 0.9248515962874112, 0.934783313067054, 0.9446196197729506, 0.9543749066293263, 0.9640634453307759, 0.9736994099213281, 0.9832968975306506, 0.9928699489977321, 1.0024325694122114, 1.0119987486034074, 1.0215824816070243, 1.0311977891394708, 1.0408587381097534, 1.0505794621989442, 1.0603741825373374, 1.0702572285095393, 1.080243058717932, 1.0903462821351788, 1.100581679476717, 1.1109642248245049, 1.121509107533658, 1.1322317544540255, 1.14314785249921, 1.154273371596059, 1.1656245880481904, 1.177218108347742, 1.1890708934701755, 1.2012002836876792, 1.2136240239374705, 1.226360289782111, 1.239427713999815, 1.25284541384365, 1.2666330190095079, 1.2808107003537679, 1.2953991994026597, 1.3104198586964981, 1.325894653013187, 1.341846221516664, 1.3582979008773255, 1.3752737594128803, 1.3927986322995842, 1.4108981579053637, 1.4295988152979848, 1.4489279629831442, 1.4689138789291474, 1.4895858019367365, 1.5109739744145858, 1.5331096866230476, 1.5560253224508704, 1.579754406791868, 1.60433165459084, 1.6297930216305052, 1.6561757571337408, 1.6835184582580904, 1.7118611265622565, 1.7412452265271898, 1.771713746217395, 1.8033112601711885, 1.8360839946119238, 1.8700798950755808, 1.905348696553661, 1.9419419962539977, 1.9799133290859414, 2.0193182459803367, 2.0602143951588827, 2.102661606471771, 2.146721978926969, 2.1924599715392343, 2.2399424976317257, 2.28923902272821, 2.340421666179056, 2.393565306669703, 2.448747691765946, 2.5060495516563193, 2.5655547172579545, 2.6273502428587223, 2.691526533475071, 2.7581774771118717, 2.8274005821177806, 2.899297119837051, 2.9739722727665043, 3.051535288434388, 3.132099639226264, 3.2157831883917414, 3.302708362474893, 3.393002330420657, 3.4867971896192227, 3.5842301591605796, 3.6854437805819704, 3.7905861264019154, 3.8998110167459177, 4.0132782443807145, 4.131153808486355, 4.253610157508058, 4.38082644144318, 4.512988773932292, 4.650290504537949, 4.79293250160932])
push!(priceArray2,[-2.5237215955836445e-18, 0.1197657794485899, 0.16434071942482092, 0.18651861492670854, 0.19722099058589596, 0.2015902758216534, 0.20220485754904918, 0.20048135118853258, 0.1972553169056382, 0.19304637029359892, 0.18819047036338407, 0.18291098974408326, 0.1773591127442953, 0.1716379852139207, 0.16581781640233276, 0.15994576745368735, 0.15405278760342694, 0.14815867496534016, 0.14227612909681792, 0.13641422398473843, 0.13058146649895405, 0.12478838993883183, 0.11904948808161414, 0.11338426465465307, 0.10781727160635739, 0.10237718893114697, 0.09709516432729288, 0.09200270966479436, 0.08712928333087108, 0.08250052658325535, 0.07813619500863411, 0.07404888242159534, 0.0702434418134196, 0.06671717102159079, 0.06346070671232906, 0.0604594447466172, 0.05769524356170173, 0.05514802854553142, 0.0527972373766516, 0.05062287179274576, 0.048606172986496, 0.04672998246444654, 0.04497886079055548, 0.04333907113178125, 0.04179847625874443, 0.0403464023938958, 0.03897347229027569, 0.03767145976031507, 0.0364331481053908, 0.03525221065552313, 0.034123080154184145, 0.03304085819571974, 0.03200122916343647, 0.03100038599650254, 0.030034966398509468, 0.02910199809342365, 0.028198851814860346, 0.02732320376441039, 0.026472995143516437, 0.025646404360572927, 0.02484182065546531, 0.02405782100518875, 0.023293149880738228, 0.02254670159195426, 0.021817505054217198, 0.021104710776171418, 0.02040757970742016, 0.019725473419845426, 0.019057845091192762, 0.018404213225853866, 0.017764182936075987, 0.017137426286456366, 0.016523675456309037, 0.015922716309432552, 0.015334382332344023, 0.014758548910296316, 0.014195127915456266, 0.01364406258432556, 0.013105322662876298, 0.012578899799282454, 0.012064803167997363, 0.011563055320274452, 0.011073688282678296, 0.010596739968964014, 0.010132251005129272, 0.009680262013680798, 0.009240811187188338, 0.008813931735776452, 0.008399648944766401, 0.007997977256126016, 0.007608918090971192, 0.007232458368394837, 0.006868569199203493, 0.006517204768241208, 0.0061783014265781285, 0.005851776987850321, 0.005537530219727022, 0.005235440549345986, 0.004945368080455716, 0.004667154114266629, 0.0044006222749926735, 0.00414557981606022, 0.003901818163373051, 0.00366911260070169, 0.0034472226782417932, 0.0032358938597192213, 0.003034858951320448, 0.002843839429167368, 0.002662546750585635, 0.0024906836603488317, 0.0023279455381819804, 0.002174021928287885, 0.0020285984094903496, 0.0018913585664369953, 0.0017619850875210793, 0.0016401596629978625, 0.0015255636265610589, 0.0014178792538545816, 0.0013167901116275743, 0.0012219805264573754, 0.0011331334576621696, 0.0010499216799489516, 0.0009719598511213817, 0.0008985650491110353, 0.0008278174136295088, 0.0007538137571378763, 0.0006610421521548383, 0.0005174193958705766])
push!(B,0.52)
push!(spotArray2,[0.52, 0.5402031878639764, 0.5597437409989813, 0.5786498202116771, 0.5969486719386295, 0.6146666675124242, 0.6318293411666291, 0.6484614268343756, 0.6645868937935875, 0.6802289812102311, 0.6954102316293665, 0.7101525234622643, 0.7244771025164101, 0.7384046126138326, 0.7519551253418832, 0.7651481689793409, 0.7780027566395292, 0.7905374136710052, 0.8027702043553071, 0.8147187579402367, 0.8264002940461947, 0.8378316474821819, 0.8490292925072317, 0.8600093665722364, 0.8707876935763839, 0.8813798066717201, 0.8918009706487023, 0.9020662039350038, 0.9121903002392735, 0.9221878498710431, 0.9320732607675051, 0.9418607792574665, 0.951564510592401, 0.9611984392741872, 0.97077644920883, 0.9803123437152076, 0.9898198654176812, 0.9993127160512353, 1.0088045762076894, 1.0183091250514398, 1.0278400600331448, 1.0374111166297628, 1.0470360881393903, 1.0567288455594321, 1.0665033575767437, 1.0763737106985614, 1.0863541295532282, 1.0964589973899705, 1.106702876807275, 1.1171005307397293, 1.12766694373358, 1.138417343541664, 1.1493672230688365, 1.1605323626995223, 1.1719288530395668, 1.1835731181051623, 1.1954819389922673, 1.2076724780606298, 1.2201623036672684, 1.2329694154850546, 1.2461122704428842, 1.2596098093248234, 1.2734814840665594, 1.2877472857884962, 1.3024277736058956, 1.3175441042575802, 1.3331180625959027, 1.349172092981916, 1.3657293316309935, 1.3828136399555142, 1.4004496389526604, 1.4186627446868922, 1.4374792049182274, 1.4569261369291189, 1.4770315666044405, 1.4978244688209033, 1.5193348092041083, 1.5415935873134112, 1.5646328813168435, 1.588485894220461, 1.6131870017187508, 1.6387718017350579, 1.6652771657234184, 1.6927412918057396, 1.721203759820905, 1.7507055883651303, 1.7812892939057903, 1.8129989520538863, 1.845880261083479, 1.8799806077896088, 1.915349135779628, 1.9520368162963546, 1.9900965216751232, 2.0295831015405748, 2.070553461853033, 2.11306664691834, 2.1571839244793773, 2.2029688740118774, 2.25048747835179, 2.2998082187862314, 2.351002173745083, 2.4041431212354416, 2.4593076451665628, 2.5165752457185278, 2.5760284539136675, 2.637752950555889, 2.7018376897093037, 2.768375026894085, 2.8374608521843485, 2.909194728399814, 2.9836800345904693, 3.061024115020947, 3.1413384338693806, 3.2247387358636574, 3.3113452130865597, 3.401282678190213, 3.494680744269447, 3.591674011653297, 3.6924022618838417, 3.7970106591619466, 3.905649959550175, 4.018476728234425, 4.1356535651573605, 4.257349339348792, 4.383739432290755, 4.515005990667983, 4.6513381888680225, 4.79293250160932])
push!(priceArray2,[5.473362196191924e-19, 0.10427492087848977, 0.14657694324262344, 0.16835909554148454, 0.17939004073332965, 0.18436040280745347, 0.18567291526314325, 0.18466961302514978, 0.18215213791885018, 0.17862398252109077, 0.17441350657059826, 0.16974070047033837, 0.1647555645961922, 0.1595612845044423, 0.15422889033439402, 0.1488070014117282, 0.14332869639855053, 0.13781670347391067, 0.13228759753334296, 0.12675534326223553, 0.12123426080616245, 0.11574130746649239, 0.11029749174021415, 0.10492828518995481, 0.09966304582234113, 0.09453363458956877, 0.0895725105175609, 0.08481049147877817, 0.08027493872994727, 0.07598774690962626, 0.071963985347271, 0.06821117153490663, 0.06472926864893297, 0.06151138734650496, 0.058545049200076596, 0.055813774095686636, 0.05329874055951454, 0.050980200641754, 0.048838638259534634, 0.046855535316327836, 0.04501381611308157, 0.04329802337863721, 0.04169434856568511, 0.040190557042623476, 0.03877586652507782, 0.0374407954201036, 0.03617701944801368, 0.03497723238500272, 0.03383501931980668, 0.032744742719675285, 0.031701441830914216, 0.030700745601634456, 0.029738797339764263, 0.028812189838206975, 0.027917909650531224, 0.02705328923893782, 0.02621596997048899, 0.0254038584565994, 0.02461509981122017, 0.023848051041068886, 0.023101257831041718, 0.022373434214994763, 0.02166344475853283, 0.020970288993307672, 0.0202930878888139, 0.019631072103550654, 0.018983571659004994, 0.01835000662639563, 0.01772985974792371, 0.01712270376598462, 0.016528179765144597, 0.01594599002246301, 0.015375891426716568, 0.01481768940837479, 0.014271232315940017, 0.013736406177679418, 0.0132131297954045, 0.012701350125622195, 0.012201037911616098, 0.011712183537854316, 0.011234793087680746, 0.010768884601591103, 0.010314484563324799, 0.009871624683620815, 0.009440339074995604, 0.009020661836435401, 0.00861262483998196, 0.008216255308378275, 0.007831573026107397, 0.007458587712958434, 0.00709729716618, 0.006747685930646964, 0.0064097241102646285, 0.006083366359535759, 0.00576855106766434, 0.005465199728141098, 0.005173216485950009, 0.004892487881510574, 0.004622882884662521, 0.00436425339657267, 0.0041164353032835, 0.0038792496708656313, 0.0036525032106725867, 0.0034359879651856278, 0.0032294816905907987, 0.003032749367229718, 0.0028455445020480635, 0.0026676103362113647, 0.002498681037130545, 0.0023384828828620207, 0.0021867354724963307, 0.0020431530786892923, 0.0019074462964683838, 0.0017793238420103288, 0.0016584936919347123, 0.0015446630700682125, 0.0014375387307947198, 0.0013368281418238467, 0.0012422397848401202, 0.0011534825563272997, 0.0010702635222137507, 0.000992278789203804, 0.0009191656864383693, 0.0008502714400422238, 0.0007837676684006828, 0.0007141148780644238, 0.0006269797760157475, 0.0004931442328178156])
push!(B,0.54)
push!(spotArray2,[0.54, 0.5593973341098653, 0.578169554997578, 0.5963432961004607, 0.6139443417515164, 0.6309976637614306, 0.6475274568477923, 0.6635571729617984, 0.6791095545611457, 0.6942066668763163, 0.708869929216032, 0.7231201453562981, 0.7369775330561469, 0.7504617527419581, 0.7635919354010561, 0.7763867097241518, 0.7888642285351457, 0.8010421945457826, 0.8129378854717044, 0.8245681785455293, 0.8359495744617396, 0.8470982207873485, 0.8580299348715619, 0.8687602262869351, 0.8793043188338684, 0.8896771721396564, 0.8998935028827371, 0.9099678056722535, 0.9199143736125492, 0.9297473185817765, 0.9394805912533849, 0.9491280008888996, 0.9587032349300668, 0.968219878418167, 0.9776914332680441, 0.9871313374241987, 0.9965529839261212, 1.0059697399099157, 1.015394965573173, 1.0248420331299988, 1.0343243457830915, 1.0438553567397846, 1.0534485882990356, 1.063117651036438, 1.072876263114479, 1.0827382697454375, 1.092717662834536, 1.1028286008312167, 1.1130854288167042, 1.123502698856357, 1.1340951906456802, 1.1448779324792926, 1.1558662225726009, 1.1670756507664246, 1.178522120645375, 1.1902218721013595, 1.2021915043742313, 1.2144479996022692, 1.2270087469159017, 1.2398915671088586, 1.2531147379217549, 1.2666970199739738, 1.2806576833806438, 1.2950165350924745, 1.309793946997234, 1.3250108848227447, 1.3406889378823985, 1.356850349705398, 1.3735180495951758, 1.3907156851607683, 1.4084676558673048, 1.4267991476531983, 1.4457361686631658, 1.465305586147767, 1.4855351645818138, 1.506453605055737, 1.5280905859957914, 1.5504768052708733, 1.5736440237456941, 1.597625110342097, 1.6224540886724519, 1.648166185311285, 1.6747978797736396, 1.7023869562710634, 1.730972557318664, 1.7605952392692754, 1.791297029853542, 1.8231214878075397, 1.8561137646725439, 1.890320668854617, 1.9257907320349057, 1.9625742780248636, 2.000723494164098, 2.040292505362121, 2.081337450889081, 2.1239165640243867, 2.168090254676259, 2.2139211950894113, 2.26147440876246, 2.3108173627012225, 2.3620200631387864, 2.4151551548581525, 2.470298024258373, 2.5275269063104053, 2.5869229955544304, 2.648570561296117, 2.7125570671652675, 2.778973295206464, 2.8479134746777848, 2.9194754157403127, 2.9937606482281263, 3.070874565695628, 3.150926574946622, 3.2340302512572356, 3.3203034995129697, 3.409868721488431, 3.5028529895071365, 3.5993882267277257, 3.6996113943124076, 3.8036646857431347, 3.9116957285612726, 4.0238577938168945, 4.140310013524957, 4.261217606436775, 4.3867521124472315, 4.517091635970159, 4.652421098627305, 4.792932501609322])
push!(priceArray2,[-1.8936971288266193e-18, 0.09049214149976108, 0.1302436674266077, 0.15136965738631686, 0.1625342457194008, 0.1679548868923357, 0.16984428321032488, 0.1694611545123685, 0.16756748435470245, 0.16464700012106215, 0.16101789949642603, 0.15689495995024738, 0.1524256811648369, 0.1477123707131168, 0.1428263067578704, 0.13781731521539237, 0.13272065548817183, 0.12756230187242315, 0.12236320516042629, 0.11714277188304574, 0.1119215592548602, 0.10672305987191842, 0.10157445917466647, 0.09650637148892391, 0.09155172304167777, 0.08674406116730958, 0.08211557341826989, 0.07769514937806384, 0.0735065426831214, 0.06956698100986272, 0.06588634392565466, 0.062467016768650886, 0.05930442940381013, 0.056388171941387594, 0.05370347999442944, 0.05123287039027426, 0.04895753476410795, 0.04685857887847734, 0.044917863352384194, 0.043118517863989604, 0.04144518164458892, 0.03988407397904975, 0.03842294868161039, 0.03705098790354755, 0.035758671696509925, 0.034537631993703646, 0.0333805199029991, 0.03228088023989553, 0.031233045486990463, 0.030232021626844217, 0.029273407905000094, 0.02835332095578086, 0.02746832930753835, 0.02661539703115846, 0.025791835304154936, 0.02499526074714878, 0.024223560656961875, 0.023474860002926964, 0.022747495198263103, 0.022039990650066992, 0.021351038254461836, 0.02067947946712189, 0.02002428970632182, 0.019384564907516704, 0.018759510006415372, 0.018148429002355217, 0.017550716147128637, 0.01696584783365428, 0.016393357345331854, 0.01583285886289769, 0.015284027247158076, 0.014746591972723648, 0.014220331479972974, 0.01370506789577797, 0.013200662079170943, 0.012707008954644383, 0.012224033101948445, 0.011751684576512392, 0.011289934938965188, 0.010838773476656407, 0.01039820360744587, 0.009968239471704724, 0.009548902747783802, 0.009140219763361764, 0.008742218982388188, 0.008354928851161757, 0.007978375762987868, 0.007612581759892733, 0.007257561925571483, 0.006913322065831554, 0.00657985714328364, 0.00625715008096332, 0.005945170693361275, 0.005643874785403925, 0.005353203424750281, 0.005073082380894816, 0.00480342172429042, 0.004544115604825309, 0.00429504229800562, 0.004056064681925364, 0.0038270312129891257, 0.0036077770093336427, 0.0033981242440210228, 0.003197881833975983, 0.003006845791645157, 0.0028248005999635404, 0.002651520391583046, 0.0024867700400625923, 0.002330306236633275, 0.0021818785581599765, 0.0020412305494058518, 0.0019081009121456269, 0.0017822249426604532, 0.0016633361490543043, 0.00155116741208126, 0.0014454510801791773, 0.0013459189627377348, 0.0012523033486439527, 0.0011643372791582645, 0.0010817538937044023, 0.0010042840905352143, 0.0009316471760660908, 0.0008635035597269694, 0.000799232973264765, 0.0007371059956817622, 0.0006719614395909245, 0.000590652005104397, 0.0004667670518973065])
push!(B,0.56)
push!(spotArray2,[0.56, 0.5786030206267929, 0.5966178508678748, 0.6140696360148992, 0.630982735455188, 0.6473807566728247, 0.6632865882002311, 0.678722431566225, 0.6937098322851493, 0.7082697099303307, 0.7224223873338425, 0.7361876189533301, 0.7495846184454921, 0.762632085484708, 0.7753482318642415, 0.7877508069164558, 0.7998571222875204, 0.8116840761011906, 0.823248176545389, 0.8345655649145092, 0.8456520381396064, 0.8565230708379223, 0.8671938369125192, 0.8776792307321764, 0.887993887921108, 0.8981522057875233, 0.9081683634195423, 0.9180563414765177, 0.9278299417033882, 0.9375028061953007, 0.9470884364393918, 0.9566002121603072, 0.9660514099957638, 0.9754552220282214, 0.9848247741985322, 0.9941731446272685, 1.003513381869302, 1.0128585231271163, 1.0222216124482726, 1.031615718932429, 1.0410539549733304, 1.0505494945612261, 1.0601155916712655, 1.0697655987635382, 1.079512985420579, 1.0893713571483556, 1.0993544743669774, 1.1094762716176367, 1.1197508770125897, 1.1301926319553253, 1.1408161111584494, 1.1516361429872228, 1.1626678301571522, 1.1739265708145206, 1.1854280800292831, 1.1971884117303297, 1.2092239811137298, 1.2215515875552363, 1.234188438059035, 1.247152171275462, 1.2604608821212209, 1.2741331470364592, 1.2881880499139609, 1.3026452087366485, 1.31752480296057, 1.3328476016815998, 1.348634992625163, 1.364909011999451, 1.3816923752537937, 1.3990085087851265, 1.416881582636803, 1.4353365442354007, 1.4543991532126026, 1.4740960173607651, 1.4944546297723587, 1.515503407215118, 1.537271729796469, 1.559789981972594, 1.5830895949593806, 1.607203090604446, 1.632164126781479, 1.6580075443702564, 1.6847694158879138, 1.7124870958393505, 1.7411992728570431, 1.7709460237030499, 1.8017688692085834, 1.8337108322292255, 1.8668164976966897, 1.9011320748509417, 1.9367054617395474, 1.9735863120742771, 2.0118261045382893, 2.051478214640612, 2.0925979892182616, 2.135242823689935, 2.1794722421691515, 2.22534798054864, 2.272934072671955, 2.322296939712602, 2.3735054828854096, 2.4266311796195814, 2.481748183327657, 2.5389334269096255, 2.598266730136689, 2.6598309110645526, 2.723711901631738, 2.789998867604308, 2.858784333034383, 2.930164309406198, 3.004238429649959, 3.0811100872105412, 3.160886580365158, 3.2436792619914465, 3.3296036949949848, 3.4187798136132406, 3.5113320908210395, 3.607389712071277, 3.7070867556133438, 3.81056237964098, 3.9179610165307466, 4.029432574442292, 4.145132646561736, 4.265222728280309, 4.389870442611331, 4.519249774160196, 4.653541311973929, 4.79293250160932])
push!(priceArray2,[5.350843794523612e-19, 0.0782656580368799, 0.11523307847040862, 0.13550931840462174, 0.14663845481454937, 0.15237131936996576, 0.15472472971854967, 0.15486726746218812, 0.15351657067223393, 0.1511332801985823, 0.14802312223005076, 0.1443940461778904, 0.1403899445154661, 0.13611155424204974, 0.13163008151498273, 0.12699659112667427, 0.12224887678054565, 0.11741675994295592, 0.11252627532868781, 0.10760287881097565, 0.10267363579759248, 0.09776831978902727, 0.09291944887092592, 0.08816143565169204, 0.08352912804714363, 0.07905603058309231, 0.07477240181156876, 0.07070367244434347, 0.06686897403405936, 0.06328027087412195, 0.05994211686321746, 0.05685206792119104, 0.0540016709068122, 0.05137784964916017, 0.04896448318730021, 0.046743803702629455, 0.04469766638492931, 0.042808446367129514, 0.041059609378198206, 0.03943601417668721, 0.03792401626024069, 0.03651145290415986, 0.03518755650617303, 0.033942838741338646, 0.03276894860228697, 0.03165854705792302, 0.03060518465118697, 0.02960319605609771, 0.02864758933171956, 0.027733964833319556, 0.026858439464299586, 0.026017581119579063, 0.025208352190768286, 0.024428060976878392, 0.023674319893852408, 0.022945013123097702, 0.02223825823393602, 0.02155238285615731, 0.020885901314418125, 0.020237494229739247, 0.019605990656299768, 0.0189903524392503, 0.018389660572565063, 0.017803103368715358, 0.01722996620711257, 0.016669622541781975, 0.016121525806924317, 0.015585187642783308, 0.01506019817139427, 0.014546208672703604, 0.014042925385917136, 0.013550103812668739, 0.013067543471629882, 0.012595083048068254, 0.01213259588423768, 0.011679985762774955, 0.011237182942731188, 0.010804140415024213, 0.010380830350558694, 0.009967240720966983, 0.00956337208200456, 0.00916923452855016, 0.008784844861526468, 0.008410224039401976, 0.008045394976188731, 0.007690380630299597, 0.007345202121154421, 0.007009876551673093, 0.006684414609648199, 0.006368818552732984, 0.006063080884869323, 0.005767183274994337, 0.005481095599444556, 0.0052047751372443956, 0.004938165922042613, 0.004681198244367043, 0.004433788297958973, 0.004195837989366849, 0.0039672349936351456, 0.003747853204277154, 0.0035375536310359407, 0.0033361853759214318, 0.0031435859634962684, 0.0029595810395059226, 0.0027839846899796144, 0.0026166006848158973, 0.0024572235389877764, 0.002305639491845688, 0.0021616274729170776, 0.002024960058363266, 0.0018954044325183643, 0.0017727234248043633, 0.001656676746270476, 0.0015470224189899529, 0.0014435179264454914, 0.001345920418685008, 0.0012539865105309946, 0.001167473066410587, 0.0010861374609383472, 0.0010097368752355674, 0.0009380258740754188, 0.000870746889137663, 0.0008075836846472295, 0.0007479498898519676, 0.0006902184158827907, 0.0006296076574516963, 0.0005541308678627604, 0.00044006871794004384])
push!(B,0.58)
push!(spotArray2,[0.58, 0.5978210066460758, 0.6150901242493281, 0.6318310503099791, 0.6480667575188263, 0.6638195252814554, 0.6791109702910954, 0.6939620761920667, 0.7083932223745317, 0.722424211940057, 0.7360742988763677, 0.7493622144785789, 0.7623061930531663, 0.7749239969399435, 0.7872329408863854, 0.7992499158077437, 0.8109914119655601, 0.8224735415963834, 0.8337120610217427, 0.8447223922697171, 0.855519644237773, 0.8661186334259073, 0.8765339042685498, 0.8867797490931246, 0.8968702277326579, 0.9068191868193457, 0.9166402787855582, 0.9263469805983532, 0.9359526122532076, 0.9454703550523459, 0.9549132696927463, 0.9642943141886476, 0.973626361653149, 0.9829222179633047, 0.992194639332955, 1.0014563498174058, 1.01072005877398, 1.0199984783023976, 1.0293043406889226, 1.0386504158782062, 1.0480495289968101, 1.057514577952453, 1.0670585511331279, 1.0766945452303835, 1.0864357832112246, 1.0962956324632918, 1.1062876231382242, 1.1164254667183733, 1.1267230748323476, 1.137194578345208, 1.1478543467495093, 1.1587170078837974, 1.1697974680056218, 1.1811109322466071, 1.192672925477655, 1.2044993136129047, 1.2166063253816934, 1.2290105745983846, 1.241729082960631, 1.2547793034073529, 1.2681791440684878, 1.2819469928393727, 1.2961017426134838, 1.3106628172081596, 1.3256501980188813, 1.3410844514386904, 1.3569867570803638, 1.373378936840083, 1.3902834848424719, 1.4077235983080971, 1.425723209385794, 1.4443070179934923, 1.4635005257126124, 1.4833300707825412, 1.503822864243212, 1.525007027275379, 1.5469116297898338, 1.5695667303185101, 1.5930034172622265, 1.6172538515516575, 1.6423513107800873, 1.6683302348684956, 1.6952262733256476, 1.7230763341680364, 1.7519186345668099, 1.7817927532911793, 1.8127396850202815, 1.8448018965980157, 1.8780233853080563, 1.912449739249007, 1.9481281998925466, 1.985107726910412, 2.0234390653591765, 2.063174815315014, 2.1043695040540165, 2.1470796608771012, 2.1913638946821896, 2.237282974390114, 2.284899912334601, 2.3342800507307793, 2.3854911513408625, 2.4386034884600374, 2.493689945350182, 2.550826114253728, 2.6100904001249052, 2.6715641282207363, 2.735331655699397, 2.801480487379103, 2.8701013958163593, 2.941288545868359, 3.015139623910455, 3.091755971886024, 3.1712427263726797, 3.253708962855655, 3.339267845406345, 3.4280367819714024, 3.520137585485475, 3.6156966410286704, 3.7148450792581578, 3.8177189563518477, 3.924459440711121, 4.03521300667879, 4.150131635538114, 4.269373024068715, 4.393100800945566, 4.521484751277974, 4.6547010495967704, 4.79293250160932])
push!(priceArray2,[-1.0979036990064197e-18, 0.06745944706028875, 0.1014783086357285, 0.12073437247466165, 0.13167530518570233, 0.13759357943186976, 0.14030536107057084, 0.1408841597856295, 0.1399992083529778, 0.13808512952056415, 0.13543312639953808, 0.1322429328578288, 0.12865397764225334, 0.12476503263664228, 0.12064726195564804, 0.1163533794661314, 0.11192441256009206, 0.10739485073273722, 0.10279651133829079, 0.09816120611427095, 0.09352221787704586, 0.08891466034754814, 0.08437491470827564, 0.07993942058182286, 0.07564310465242423, 0.07151760250706281, 0.0675898662591317, 0.0638806946458059, 0.06040386140667432, 0.0571658191368119, 0.054166024033207145, 0.05139782464793361, 0.048849759279523286, 0.04650706104927014, 0.04435306849799364, 0.04237049035073326, 0.040542339184006196, 0.03885254736622777, 0.03728631522006528, 0.03583024497815439, 0.03447235620182484, 0.03320201859047305, 0.03200984840933458, 0.03088757842887497, 0.029827937353149692, 0.028824531630384075, 0.027871737150773585, 0.026964602717613122, 0.026098763573041348, 0.025270366557797713, 0.024476005157773675, 0.023712663410859553, 0.022977667596194032, 0.022268644650709872, 0.02158348826324671, 0.020920324232870654, 0.020277485370227045, 0.01965348824059118, 0.019047012872040844, 0.018456885003824357, 0.017882060573311025, 0.017321612243895914, 0.01677471781900406, 0.016240650339823293, 0.015718769552024992, 0.015208514340849029, 0.014709395769860552, 0.0142209771302258, 0.013742891572566899, 0.013274825989593037, 0.012816515837580799, 0.012367740316406623, 0.011928317867334351, 0.011498101952234349, 0.01107697708282927, 0.010664855073199305, 0.010261671492780854, 0.00986738230038763, 0.009481960642765281, 0.00910539380531898, 0.008737680311415611, 0.008378827185423656, 0.008028847425136432, 0.0076877577543256834, 0.007355576695762043, 0.00703232286734106, 0.006718013222648768, 0.006412660993892447, 0.006116273528291883, 0.005828850574923358, 0.005550383158562591, 0.005280852611548145, 0.00502022972196177, 0.004768474019557041, 0.0045255332024590035, 0.004291342697622332, 0.004065825349709875, 0.0038488912573554014, 0.003640437833806725, 0.0034403502253398995, 0.003248502128301299, 0.003064756660226159, 0.0028889666341990745, 0.0027209742698408525, 0.0025606114769587836, 0.00240770097216467, 0.002262057227989142, 0.002123487346230196, 0.001991791918966536, 0.001866765880521785, 0.001748199358263486, 0.0016358785731261805, 0.0015295868943734385, 0.0014291060849264505, 0.0013342174071800503, 0.0012447019323361072, 0.0011603402558081025, 0.0010809129958871833, 0.0010062010516212712, 0.0009359848696682323, 0.0008700419649457249, 0.0008081373341359386, 0.0007499780708184287, 0.0006950133333386815, 0.0006417233507444828, 0.0005857117086110239, 0.0005161348315412468, 0.0004118595300873475])
push!(B,0.6)
push!(spotArray2,[0.6, 0.6170521305027947, 0.6335880272688981, 0.6496299815511339, 0.665199618742049, 0.6803179275260733, 0.6950052881733628, 0.7092815000134689, 0.7231658081258683, 0.7366769292833364, 0.7498330771831351, 0.7626519870000276, 0.7751509392942211, 0.7873467833064648, 0.7992559596717068, 0.810894522581929, 0.8222781614280364, 0.8334222219499766, 0.8443417269235979, 0.8550513964121371, 0.8655656676096333, 0.8758987143030199, 0.8860644659791301, 0.8960766266023727, 0.9059486930883922, 0.9156939734986167, 0.9253256049802192, 0.9348565714756794, 0.9442997212258155, 0.9536677840898853, 0.9629733887061004, 0.9722290795156913, 0.9814473336734689, 0.9906405778676819, 0.9998212050718408, 1.0090015912510948, 1.0181941120456786, 1.0274111594539224, 1.0366651585373134, 1.0459685841701272, 1.0553339778562107, 1.0647739646355845, 1.0743012701036563, 1.0839287375659878, 1.0936693453517436, 1.1035362243091547, 1.1135426755065896, 1.1237021881630882, 1.1340284578325353, 1.1445354048659813, 1.1552371931770042, 1.1661482493354038, 1.1772832820149726, 1.1886573018215558, 1.2002856415281329, 1.2121839767441962, 1.2243683470472924, 1.23685517760521, 1.2496613013179643, 1.262803981509426, 1.276300935199184, 1.2901703569860132, 1.3044309435751449, 1.3191019189824005, 1.334203060449172, 1.3497547251031745, 1.3657778774009195, 1.3822941173888963, 1.3993257098215643, 1.4168956141754052, 1.4350275155994958, 1.453745856844328, 1.473075871211913, 1.4930436165715903, 1.5136760104873976, 1.5350008665043537, 1.5570469316425704, 1.5798439251497358, 1.6034225785642164, 1.6278146771427733, 1.6530531027087516, 1.6791718779784945, 1.7062062124257422, 1.7341925497458452, 1.7631686169837666, 1.7931734753921098, 1.8242475730877317, 1.8564327995779126, 1.8897725422296054, 1.9243117447578697, 1.960096967812348, 1.997176451743458, 2.035600181632902, 2.0754199546761667, 2.116689450007856, 2.159464301063963, 2.2038021705786495, 2.249762828316628, 2.297408231645922, 2.346802609059642, 2.398012546759344, 2.4511070784167073, 2.506157778234541, 2.5632388574325367, 2.6224272642878788, 2.683802787865537, 2.7474481655780982, 2.81344919472012, 2.881894848127372, 2.9528773941168502, 3.0264925208693154, 3.1028394654219387, 3.1820211474450204, 3.2641443079830856, 3.3493196533473824, 3.437662004353785, 3.5292904511072414, 3.6243285135414554, 3.7229043079302304, 3.825150719594878, 3.931205582040586, 4.0412118627631815, 4.155317855976785, 4.273677382522166, 4.396449997225266, 4.52380120398543, 4.655902678883343, 4.79293250160932])
push!(priceArray2,[-2.5917465398697486e-19, 0.05792840406310462, 0.0889151057795343, 0.10701037858287357, 0.11762466307960592, 0.12360823553686502, 0.12657660199799894, 0.12750463145645877, 0.1270097501885214, 0.12549795814240422, 0.12324411083933068, 0.12043856870770925, 0.11721571420123256, 0.1136722813905084, 0.10987975837985048, 0.10589319587927422, 0.1017576725729072, 0.09751303161816165, 0.09319714972126057, 0.08884786236168418, 0.08450368615324781, 0.08020356443764712, 0.07598592350381302, 0.07188732337965974, 0.06794085030763428, 0.06417487954717008, 0.060611679947084524, 0.05726658036643718, 0.054147664379834176, 0.05125604774774401, 0.04858669765247705, 0.04612965782662776, 0.04387147931265049, 0.04179663835879352, 0.03988877467364582, 0.038131643016636664, 0.0365097586925068, 0.035008778280241425, 0.033615661985909887, 0.03231871341264681, 0.031107530664565383, 0.029972913586467503, 0.02890674963247945, 0.027901897130329437, 0.02695207235723665, 0.02605174510655694, 0.02519604901043274, 0.024380688681524935, 0.023601871207459896, 0.022856242391287265, 0.022140831563570853, 0.02145300397676089, 0.020790419794511676, 0.02015099875200007, 0.019532891394963968, 0.018934449299019055, 0.018354203800542507, 0.017790846110226115, 0.0172432099452521, 0.016710256334657093, 0.016191060347140716, 0.0156847995562527, 0.015190744067615978, 0.014708247883031269, 0.014236741308219425, 0.01377572409615844, 0.01332474583607352, 0.012883425769643178, 0.012451436310683973, 0.012028497850863792, 0.011614373992039597, 0.01120886715978125, 0.010811814549043845, 0.01042308435570642, 0.0100425722533017, 0.009670198080655541, 0.009305902712247276, 0.008949645088479438, 0.008601399388017596, 0.008261152330498846, 0.007928900608780837, 0.007604648470361003, 0.0072884054971891375, 0.0069801846502221794, 0.006680000596807873, 0.006387868187956679, 0.00610380080520974, 0.005827808426210564, 0.005559895706511893, 0.005300060550489909, 0.005048293139525878, 0.00480457506675676, 0.004568878580141724, 0.004341165951371295, 0.00412138897062274, 0.003909488560115516, 0.0037053945020231248, 0.003509025298893918, 0.0033202882370560916, 0.0031390797717806888, 0.0029652862645353603, 0.0027987847552100554, 0.0026394431891341857, 0.0024871201437471195, 0.0023416650755222774, 0.0022029193144359875, 0.002070716906441863, 0.0019448853857162132, 0.001825246535397951, 0.001711617139684782, 0.0016038097303985046, 0.0015016333630440415, 0.0014048945070692625, 0.0013133981107149447, 0.0012269486254732065, 0.0011453504042505556, 0.0010684074253570312, 0.0009959234810797595, 0.0009277024163882152, 0.0008635473803895035, 0.0008032583037474821, 0.0007466223008229452, 0.0006933696569984776, 0.0006429856818972814, 0.000594058672852796, 0.0005425697983009377, 0.00047877063912453644, 0.00038395895237070275])
push!(B,0.62)
push!(spotArray2,[0.62, 0.6362973191200498, 0.6521133873489002, 0.6674691324864561, 0.6823848732337363, 0.6968803460785504, 0.7109747314107928, 0.7246866789019085, 0.7380343321821128, 0.7510353528480179, 0.7637069438324338, 0.7760658721672652, 0.7881284911696255, 0.7999107620805236, 0.811428275184756, 0.822696270439951, 0.8337296576420588, 0.8445430361539745, 0.8551507142233938, 0.8655667279154683, 0.8758048596853059, 0.8858786566148986, 0.8958014483386014, 0.9055863646808872, 0.9152463530297119, 0.9247941954684813, 0.9342425256892857, 0.9436038457097845, 0.9528905424158586, 0.9621149039519212, 0.9712891359805735, 0.980425377833121, 0.9895357185723197, 0.998632212988608, 1.0077268975509888, 1.0168318063336683, 1.0259589869395278, 1.0351205164414938, 1.0443285173629062, 1.053595173718023, 1.0629327471338919, 1.0723535930749195, 1.0818701771916037, 1.0914950918150654, 1.1012410726192048, 1.111121015472527, 1.1211479935019382, 1.1313352743910885, 1.1416963379361533, 1.152244893882282, 1.1629949000643127, 1.1739605808757607, 1.185156446090517, 1.1965973100621616, 1.2082983113262944, 1.2202749326318263, 1.232543021427731, 1.2451188108323683, 1.2580189411131226, 1.2712604817047843, 1.284860953795801, 1.2988383535122905, 1.3132111757304903, 1.3279984385491534, 1.3432197084542694, 1.358895126209413, 1.375045433505974, 1.3916920004085362, 1.408856853631717, 1.4265627056858858, 1.4448329849303274, 1.463691866573615, 1.4831643046622127, 1.5032760650996368, 1.524053759739862, 1.5455248816000933, 1.5677178412394865, 1.590662004351962, 1.614387730622851, 1.6389264139007897, 1.6643105237380182, 1.6905736483540452, 1.7177505390795362, 1.7458771563392257, 1.7749907172347044, 1.8051297447900323, 1.8363341189253586, 1.8686451292259707, 1.9021055295766223, 1.936759594733413, 1.972653178908085, 2.009833776442254, 2.0483505846518595, 2.08825456892498, 2.129598530159172, 2.172437174627542, 2.216827186366012, 2.262827302177561, 2.310498389352679, 2.359903526208887, 2.41110808555588, 2.464179821196732, 2.5191889575796473, 2.5762082827188477, 2.635313244507575, 2.6965820505506417, 2.7600957716486234, 2.8259384490706356, 2.894197205757627, 2.9649623616033276, 3.0383275529654212, 3.114389856565037, 3.1932499179385387, 3.2750120846115602, 3.35978454417151, 3.447679467421232, 3.538813156803272, 3.6333062002911034, 3.731283630950963, 3.8328750923854478, 3.938215010277743, 4.047442770263505, 4.160702902365747, 4.278145272236761, 4.399925279460155, 4.5262040631753635, 4.6571487152967315, 4.792932501609323])
push!(priceArray2,[1.30883867912989e-18, 0.0495489037373611, 0.07748034606921833, 0.09430374266757734, 0.10446736305300046, 0.11040317791726409, 0.11352964228786959, 0.11472129665944605, 0.11454131554899603, 0.11336501296857152, 0.11144945592371845, 0.10897482110975089, 0.1060702152331346, 0.10283060280773161, 0.09932842509498854, 0.09562184889510623, 0.09176065852427821, 0.08779029709538082, 0.08375433638654983, 0.07969561325993278, 0.0756563127959875, 0.07167731023611411, 0.06779706465696257, 0.06405021972173729, 0.06046651817496782, 0.057069534859310656, 0.05387592069258245, 0.050895130451005535, 0.04812969319698702, 0.04557599456116144, 0.0432254498281699, 0.041065882859649905, 0.03908291800931353, 0.03726117788491302, 0.03558524702218471, 0.03404033004415683, 0.03261265119735248, 0.03128963896794134, 0.03005998411287159, 0.028913606796433886, 0.0278415760805081, 0.026836009058780586, 0.025889957784674362, 0.024997304484972338, 0.024152661932394445, 0.023351288049014096, 0.022588994835667037, 0.021862082931172384, 0.02116727972903198, 0.020501685696803662, 0.01986272800053784, 0.019248120516736692, 0.018655829355318645, 0.01808404570110086, 0.017531154768444256, 0.016995716119775038, 0.016476444245562997, 0.015972191589773622, 0.015481933673295464, 0.015004756069307772, 0.01453984307374436, 0.014086467959142094, 0.01364398467501808, 0.013211820771127671, 0.012789471230290976, 0.01237649288291356, 0.01197249312120603, 0.011577134155987063, 0.011190124728223921, 0.010811215850720353, 0.010440196773715684, 0.010076891182091917, 0.009721153625945214, 0.009372866175676802, 0.009031935284236568, 0.0086982888350013, 0.008371873353454757, 0.008052651362950925, 0.00774059886860981, 0.007435702959711247, 0.007137959532834051, 0.00684737115887882, 0.006563945143394543, 0.0062876918360679915, 0.006018623181362699, 0.005756751349747431, 0.005502087185582278, 0.0052546384135013025, 0.005014407975523891, 0.004781392863266091, 0.004555583270908792, 0.004336961826289662, 0.004125502923963185, 0.003921172170884441, 0.003723925943113854, 0.0035337110472220016, 0.0033504644827711243, 0.0031741133231264684, 0.003004574778548509, 0.0028417565461687708, 0.0026855574681957433, 0.0025358682105635897, 0.002392571450554364, 0.002255541622959408, 0.002124645130757483, 0.001999741224667876, 0.0018806827471263376, 0.0017673168114808002, 0.0016594854706576685, 0.0015570263780335352, 0.0014597734402650583, 0.001367557484471707, 0.001280207005156275, 0.0011975490586545918, 0.0011194101813072634, 0.001045616861092272, 0.0009759953343877486, 0.0009103714921811288, 0.0008485710686494511, 0.00079041889979145, 0.0007357363875086759, 0.0006843319906321184, 0.000635958941894161, 0.0005901398553392843, 0.0005455741305479354, 0.0004986193611106282, 0.00044058871292816594, 0.00035508119614171353])
push!(B,0.64)
push!(spotArray2,[0.6400000000000001, 0.655557598400998, 0.6706682278208901, 0.685351496712655, 0.6996264589607899, 0.7135116386067646, 0.7270250538869234, 0.7401842406140201, 0.7530062749327349, 0.7655077954786957, 0.7777050249697645, 0.789613791257601, 0.8012495478668273, 0.812627394048441, 0.8237620943735033, 0.8346680978925268, 0.8453595568854227, 0.8558503452263437, 0.8661540763872491, 0.8762841211035568, 0.886253624724807, 0.8960755242728501, 0.9057625652296973, 0.9153273180768179, 0.9247821946073437, 0.9341394640323526, 0.943411268902127, 0.9526096408630509, 0.9617465162705907, 0.9708337516786213, 0.9798831392251972, 0.9889064219347325, 0.9979153089564501, 1.0069214907588702, 1.0159366543000596, 1.0249724981933244, 1.0340407478880274, 1.0431531708852302, 1.0523215920079034, 1.0615579087455214, 1.0708741066929532, 1.0802822751036856, 1.089794622577557, 1.0994234929033653, 1.1091813810769027, 1.1190809495152076, 1.1291350444880708, 1.1393567127881197, 1.1497592186611156, 1.1603560610184278, 1.1711609909540253, 1.1821880295887164, 1.1934514862647896, 1.2049659771146686, 1.2167464440276778, 1.2288081740395262, 1.241166819169677, 1.2538384167323404, 1.2668394101474472, 1.2801866702786109, 1.2938975173257656, 1.3079897433008887, 1.3224816351159756, 1.337391998313227, 1.35274018146824, 1.3685461012978726, 1.3848302685053628, 1.4016138143962378, 1.4189185182995572, 1.4367668358300651, 1.4551819280279352, 1.474187691413917, 1.4938087889988834, 1.5140706822880232, 1.5349996643212087, 1.556622893792409, 1.5789684302924292, 1.6020652707207081, 1.625943386913423, 1.6506337645367275, 1.6761684432956008, 1.7025805585104758, 1.7299043841156063, 1.7581753771349644, 1.7874302236933888, 1.8177068866226846, 1.8490446547244568, 1.8814841937536002, 1.9150675991886041, 1.9498384508571545, 1.9858418694879136, 2.0231245752618667, 2.061734948439213, 2.10172309214047, 2.1431408973632724, 2.186042110319221, 2.2304824021781635, 2.2765194413104304, 2.3242129681207406, 2.373624872570913, 2.424819274491967, 2.4778626067898375, 2.532823701652667, 2.589773879871559, 2.6487870433906817, 2.7099397712068347, 2.773311418742921, 2.838984220824264, 2.9070433983914254, 2.977577269087964, 3.0506773618666827, 3.1264385357630435, 3.2049591029899056, 3.2863409565133117, 3.3706897022748663, 3.4581147962323016, 3.548729686396049, 3.642651960046133, 3.740003496320435, 3.8409106243723223, 3.9455042873028825, 4.053920212080493, 4.166299085668228, 4.28278673758763, 4.4035343291557965, 4.528698549641308, 4.658441819593555, 4.792932501609322])
push!(priceArray2,[1.8617051570132356e-19, 0.04216964412853073, 0.06711183581497089, 0.08257937938946545, 0.0921811035707756, 0.09796221720727288, 0.10115085421699439, 0.10252129267591858, 0.10258099620533744, 0.10167317082991541, 0.100036159172698, 0.09783959607591326, 0.09520748474799083, 0.09223358373227182, 0.0889920172688461, 0.08554467432464694, 0.08194623309323096, 0.07824729824149784, 0.07449602098399787, 0.07073856117971264, 0.06701874474572067, 0.06337722710188778, 0.05985033883836704, 0.05646916047915268, 0.05325843156563118, 0.050235910626153533, 0.047412176661875374, 0.04479093139324028, 0.04236977679871091, 0.04014135749352179, 0.03809469684251246, 0.03621655026331922, 0.0344925654966032, 0.03290823489531185, 0.03144955131056721, 0.03010341396153917, 0.02885782709194274, 0.027701970704208753, 0.026626180954948713, 0.025621881466733068, 0.02468149332337445, 0.023798328685516935, 0.022966493959019476, 0.022180795853960225, 0.0214366595278114, 0.020730043664686542, 0.020057376916692415, 0.019415498577324285, 0.01880160699540782, 0.0182132149234536, 0.017648110961074232, 0.017104326279570224, 0.016580107723070035, 0.016073889222431092, 0.015584272071965955, 0.015110006213320759, 0.014649973921015899, 0.014203175554578484, 0.013768717124173541, 0.013345799484403013, 0.012933709002103201, 0.012531809524088404, 0.012139535414512584, 0.011756385388099219, 0.011381916888274113, 0.011015729174951985, 0.010657481148014614, 0.010306876497953256, 0.009963659533457363, 0.009627611348643607, 0.009298546289941844, 0.008976308682835422, 0.008660769782170502, 0.008351824914578615, 0.0080493907866149, 0.007753402936921734, 0.007463813314818403, 0.007180587971295759, 0.006903704852297231, 0.006633151690526419, 0.006368924004277096, 0.006111023231983144, 0.005859455051745572, 0.005614227927554154, 0.005375351846018555, 0.005142837067350803, 0.0049166926678941405, 0.004696924907281609, 0.0044835358180322195, 0.004276522257368012, 0.004075875159688355, 0.003881578858707238, 0.0036936105035643846, 0.003511939574327143, 0.003336527495166237, 0.003167327339257877, 0.003004283622338293, 0.0028473322009312132, 0.0026964003324482025, 0.0025514069883087647, 0.0024122634344294373, 0.002278873821553051, 0.002151135338831228, 0.002028937979866253, 0.0019121647166389043, 0.0018006922686833748, 0.001694391753437312, 0.0015931292770643179, 0.0014967665157736732, 0.001405161290515037, 0.0013181681328347553, 0.001235638854707385, 0.0011574231691844196, 0.0010833694252448392, 0.001013325403663091, 0.0009471388335985748, 0.0008846572896828539, 0.000825727897220389, 0.0007701974092546767, 0.0007179114974134894, 0.0006687122289864972, 0.000622428738775158, 0.0005788369782553819, 0.0005374976716740754, 0.0004972227331294951, 0.0004547402317345482, 0.0004023815777730873, 0.0003259049929674888])
push!(B,0.66)
push!(spotArray2,[0.66, 0.6748341044327028, 0.6892547901571209, 0.703280391756758, 0.716928741501773, 0.7302171920210816, 0.7431626383646373, 0.7557815394839428, 0.7680899391581025, 0.7801034863920203, 0.7918374553126791, 0.8033067645887964, 0.814525996398548, 0.8255094149694742, 0.8362709847141412, 0.8468243879846147, 0.8571830424683202, 0.8673601182474052, 0.877368554543295, 0.8872210761677297, 0.896930209701199, 0.9065082994193439, 0.915967522987575, 0.9253199069438602, 0.9345773419893688, 0.9437515981064104, 0.9528543395228917, 0.9618971395423157, 0.9708914952581793, 0.9798488421714763, 0.9887805687298914, 0.9976980308071706, 1.006612566141076, 1.0155355087482834, 1.0244782033345476, 1.0334520197184593, 1.0424683672871293, 1.051538709502182, 1.0606745784744975, 1.0698875896262379, 1.079189456458793, 1.0885920054454277, 1.098107191067561, 1.1077471110137962, 1.117524021561026, 1.127450353157168, 1.1375387262253431, 1.1478019672095892, 1.158253124882512, 1.168905486935608, 1.1797725968733466, 1.1908682712325005, 1.2022066171486077, 1.213802050291905, 1.2256693131955345, 1.237823493999327, 1.2502800456329919, 1.2630548054631052, 1.276164015428873, 1.2896243426922709, 1.30345290082882, 1.31766727158593, 1.3322855272364853, 1.3473262535560884, 1.3628085734531727, 1.3787521712820336, 1.3951773178696847, 1.4121048962883618, 1.4295564284064395, 1.4475541022515181, 1.466120800220475, 1.4852801281723385, 1.5050564454409783, 1.5254748958057742, 1.546561439459631, 1.568342886014991, 1.5908469285898048, 1.6141021790167989, 1.6381382042208066, 1.662985563810407, 1.6886758489316769, 1.7152417224334422, 1.7427169603951052, 1.7711364950698405, 1.800536459297764, 1.8309542324455312, 1.8624284879307895, 1.894999242391889, 1.9287079065653823, 1.96359733793599, 1.999711895225971, 2.037097494793186, 2.0758016690095507, 2.115873626694093, 2.1573643156774756, 2.2003264875774993, 2.244814764867969, 2.2908857103261666, 2.338597898947257, 2.38801199241704, 2.4391908162377334, 2.4921994396048586, 2.54710525813677, 2.603978079562032, 2.6628902124735503, 2.7239165582623444, 2.7871347063478225, 2.852625032825623, 2.920470802658488, 2.9907582755400286, 3.0635768155660656, 3.1390190048528863, 3.2171807612469494, 3.298161460275662, 3.382064061494267, 3.468995239389511, 3.5590655190065075, 3.6523894164712054, 3.7490855845872053, 3.849276963691922, 3.953090937964008, 4.060659497380671, 4.17211940553087, 4.287612373497725, 4.4072852400311895, 4.531290158240101, 4.659784789040949, 4.792932501609323])
push!(priceArray2,[-1.262555927790869e-19, 0.035696730910818306, 0.05774245418342769, 0.07179886456657245, 0.08074407424841896, 0.0862720301219197, 0.08943081094426249, 0.09089669741015448, 0.09112137145398884, 0.09041546063076084, 0.08899836787631486, 0.0870293903709466, 0.08462798284614931, 0.08188739201396938, 0.07888396298283423, 0.07568338934774028, 0.07234466456231764, 0.06892227558606254, 0.06546710144620911, 0.062026426400166625, 0.05864340548957568, 0.055356196980469057, 0.05219719975995664, 0.04919218961314829, 0.04635983172290529, 0.04371159392400603, 0.04125211724079679, 0.03898001944382814, 0.0368890279409794, 0.03496928262048452, 0.03320864439370204, 0.03159381829292316, 0.030111273797892555, 0.0287478838834041, 0.027491325287474108, 0.026330281735491293, 0.025254522119914656, 0.024254891047904653, 0.023323250594013523, 0.022452400132308377, 0.021635978548203066, 0.020868375101222077, 0.02014464194350522, 0.01946041624995327, 0.01881184302950585, 0.018195513727965223, 0.01760841012406223, 0.017047855434315, 0.016511471909953463, 0.015997144167069398, 0.015502987505957034, 0.015027321025739107, 0.01456864305740992, 0.014125610954638587, 0.013697023269070895, 0.013281804214406239, 0.012878990087685668, 0.012487717391225516, 0.012107212468868037, 0.011736782517079861, 0.011375807835782544, 0.01102373514431662, 0.010680071735677801, 0.010344380229644942, 0.010016267943138762, 0.00969539382753144, 0.009381459875024067, 0.009074207131188327, 0.008773412044110212, 0.008478883117480423, 0.008190457830278428, 0.007907999786367734, 0.007631396060780619, 0.007360554714057386, 0.007095402450741412, 0.006835882402486389, 0.0065819520199985335, 0.006333581061428516, 0.006090749668861407, 0.005853446531558857, 0.0056216671478250425, 0.005395412217070131, 0.00517468620790756, 0.0049594961255781935, 0.004749850412507834, 0.0045457578007469445, 0.004347225957462395, 0.0041542600423677386, 0.003966861544589458, 0.0037850275092184324, 0.003608749879898083, 0.003438014908315413, 0.003272802646895476, 0.0031130865290227437, 0.002958833034595362, 0.0028100014352951015, 0.0026665436170769242, 0.0025284039943737325, 0.0023955195661617677, 0.002267820192130024, 0.0021452290983506868, 0.002027663386531146, 0.0019150341613871696, 0.0018072463191977897, 0.001704198686015455, 0.0016057846826621892, 0.0015118928880545385, 0.00142240754772312, 0.0013372090735388202, 0.0012561745377990036, 0.0011791781584892939, 0.0011060917815644447, 0.001036785391155233, 0.0009711277011703844, 0.0009089868224618403, 0.0008502307805367912, 0.0007947275145669233, 0.0007423444991255103, 0.0006929486383673131, 0.0006464056032813007, 0.0006025773496463216, 0.0005613130564551634, 0.0005224111767088994, 0.0004854711839271385, 0.0004494181221959431, 0.0004113423817015914, 0.0003645464597129516, 0.00029680339565753814])
push!(B,0.68)
push!(spotArray2,[0.6800000000000002, 0.6941280952835371, 0.7078755573889676, 0.7212594939358912, 0.7342965601653578, 0.747002979665986, 0.7593945645629245, 0.7714867351947805, 0.783294539303, 0.7948326707575818, 0.8061154878424258, 0.817157031123072, 0.8279710409190639, 0.838570974402679, 0.848970022345307, 0.8591811255323115, 0.8692169908668048, 0.8790901071823752, 0.8888127607844434, 0.89839705073959, 0.9078549039318771, 0.917198089904905, 0.9264382355080696, 0.9355868393652487, 0.9446552861839217, 0.9536548609225298, 0.9625967628337065, 0.9714921194008536, 0.9803520001854058, 0.989187430602018, 0.9980094056388127, 1.0068289035397662, 1.0156568994662565, 1.0245043791547768, 1.0333823525878072, 1.0423018676948592, 1.0512740241007434, 1.060309986938166, 1.0694210007418454, 1.0786184034414394, 1.0879136404706926, 1.0973182790103668, 1.106844022382673, 1.1165027246151233, 1.1263064051919214, 1.136267264011252, 1.1463976965670806, 1.156710309374356, 1.1672179356568129, 1.1779336513168952, 1.1888707912076717, 1.200042965726998, 1.2114640777545698, 1.2231483399529484, 1.2351102924540838, 1.247364820953349, 1.2599271752336008, 1.272812988142317, 1.286038295045428, 1.2996195537820499, 1.313573665144952, 1.3279179939122454, 1.3426703904564625, 1.3578492129579225, 1.3734733502500214, 1.389562245324877, 1.4061359195285834, 1.423214997476176, 1.4408207327173213, 1.4589750341846612, 1.4777004934577342, 1.4970204128763913, 1.5169588345386993, 1.5375405702194134, 1.558791232246251, 1.580737265372389, 1.603405979684851, 1.6268255845897284, 1.6510252239165384, 1.6760350121853915, 1.7018860720821105, 1.7286105731879298, 1.7562417720119703, 1.7848140533763137, 1.814362973205168, 1.844925302771378, 1.8765390744553414, 1.9092436290732713, 1.943079664833701, 1.978089287983158, 2.0143160652040306, 2.0518050778298202, 2.0906029779452755, 2.1307580464411737, 2.1723202530960526, 2.2153413187595996, 2.2598747797151324, 2.3059760543012233, 2.3537025118754022, 2.4031135442057483, 2.4542706393791995, 2.507237458318574, 2.562079914003509, 2.6188662534939047, 2.677667142857939, 2.7385557551103608, 2.8016078612704556, 2.866901924653045, 2.934519198509812, 3.004543827142492, 3.07706295061374, 3.152166813185995, 3.229948875623246, 3.3105059314955096, 3.393938227630703, 3.4803495888638105, 3.5698475472386186, 3.662543475822755, 3.758552727302596, 3.8579947775304637, 3.9609933742028, 4.067676690854308, 4.178177486359665, 4.29263327014138, 4.4111864732892645, 4.5339846258045835, 4.661180540189353, 4.792932501609323])
push!(priceArray2,[-5.02456519965762e-19, 0.030035664151704392, 0.04931367570847236, 0.06192926497328943, 0.0701390582509208, 0.0753241926834038, 0.0783657680585271, 0.07984634778813686, 0.08016318582582459, 0.07959491287570516, 0.0783425576998747, 0.07655579029465868, 0.0743502417745312, 0.07181911722337664, 0.06904091579615239, 0.06608435166127354, 0.06301122535467606, 0.05987783077522901, 0.05673537410972079, 0.05362977802192497, 0.050601152193768595, 0.04768317633247442, 0.04490252633950883, 0.042278574516138735, 0.03982345019560307, 0.03754251182645846, 0.03543520397233871, 0.033496199229023096, 0.031716675823884266, 0.030085575726192484, 0.02859069030058666, 0.027219530277438078, 0.025959934715124804, 0.024800454817917757, 0.023730552484489565, 0.022740681037972593, 0.021822282794876094, 0.020967739758592934, 0.020170302555540698, 0.019424002067071386, 0.01872356815008211, 0.018064349301172913, 0.017442239899020235, 0.01685361063310051, 0.016295250394965176, 0.015764314006546543, 0.015258276561757694, 0.014774893754853287, 0.01431216751960204, 0.013868316307477285, 0.01344174937768664, 0.013031045256133519, 0.012634930988266646, 0.012252266029875263, 0.011882027617912386, 0.011523297976546169, 0.011175253120545778, 0.010837153090035643, 0.010508333504235207, 0.010188198333694555, 0.00987621375026989, 0.009571902844521037, 0.009274840956524177, 0.008984651396193335, 0.008700996889954103, 0.008423583543024226, 0.00815215397047297, 0.007886484128846787, 0.007626380355966645, 0.007371676604211487, 0.007122231853181743, 0.006877927686885399, 0.0066386660197726545, 0.006404366955780867, 0.006174966765224708, 0.0059504159657184415, 0.005730677495115807, 0.005515724966775894, 0.005305541001203584, 0.005100115635410322, 0.004899444824762028, 0.004703529069979023, 0.004512372209016597, 0.004325980377592686, 0.004144361046540193, 0.003967521961262687, 0.0037954698985928825, 0.0036282094309598454, 0.0034657419938605876, 0.0033080652458041637, 0.0031551724967061267, 0.0030070521976973976, 0.0028636875047898147, 0.0027250559188625708, 0.002591128999184658, 0.002461872145495223, 0.00233724444664807, 0.0022171986086913973, 0.002101681005672374, 0.0019906319192672, 0.001883985972813927, 0.0017816725655263894, 0.001683615979543994, 0.0015897351932573816, 0.0014999439861349715, 0.0014141515064901525, 0.0013322627550405226, 0.001254179018182419, 0.0011797982931682459, 0.001109015708652211, 0.001041723937211114, 0.0009778136009876205, 0.0009171736885320746, 0.0008596920237225229, 0.0008052558062612209, 0.0007537520900804273, 0.0007050678744626335, 0.0006590897484999882, 0.0006157035801381071, 0.0005747938918069541, 0.0005362414646871813, 0.0004999146627389096, 0.0004656341426252405, 0.00043303906644708305, 0.0004011694550557561, 0.0003674730638368857, 0.0003261829527387137, 0.00026695426711585777])
push!(B,0.7)
push!(spotArray2,[0.7, 0.7134409629910365, 0.726533277915649, 0.7392928738135603, 0.7517352749149067, 0.7638756195281181, 0.7757286784582592, 0.7873088729782387, 0.7986302923747541, 0.8097067110903184, 0.8205516054822238, 0.8311781702188352, 0.8415993343331603, 0.851827776953229, 0.8618759427284205, 0.8717560569705073, 0.8814801405278369, 0.8910600244107499, 0.9005073641860266, 0.9098336541578769, 0.919050241352727, 0.928168339324817, 0.9371990417994074, 0.9461533361701921, 0.9550421168673422, 0.9638761986124428, 0.9726663295764508, 0.9814232044566832, 0.9901574774887445, 0.9988797754092263, 1.0076007103849494, 1.0163308929244794, 1.0250809447876243, 1.0338615119086216, 1.042683277348738, 1.0515569742940396, 1.060493399114148, 1.0695034244978707, 1.078598012681685, 1.0877882287871738, 1.0970852542836382, 1.106500400592266, 1.1160451228484105, 1.1257310338387208, 1.1355699181300825, 1.1455737464075582, 1.1557546900387738, 1.1661251358824665, 1.1766977013592177, 1.1874852498027024, 1.198500906110134, 1.2097580727109454, 1.2212704458731372, 1.2330520323671288, 1.2451171665073895, 1.257480527592583, 1.2701571577654456, 1.2831624803141224, 1.2965123184372367, 1.3102229144955153, 1.324310949773399, 1.338793564774675, 1.3536883800768336, 1.3690135177695109, 1.3847876235031111, 1.4010298891744282, 1.4177600762768694, 1.4349985399436913, 1.4527662537134995, 1.4710848350481456, 1.4899765716340636, 1.5094644484990547, 1.5295721759775005, 1.5503242185580413, 1.5717458246488074, 1.5938630572964279, 1.6167028258961795, 1.6402929189318702, 1.6646620377852797, 1.689839831656299, 1.71585693363625, 1.7427449979782805, 1.7705367386101751, 1.7992659689364427, 1.8289676429781028, 1.8596778979002306, 1.8914340979789965, 1.9242748800616978, 1.958240200575088, 1.9933713841392056, 2.0297111738458335, 2.067303783262787, 2.106194950227274, 2.1464319924937936, 2.18806386530428, 2.2311412209505215, 2.275716470401333, 2.3218438470694647, 2.3695794727958215, 2.418981426131277, 2.4701098129991585, 2.52302683982439, 2.5777968892182295, 2.63448659831074, 2.693164939826242, 2.7539033060004305, 2.8167755954412317, 2.881858303039103, 2.9492306130361436, 3.018974495367258, 3.0911748053906085, 3.165919387128664, 3.2432991801454643, 3.3234083301901647, 3.406344303741427, 3.49220800659206, 3.5811039066181674, 3.6731401608821814, 3.76842874722443, 3.867085600503319, 3.9692307536499154, 4.0749884837085215, 4.184487463040966, 4.2978609158785215, 4.415246780411966, 4.536787876616971, 4.662632080019032, 4.792932501609323])
push!(priceArray2,[-3.7832275478413335e-19, 0.02509694457830772, 0.04176271388872619, 0.052931380943429175, 0.060342927209879424, 0.06510558128793388, 0.06794893412808735, 0.0693681075519787, 0.06970877828389656, 0.06921923021370709, 0.06808334208206601, 0.06644208263484445, 0.06440773169601861, 0.06207325315834418, 0.05951830130561237, 0.056812853047571735, 0.05401919117779098, 0.05119278631380814, 0.04838244933881975, 0.045630239919467803, 0.042971041198280296, 0.04043227121559056, 0.03803380453637446, 0.035788213102093167, 0.03370136898690152, 0.03177337726401859, 0.029999740551081972, 0.02837261958162394, 0.026882026680474003, 0.025516871075847988, 0.024265766553109986, 0.02311759896876709, 0.022061884204815585, 0.02108895429490791, 0.02019003624022014, 0.019357253764040513, 0.018583585520879388, 0.017862802530034184, 0.01718938959039003, 0.01655847212137488, 0.01596574355529576, 0.015407399009506518, 0.014880072505891783, 0.014380783571301643, 0.013906889355897468, 0.013456042716429332, 0.013026155725934872, 0.01261536801677653, 0.012222019359503324, 0.011844625914552456, 0.011481860676574772, 0.011132533479813689, 0.01079557654152535, 0.010470030937298418, 0.010155034793933579, 0.00984981298066084, 0.009553668134841457, 0.009265972896721122, 0.008986163234819732, 0.00871373271603871, 0.00844822753235122, 0.008189242077313143, 0.007936414899748341, 0.007689418140660098, 0.007447967385432841, 0.0072118121640486026, 0.006980732989453154, 0.006754538638902715, 0.006533063649682138, 0.006316166001127986, 0.006103724957244513, 0.005895639047357372, 0.005691824165626984, 0.005492211773492634, 0.005296747192029409, 0.0051053879736827025, 0.004918102345002359, 0.004734867714350795, 0.004555669242386749, 0.0043804984805713145, 0.00420935209563742, 0.004042230712110382, 0.003879137903398954, 0.0037200793159814635, 0.0035650618193683427, 0.0034140925286117777, 0.0032671776864364645, 0.0031243216372330066, 0.0029855260983889347, 0.00285078962269718, 0.0027201071057511997, 0.0025934693503911347, 0.002470862695896322, 0.002352268712541882, 0.0022376639588636295, 0.002127019797257182, 0.0020203022662944156, 0.0019174720208798947, 0.001818484376729201, 0.0017232895138856485, 0.0016318328425499668, 0.0015440553685460937, 0.0014598937846683569, 0.0013792803090058603, 0.0013021427586737374, 0.0012284050264206377, 0.0011579874906704009, 0.0010908073795724742, 0.0010267791275875772, 0.0009658147282662588, 0.0009078240801384253, 0.0008527153243317978, 0.000800395182969818, 0.0007507693257207348, 0.0007037427902998148, 0.000659220391975247, 0.0006171068846204702, 0.000577306686319096, 0.0005397234119521546, 0.000504259245259937, 0.00047081267565961, 0.00043927031781544634, 0.0004094745750214259, 0.00038110469605863467, 0.00035331560138366464, 0.00032390186780841625, 0.0002879745776410507, 0.00023692082446916322])
push!(B,0.72)
push!(spotArray2,[0.72, 0.732774245066986, 0.7452309883714019, 0.757385030275421, 0.7692508114894001, 0.7808424302294735, 0.7921736589682158, 0.803257960798275, 0.8141085054284197, 0.824738184831004, 0.8351596285594433, 0.8453852187538988, 0.8554271048530013, 0.8652972180290914, 0.87500728536413, 0.8845688437831202, 0.8939932537615959, 0.9032917128234638, 0.912475268845236, 0.9215548331824601, 0.9305411936339437, 0.9394450272591761, 0.9482769130641753, 0.9570473445708347, 0.9657667422847026, 0.974445466076007, 0.9830938274886382, 0.9917221019917122, 1.0003405411882729, 1.0089593849956373, 1.0175888738118573, 1.0262392606827513, 1.0349208234839646, 1.0436438771325303, 1.0524187858424408, 1.061255975438791, 1.0701659457451242, 1.0791592830586985, 1.0882466727284958, 1.0974389118509187, 1.1067469220982595, 1.116181762695181, 1.1257546435586312, 1.135476938616799, 1.1453601993229423, 1.1554161683801356, 1.1656567936932536, 1.1760942425647614, 1.1867409161511802, 1.1976094641974049, 1.208712800066379, 1.2200641160819843, 1.2316768992033782, 1.2435649470493952, 1.2557423842920588, 1.268223679438677, 1.2810236620224627, 1.294157540222105, 1.307640918931222, 1.3214898182991686, 1.3357206927652259, 1.350350450608789, 1.3653964740387792, 1.380876639846151, 1.3968093406440338, 1.413213506720742, 1.4301086285316176, 1.4475147798564323, 1.465452641649859, 1.483943526613353, 1.5030094045176388, 1.5226729283058837, 1.5429574610085819, 1.5638871035021158, 1.585486723143988, 1.6077819833187352, 1.6307993739296398, 1.6545662428724544, 1.6791108285285514, 1.704462293316087, 1.730650758339053, 1.7577073391753846, 1.785664182846639, 1.8145545060131791, 1.8444126344402316, 1.8752740437817246, 1.90717540173035, 1.9401546115839383, 1.9742508572799036, 2.0095046499512685, 2.04595787605959, 2.0836538471619557, 2.1226373513712122, 2.16295470657053, 2.204653815445566, 2.247784222399587, 2.292397172419183, 2.338545671960518, 2.3862845519284503, 2.435670532823355, 2.4867622921330526, 2.539620534049912, 2.5943080615959677, 2.6508898512417347, 2.709433130107394, 2.7700074558380554, 2.8326847992480273, 2.897539629832251, 2.9646490042465405, 3.0340926578617093, 3.1059530995004136, 3.180315709469221, 3.2572688410024235, 3.3369039252380994, 3.4193155798511548, 3.504601721472429, 3.5928636820274105, 3.684206329132809, 3.778738190694037, 3.8765715838515997, 3.9778227484296638, 4.082611985045301, 4.191063798042533, 4.303307043421001, 4.419475081934991, 4.539705937544779, 4.664142461408481, 4.792932501609322])
push!(priceArray2,[4.760326918781838e-19, 0.02079729073882271, 0.03502299134422277, 0.04475910584528149, 0.05132555308006528, 0.055597580177834116, 0.058170568217443284, 0.0594602686294638, 0.05976499011921838, 0.059305142232359345, 0.05824891955737137, 0.05672918251383015, 0.05485452774174266, 0.052716414956858935, 0.050393594292856066, 0.047954704283611975, 0.04545965798036325, 0.042960206479709176, 0.04050024373483073, 0.03811568838680983, 0.035834483593214575, 0.03367677086344132, 0.03165533869417565, 0.02977637814926128, 0.028040506391634513, 0.02644396043902949, 0.024979844305351828, 0.023639247047115237, 0.02241223085801407, 0.021288572846830084, 0.020258280822488112, 0.019311917166514782, 0.018440767057059427, 0.01763691139279698, 0.01689323049059988, 0.01620336867495421, 0.015561679655972365, 0.01496315816729788, 0.014403375059287144, 0.013878412907377779, 0.013384807241534732, 0.012919489837859878, 0.012479741273498234, 0.012063147932945316, 0.011667564288413809, 0.011291079999321735, 0.010931991319978225, 0.010588776295265521, 0.010260073250973374, 0.009944663117476268, 0.00964145115181455, 0.009349453905921226, 0.00906778694978643, 0.008795654152857987, 0.008532338320832925, 0.008277193032448338, 0.00802963555707566, 0.007789140747204272, 0.00755523578522129, 0.007327495632523003, 0.007105539008608941, 0.006889024745283584, 0.0066776430267876654, 0.0064711231764820265, 0.00626922570580206, 0.006071739613691989, 0.005878479917806478, 0.0056892853919558554, 0.005504016483071471, 0.0053225533824002545, 0.005144794228357163, 0.0049706534216500935, 0.004800060036484404, 0.004632956314597694, 0.004469296231437323, 0.004309044125953226, 0.004152173387432258, 0.003998665195196936, 0.0038485073112092486, 0.0037016929335020473, 0.0035582196301131856, 0.0034180883823579614, 0.0032813027549186825, 0.0031478681580434206, 0.003017791091327576, 0.0028910782553945953, 0.0027677355809442735, 0.0026477674070183688, 0.0025311759186753408, 0.002417960691295119, 0.0023081182730085555, 0.0022016418183936435, 0.0020985207773432436, 0.001998740639325428, 0.0019022827304719546, 0.0018091240596429083, 0.0017192372121136153, 0.0016325903001385366, 0.0015491470003758886, 0.0014688667226968363, 0.0013917049126364886, 0.001317613354808687, 0.001246540251968969, 0.0011784300879322295, 0.0011132236737514591, 0.0010508585395942246, 0.0009912692775009668, 0.0009343878422283174, 0.000880143845240025, 0.0008284648455972174, 0.0007792766351999894, 0.0007325035158015549, 0.0006880685709217188, 0.0006458939483355284, 0.0006059011755875519, 0.000568011487505067, 0.0005321460153307916, 0.0004982256096563467, 0.00046617032688175513, 0.0004358987327989639, 0.00040732581064189556, 0.0003803553746664877, 0.00035485090807404904, 0.0003305315312055111, 0.0003066649970879555, 0.00028137677504784953, 0.00025059272247247687, 0.0002072678727405607])
push!(B,0.74)
push!(spotArray2,[0.74, 0.7521296344649457, 0.7639720334630578, 0.7755409201233489, 0.7868497006251072, 0.7979114797331551, 0.8087390759838241, 0.8193450365392447, 0.8297416517271646, 0.8399409692831433, 0.8499548083116277, 0.8597947729820876, 0.8694722659760802, 0.8789985017008298, 0.888384519284631, 0.897641195369137, 0.9067792567133565, 0.9158092926239638, 0.9247417672263284, 0.9335870315904817, 0.9423553357260747, 0.9510568404602247, 0.9597016292120168, 0.9682997196773038, 0.9768610754373442, 0.9853956175047336, 0.9939132358200043, 1.0024238007122208, 1.0109371743368472, 1.0194632221041435, 1.028011824111335, 1.0365928865917988, 1.0452163533945393, 1.0538922175072518, 1.0626305326363292, 1.0714414248572304, 1.0803351043487108, 1.089321877224513, 1.0984121574762278, 1.1076164790411662, 1.1169455080092248, 1.1264100549828935, 1.1360210876047239, 1.1457897432667794, 1.155727342016793, 1.1658453996759883, 1.1761556411837666, 1.186670014184721, 1.1974007028737266, 1.2083601421151462, 1.2195610318525167, 1.2310163518254118, 1.242739376610536, 1.2547436910044811, 1.267043205765968, 1.2796521737358177, 1.2925852063533338, 1.305857290588231, 1.3194838063077352, 1.3334805440989759, 1.3478637235673308, 1.362650012131917, 1.3778565443400184, 1.3935009417228268, 1.4096013332155048, 1.4261763761652375, 1.4432452779516125, 1.4608278182443875, 1.478944371924433, 1.497615932694415, 1.5168641374065752, 1.5367112911358036, 1.5571803930270542, 1.5782951629470592, 1.600080068971226, 1.6225603557375656, 1.6457620737005167, 1.669712109318555, 1.6944382162105796, 1.7199690473171745, 1.7463341881040146, 1.7735641908458983, 1.8016906100311265, 1.8307460389272627, 1.8607641473506449, 1.891779720683413, 1.9238287001832732, 1.9569482246327028, 1.9911766733758642, 2.026553710793098, 2.0631203322645324, 2.100918911676074, 2.1399932505228314, 2.18038862866686, 2.2221518568080776, 2.265331330729117, 2.309977087377003, 2.35614086284663, 2.403876152333236, 2.4532382721233406, 2.504284423696, 2.5570737600086253, 2.6116674540442286, 2.668128769699479, 2.726523135095746, 2.7869182183980756, 2.849384006229957, 2.9139928847747485, 2.980819723657754, 3.0499419627061224, 3.1214397016871622, 3.1953957931290065, 3.2718959383312214, 3.3510287866766117, 3.4328860383592974, 3.517562550648116, 3.6051564478084916, 3.695769234810116, 3.789505914952274, 3.8864751115430485, 3.9867891937734514, 4.0905644069323355, 4.197921007112973, 4.308983400567415, 4.423880287870103, 4.542744813057805, 4.665714717918686, 4.792932501609323])
push!(priceArray2,[3.608220720167809e-19, 0.01705562894051348, 0.029017706925926777, 0.03734954319030117, 0.04303710822533952, 0.04676152393273508, 0.049002535106575026, 0.050106016929983274, 0.05032801202118207, 0.04986368174362543, 0.04886643247149395, 0.04746050869163864, 0.04574917625775681, 0.043819922753026014, 0.04174765855598779, 0.03959659951002991, 0.03742130328386208, 0.03526723676111025, 0.03317107063211202, 0.031160989926665893, 0.029257142197630433, 0.027472312278434018, 0.025812842514010198, 0.02427975108758979, 0.022869951988647545, 0.02157747270425321, 0.02039450469271766, 0.019312312327070424, 0.01832189967220362, 0.01741446746196415, 0.016581697199741736, 0.015815899398792913, 0.015110077592656538, 0.014457932699872373, 0.013853833375197838, 0.013292768839085348, 0.012770291606393935, 0.012282460701339612, 0.01182578588171528, 0.011397177094677802, 0.010993893786640887, 0.01061350412917046, 0.01025384732093484, 0.009913000439831548, 0.00958924946981312, 0.009281064072350733, 0.008987075660745894, 0.008706058356082288, 0.008436913239463495, 0.008178652274356064, 0.007930386660604081, 0.007691315917299916, 0.0074607183446437735, 0.007237942685795203, 0.007022400850718902, 0.0068135615965584095, 0.006610945073171564, 0.006414118132694191, 0.006222690275798119, 0.006036310086708182, 0.005854662018358379, 0.005677460280278793, 0.005504452580006784, 0.005335414737988642, 0.005170148285510212, 0.005008478271953971, 0.0048502512601872865, 0.00469533348669159, 0.004543609163586647, 0.004394978901734768, 0.004249358236791996, 0.004106676242907519, 0.003966874221473141, 0.0038299044547363114, 0.0036957290161431086, 0.0035643186310098884, 0.0034356515827997595, 0.003309712662631086, 0.0031864921640487433, 0.00306598493298887, 0.0029481894930645674, 0.002833107269925497, 0.002720741917725013, 0.0026110986961579426, 0.0025041837952536495, 0.002400003546326998, 0.0022985636162926393, 0.002199868374525606, 0.0021039204563865657, 0.0020107203814151396, 0.0019202662069600456, 0.001832553225566218, 0.0017475737089505722, 0.0016653166981478102, 0.0015857678373522167, 0.001508909248172821, 0.0014347194431412905, 0.0013631732858949168, 0.0012942420219827818, 0.0012278934156586118, 0.0011640919946094219, 0.0011027992979873644, 0.0010439739462923267, 0.0009875715289754316, 0.0009335446276575299, 0.0008818431303179449, 0.0008324145134276558, 0.0007852040859928598, 0.0007401552272488908, 0.000697209621756914, 0.0006563074900198621, 0.0006173878117363849, 0.0005803885412758856, 0.0005452468223678866, 0.0005118992175022022, 0.0004802819518558562, 0.00045033108802622996, 0.0004219824377085437, 0.0003951711020655076, 0.00036983069299593904, 0.000345891419701757, 0.0003232732024701649, 0.0003018598976103848, 0.00028141017356864535, 0.000261302169045639, 0.00023997556690061962, 0.00021410800101921457, 0.00017806469956963572])
push!(B,0.76)
push!(spotArray2,[0.76, 0.7715089864370408, 0.7827600796623843, 0.7937659785586769, 0.8045391052626572, 0.8150916191857712, 0.8254354307382568, 0.8355822147721828, 0.8455434237586206, 0.8553303007138179, 0.8649538918889645, 0.8744250592378745, 0.8837544926766545, 0.8929527221491973, 0.902030129512118, 0.9109969602525462, 0.9198633350520016, 0.9286392612094037, 0.9373346439361088, 0.9459592975357228, 0.9545229564813084, 0.9630352864024887, 0.9715058949948497, 0.9799443428639524, 0.9883601543161962, 0.9967628281087122, 1.005161848170419, 1.013566694306343, 1.021986852897283, 1.0304318276068984, 1.0389111501083022, 1.047434390842269, 1.0560111698191998, 1.0646511674770334, 1.0733641356073624, 1.082159908362084, 1.0910484133530098, 1.1000396828569612, 1.109143865138999, 1.1183712359065656, 1.1277322099074696, 1.1372373526848016, 1.1468973925020514, 1.1567232324518824, 1.1667259627622326, 1.1769168733136328, 1.1873074663818677, 1.197909469620362, 1.208734849296948, 1.2197958237999516, 1.231104877428841, 1.242674774485004, 1.2545185736785593, 1.2666496428674594, 1.2790816741455235, 1.2918286992964267, 1.304905105631095, 1.318325652226371, 1.3321054865832889, 1.3462601617237528, 1.3608056537449218, 1.3757583798511073, 1.3911352168835425, 1.406953520368934, 1.4232311441082948, 1.4399864603281711, 1.4572383804170048, 1.4750063762700367, 1.4933105022668447, 1.5121714179063162, 1.531610411124611, 1.5516494223224266, 1.5723110691286855, 1.5936186719286, 1.615596280184922, 1.6382686995820852, 1.6616615200238867, 1.685801144516292, 1.71071481896798, 1.7364306629422521, 1.762977701395016, 1.7903858974346707, 1.818686186140865, 1.8479105094802981, 1.8780918523589767, 1.9092642798516137, 1.9414629756501944, 1.9747242817750994, 2.0090857395936164, 2.0445861321921215, 2.081265528149773, 2.1191653267631088, 2.1583283047725983, 2.198798664643893, 2.2406220844582574, 2.283845769468501, 2.3285185053785997, 2.3746907134071362, 2.4224145071967174, 2.4717437516335945, 2.5227341236438745, 2.575443175034948, 2.6299303974530606, 2.6862572895303316, 2.744487426297026, 2.804686530937415, 2.866922548970207, 2.931265724937293, 2.9977886816873465, 3.0665665023437665, 3.137676815049502, 3.211199880584364, 3.2872186829537555, 3.365819023051057, 3.44708961549936, 3.5311221887818807, 3.61801158877406, 3.7078558857941846, 3.800756485293391, 3.896818242309961, 3.996149579817085, 4.098862611097702, 4.205073266284494, 4.3149014232079095, 4.428471042699856, 4.545910308505787, 4.667351771963135, 4.792932501609322])
push!(priceArray2,[-2.700065767287754e-19, 0.013801331335970259, 0.0236782217454112, 0.03064987273249758, 0.035442780475522045, 0.03858041786194696, 0.04044562364498568, 0.04132432409314001, 0.041435387777793056, 0.04095077977714363, 0.040009043803291934, 0.03872422497959657, 0.03719169561357029, 0.03549189257160543, 0.033692634449023026, 0.031850656585457424, 0.030012466293455508, 0.02821501669429746, 0.026486337825006107, 0.024846267775152416, 0.0233073568807093, 0.02187594868812804, 0.02055338233539529, 0.019337223540875338, 0.01822242482848187, 0.017202321349624742, 0.01626943293014244, 0.015416048812730326, 0.014634621742656154, 0.013918009009770508, 0.013259601738684249, 0.01265337797131833, 0.012093907339694336, 0.011576327619916175, 0.011096303858075968, 0.010649982648064141, 0.010233943919080245, 0.009845154034945142, 0.009480923522109512, 0.00913886321331012, 0.008816850132391885, 0.008512995365757165, 0.008225615741396353, 0.007953209015832529, 0.007694432218232874, 0.007448082788521934, 0.007213082161764773, 0.006988462006437546, 0.006773350660606073, 0.006566962870280089, 0.006368590324169699, 0.006177593377097748, 0.005993393810737401, 0.005815468514595694, 0.005643343997211937, 0.005476591649128242, 0.005314823670843741, 0.005157689556791475, 0.005004873008574628, 0.004856089157673631, 0.0047110805374847185, 0.004569617806837076, 0.004431496380904407, 0.004296534349426583, 0.0041645705778036645, 0.004035462973591795, 0.003909086898524352, 0.003785333706250737, 0.0036641093874717574, 0.0035453333063185013, 0.0034289370142125356, 0.0033148631297887227, 0.0032030642756041617, 0.0030935020642167835, 0.002986146127777617, 0.0028809731866363967, 0.0027779661539408754, 0.002677113275567962, 0.0025784073091535, 0.0024818447532634715, 0.002387425145391786, 0.002295150445555431, 0.0022050244953916837, 0.002117052491789842, 0.002031240389529122, 0.0019475942222493318, 0.0018661194656388713, 0.0017868205681350632, 0.0017097006078001283, 0.0016347609791308325, 0.0015620011113983488, 0.001491418224151804, 0.0014230071211222407, 0.0013567600217382308, 0.0012926664280897607, 0.0012307130245874885, 0.0011708836092889148, 0.0011131590625817014, 0.001057517371695394, 0.0010039337383776347, 0.0009523807719229742, 0.0009028286882556848, 0.0008552453724963257, 0.0008095962905446064, 0.0007658444945529789, 0.0007239508668721588, 0.0006838743483311867, 0.0006455721321548461, 0.0006089998520033109, 0.0005741117677784671, 0.0005408609479355843, 0.0005091994456845612, 0.0004790784670810153, 0.00045044853252092893, 0.0004232596391611755, 0.00039746142882786064, 0.0003730033235310517, 0.000349834498265363, 0.000327903519120454, 0.0003071575539655838, 0.000287540602357398, 0.0002689873979830883, 0.0002514011376111655, 0.00023457887789383748, 0.0002180042316226009, 0.00020040818063408046, 0.00017914522240759047, 0.00014981728107307929])
push!(B,0.78)
push!(spotArray2,[0.78, 0.7909143200364948, 0.8015991183084518, 0.8120661239488922, 0.8223268270111015, 0.8323924910817234, 0.8422741656452524, 0.8519826982134968, 0.8615287462333299, 0.8709227887857974, 0.8801751380894273, 0.8892959508203658, 0.8982952392617709, 0.9071828822946976, 0.9159686362425431, 0.9246621455809564, 0.9332729535249663, 0.9418105125049534, 0.9502841945429626, 0.9587033015407497, 0.9670770754908521, 0.975414708621897, 0.98372535348928, 0.9920181330222938, 1.0003021505387344, 1.00858649973798, 1.0168802746835104, 1.0251925797858257, 1.0335325397967254, 1.0419093098259147, 1.0503320853909375, 1.0588101125114653, 1.0673526978590253, 1.0759692189733088, 1.0846691345562731, 1.0934619948553395, 1.1023574521470845, 1.1113652713329318, 1.1204953406584781, 1.1297576825682172, 1.1391624647075818, 1.148720011084376, 1.1584408134018551, 1.1683355425758897, 1.1784150604488592, 1.1886904317131348, 1.1991729360572356, 1.2098740805479977, 1.2208056122623443, 1.2319795311825255, 1.2434081033689797, 1.2551038744252836, 1.2670796832699658, 1.2793486762303032, 1.2919243214735752, 1.3048204237916106, 1.3180511397548655, 1.3316309932526593, 1.3455748914366317, 1.3598981410849238, 1.3746164654050448, 1.3897460212938684, 1.4053034170737095, 1.4213057307239483, 1.4377705286282139, 1.4547158848577117, 1.4721604010118594, 1.4901232266380093, 1.5086240802526782, 1.5276832709873562, 1.547321720882658, 1.5675609878552876, 1.5884232893630326, 1.6099315267937604, 1.6321093106051936, 1.6549809862430598, 1.678571660866067, 1.7029072309070408, 1.7280144105004824, 1.753920760807749, 1.7806547202720484, 1.8082456358364634, 1.8367237951592728, 1.8661204598619296, 1.896467899846206, 1.927799428718157, 1.9601494403578124, 1.9935534466747182, 2.028048116590792, 2.0636713162932705, 2.100462150801951, 2.1384610068963434, 2.177709597449863, 2.2182510072197283, 2.260129740142838, 2.3033917681895284, 2.348084581828857, 2.394257242160803, 2.4419604347726067, 2.4912465253783833, 2.542169617303072, 2.5947856108738296, 2.649152264784077, 2.7053292594975327, 2.763378262761866, 2.823362997303861, 2.8853493107804224, 2.949405248062189, 3.0156011259291398, 3.0840096102601393, 3.154705795801208, 3.227767288600039, 3.3032742911972863, 3.381309690668113, 3.4619591496106747, 3.5453112001814016, 3.6314573412803064, 3.7204921389930052, 3.8125133303997245, 3.907621930865217, 4.0059223449274, 4.107522480906415, 4.212533869359933, 4.3210717855147305, 4.43325537580896, 4.549207788683971, 4.669056309769316, 4.792932501609322])
push!(priceArray2,[-1.7600352100222571e-19, 0.010964580965690324, 0.018928970801422522, 0.024597648265345873, 0.02850012248497111, 0.03103489712146436, 0.032505160534681415, 0.033145631781548224, 0.03314169016509756, 0.032642639562013895, 0.0317707533423245, 0.030627365848075003, 0.02929693036099744, 0.02784971743970503, 0.026343582235982006, 0.024825227312832314, 0.023331208982735, 0.02188888100461827, 0.02051739926752954, 0.019228839507682408, 0.018029415588979247, 0.016920737652709447, 0.015901034732881646, 0.014966219386131354, 0.014110806425707141, 0.013328609373467363, 0.012613238895584728, 0.01195843421087049, 0.01135826022861481, 0.010807216507829766, 0.010300281926857792, 0.00983291979392152, 0.009401060474398662, 0.009001067709201987, 0.008629702817343314, 0.008284085821013384, 0.00796165741422528, 0.007660144348880287, 0.0073775239277301076, 0.007111995809634615, 0.0068619555544044974, 0.006625971237199187, 0.006402762902849807, 0.0061911845849351495, 0.0059902086030367615, 0.005798911863140532, 0.005616464064590086, 0.005442116942261286, 0.005275195327846272, 0.0051150891949441665, 0.004961246696984152, 0.004813168075452183, 0.004670400343103464, 0.004532532667498885, 0.004399192387896807, 0.004270041590428605, 0.004144774148390495, 0.004023113121321411, 0.003904808414001002, 0.0037896339022629633, 0.0036773869992694297, 0.0035678863176588083, 0.0034609699174101916, 0.0033564937087451536, 0.0032543299955940896, 0.0031543661430283405, 0.0030565033519835724, 0.002960655525730584, 0.002866748214286859, 0.00277471762492291, 0.002684509688872774, 0.002596079176175508, 0.0025093888521798614, 0.0024244086706038713, 0.002341114999175535, 0.0022594898749301734, 0.002179520287590287, 0.0021011974918481754, 0.0020245163536983103, 0.001949474742073996, 0.001876072981146768, 0.0018043133711979455, 0.001734199757174984, 0.0016657370849565442, 0.0015989308869813295, 0.0015337867257400573, 0.001470309716369006, 0.001408504189228675, 0.0013483734189789678, 0.001289919374652656, 0.0012331424973542355, 0.001178041508192952, 0.0011246132469584578, 0.0010728525407144966, 0.001022752100442422, 0.0009743024434899595, 0.0009274918409162867, 0.0008823062938339433, 0.0008387295523221726, 0.000796743197259155, 0.0007563267877347091, 0.0007174580170163125, 0.0006801127686369021, 0.0006442650508325836, 0.00060988698993274, 0.000576949011687353, 0.0005454200219188894, 0.0005152675567049605, 0.0004864579270907788, 0.00045895636183754783, 0.00043272714746020783, 0.0004077337634538253, 0.0003839390102956104, 0.0003613051288611654, 0.0003397939128345479, 0.0003193668167172943, 0.00029998504441955946, 0.00028160954256090256, 0.0002642007412316278, 0.00024771785650319026, 0.0002321172519629275, 0.00021734720042653066, 0.00020332924266378355, 0.00018989771157168405, 0.00017663692741087827, 0.00016254730238726635, 0.0001455903241173608, 0.00012244658568932192])
push!(B,0.8)
push!(spotArray2,[0.8, 0.8103478112134227, 0.8204934520709496, 0.8304477378857921, 0.8402212799855534, 0.8498244970239666, 0.8592676260872409, 0.8685607336068566, 0.8777137260904403, 0.8867363606821608, 0.8956382555639025, 0.9044289002083042, 0.9131176654945933, 0.9217138136979981, 0.9302265083633887, 0.9386648240736692, 0.9470377561233377, 0.955354230107523, 0.9636231114367213, 0.9718532147873776, 0.9800533134983819, 0.9882321489235008, 0.996398439749712, 1.004560891291377, 1.012728204770157, 1.0209090865905666, 1.0291122576210534, 1.037346462490495, 1.0456204789100243, 1.053943127030123, 1.0623232788429529, 1.070769867639953, 1.0792918975347807, 1.0878984530617504, 1.09659870886, 1.1054019394537122, 1.1143175291388105, 1.1233549819866755, 1.13252393197554, 1.141834153260368, 1.1512955705921613, 1.1609182698978022, 1.1707125090317128, 1.1806887287107868, 1.1908575636442562, 1.201229853870355, 1.211816656311863, 1.2226292565628518, 1.233679180919194, 1.2449782086656638, 1.256538384632724, 1.2683720320363858, 1.2804917656148311, 1.2929105050757974, 1.305641488869063, 1.3186982882987133, 1.3320948219902295, 1.3458453707278282, 1.3599645926778603, 1.374467539014503, 1.3893696699644023, 1.4046868712873637, 1.4204354712106684, 1.4366322578350579, 1.4532944970309516, 1.4704399508439647, 1.4880868964293548, 1.5062541455355767, 1.5249610645577198, 1.5442275951821969, 1.5640742756447024, 1.584522262624092, 1.6055933537955287, 1.627310011066931, 1.6496953845235023, 1.6727733371058586, 1.6965684700480632, 1.721106149102687, 1.746412531580854, 1.7725145942360883, 1.7994401620216904, 1.827217937752307, 1.8558775327012986, 1.8854494981665302, 1.9159653580382408, 1.947457642403692, 1.9799599222244377, 2.013506845123163, 2.0481341723182593, 2.0838788167454956, 2.1207788824074205, 2.1588737049924616, 2.1982038938069968, 2.2388113750651177, 2.2807394365822224, 2.324032773920082, 2.368737538032576, 2.4149013844628757, 2.462573524144545, 2.5118047758606807, 2.5626476204170388, 2.6151562565868787, 2.669386658887184, 2.7253966372478255, 2.783245898637287, 2.8429961107106494, 2.904710967547677, 2.9684562575510776, 3.0342999335773366, 3.102312185374863, 3.1725655144066693, 3.245134811137371, 3.320097434866857, 3.3975332961957667, 3.4775249422106596, 3.5601576444796983, 3.645519489952651, 3.733701474862087, 3.8247976017259173, 3.9189049795546267, 4.016123927370044, 4.116558081146035, 4.220314504285044, 4.3275038017483265, 4.438240237961489, 4.552641858621036, 4.670830616531795, 4.79293250160932])
push!(priceArray2,[1.5470875170558832e-19, 0.008489975708934526, 0.014713822704998318, 0.019155797819569546, 0.0221987690153573, 0.024143727863793315, 0.025228650307899347, 0.025643817267872477, 0.02554328827337737, 0.025053167909560893, 0.024277432409599765, 0.023301980942496272, 0.022197472400956417, 0.021021312573179674, 0.01981918248825662, 0.018626326854325075, 0.017468770432763552, 0.01636455654892619, 0.015325034375211155, 0.014356167143257902, 0.013459799700233473, 0.012634814800203578, 0.011878120226449672, 0.011185438772846083, 0.010551897099606104, 0.00997243743549571, 0.009442086672614132, 0.008956117758770032, 0.00851013764247758, 0.008100125014964729, 0.007722436598614891, 0.007373794905454145, 0.007051264463262474, 0.006752224217960435, 0.006474337827980736, 0.006215524479827939, 0.00597393073861044, 0.005747905035319723, 0.0055359743762890415, 0.005336823486461955, 0.005149276305166123, 0.004972279673249243, 0.004804889009363282, 0.00464625576138654, 0.0044956166813226, 0.004352283670205119, 0.004215635667449608, 0.004085111304323212, 0.00396020246994398, 0.0038404486741738034, 0.0037254321133873714, 0.003614773365126799, 0.003508127651692248, 0.00340518161628487, 0.0033056505473712068, 0.0032092759731753, 0.0031158235403713564, 0.0030250810999226524, 0.0029368562906957755, 0.002850976346292982, 0.002767286045911209, 0.002685646289979657, 0.002605932802898255, 0.002528034950872287, 0.0024518546612546667, 0.0023773054297855194, 0.0023043114030284954, 0.002232806524688703, 0.0021627337360647487, 0.0020940442224596633, 0.0020266966988498935, 0.001960656729430312, 0.0018958960767854892, 0.0018323920773797204, 0.00177012704085799, 0.0017090876714759844, 0.001649264511227456, 0.0015906514065802478, 0.0015332450047022756, 0.0014770442896423193, 0.0014220501693626999, 0.0013682651125425046, 0.0013156928070204243, 0.0012643377897419974, 0.0012142050226195047, 0.001165299465872143, 0.0011176257396901996, 0.0010711878837946424, 0.0010259891473590505, 0.0009820317971427498, 0.0009393169479401233, 0.0008978444168479227, 0.0008576126014465837, 0.000818618381018591, 0.0007808570392645938, 0.0007443222067415827, 0.0007090058222127875, 0.0006748981156516368, 0.0006419876223653879, 0.0006102612427899426, 0.0005797043509800734, 0.0005503009131255513, 0.0005220335369346635, 0.000494883426886748, 0.00046883037109310465, 0.00044385286849232136, 0.0004199282677844958, 0.0003970328802764412, 0.00037514208759163384, 0.0003542304476478314, 0.0003342717985112579, 0.00031523935856755914, 0.0002971058208240983, 0.00027984343901302157, 0.00026342410383810977, 0.00024781940824199233, 0.0002330006938232752, 0.00021893903900076852, 0.0002056050796344404, 0.0001929684513466584, 0.00018099634791556618, 0.00016964913775060727, 0.00015886533363088998, 0.00014851436688584701, 0.00013827355504278095, 0.00012738423881851208, 0.00011433403433837873, 9.671252044237698e-5])
push!(B,0.82)
push!(spotArray2,[0.82, 0.8298117735757824, 0.839447656988369, 0.8489176089590036, 0.8582314167181697, 0.8673987061207291, 0.8764289515942785, 0.8853314859310057, 0.8941155099331652, 0.9027901019221409, 0.9113642271209255, 0.919846746919711, 0.9282464280341693, 0.9365719515658845, 0.9448319219743038, 0.9530348759694787, 0.9611892913347859, 0.9693035956887481, 0.9773861751950093, 0.9854453832294653, 0.9934895490135095, 1.0015269862223135, 1.0095660015770413, 1.0176149034298771, 1.025682010350737, 1.0337756597245427, 1.0419042163679393, 1.0500760811743648, 1.0582996997964038, 1.0665835713744019, 1.0749362573203576, 1.0833663901661739, 1.0918826824854126, 1.1004939358977706, 1.1092090501655862, 1.1180370323917777, 1.1269870063287164, 1.1360682218076597, 1.145290064298485, 1.154662064609608, 1.1641939087381088, 1.1738954478802455, 1.1837767086127022, 1.193847903255093, 1.2041194404244324, 1.2146019357924795, 1.2253062230570744, 1.2362433651388043, 1.247424665614574, 1.258861680399894, 1.270566229691962, 1.2825504101858805, 1.2948266075766366, 1.3074075093597628, 1.3203061179439117, 1.3335357640888927, 1.347110120683062, 1.361043216874304, 1.3753494525692067, 1.3900436133154188, 1.4051408855825689, 1.4206568724575368, 1.4366076097703033, 1.4530095826670382, 1.4698797426475614, 1.4872355250847793, 1.5050948672442104, 1.5234762268222113, 1.542398601022079, 1.561881546187726, 1.5819451980152395, 1.6026102923631942, 1.6238981866832405, 1.6458308820931098, 1.668431046114851, 1.691722036101799, 1.7157279233784868, 1.7404735181184454, 1.765984394985615, 1.7922869195658508, 1.8194082756158565, 1.8473764931576957, 1.876220477447926, 1.9059700388512861, 1.936655923649821, 1.9683098458192756, 2.0009645198056076, 2.0346536943354923, 2.0694121872957503, 2.105275921717767, 2.1422819629040752, 2.180468556735491, 2.219875169198362, 2.2605425271728183, 2.3025126605241515, 2.3458289455408377, 2.390536149764089, 2.4366804782552816, 2.4843096213490474, 2.5334728039414194, 2.58422083636394, 2.636606166896331, 2.6906829359719793, 2.746507032132286, 2.8041361497876687, 2.863629848844957, 2.925049616262772, 2.9884589295985347, 3.0539233226127456, 3.1215104529983755, 3.1912901723053433, 3.263334598132348, 3.337718188660675, 3.41451781960701, 3.4938128636747647, 3.5756852725860693, 3.660219661779179, 3.747503397858841, 3.837626688890006, 3.9306826776282056, 4.026767537782918, 4.125980573413472, 4.228424321560141, 4.334204658216583, 4.44343090775304, 4.556215955903505, 4.672676366433531, 4.792932501609323])
push!(priceArray2,[-2.730354491565806e-20, 0.006332667039685765, 0.01099130027627596, 0.014306658809935969, 0.0165527517325522, 0.01795375714868907, 0.01869284692912959, 0.01892018781303811, 0.01875935757432438, 0.018312211018768967, 0.017662477619946316, 0.016878416432063663, 0.016014858714673744, 0.015114876027669392, 0.014211260413136542, 0.01332793567529106, 0.012481353795077232, 0.01168187523353046, 0.010935093693633593, 0.010243054387095782, 0.009605292944775008, 0.00901970236233778, 0.008483192826272743, 0.007992170651866614, 0.007542867553197285, 0.007131551301507084, 0.006754655568475731, 0.006408850622586995, 0.006091075706896632, 0.005798547671968811, 0.005528752721057255, 0.005279432075841177, 0.005048562490252644, 0.004834335102797315, 0.00463513502679716, 0.004449519189062739, 0.0042761984334686665, 0.004114020223545793, 0.003961953066365281, 0.003819072625236821, 0.0036845494208319492, 0.0035576379850567616, 0.003437667320470089, 0.0033240328555790935, 0.003216188353794746, 0.0031136398667588987, 0.003015940078039245, 0.002922683339671416, 0.00283350131844356, 0.0027480591841054025, 0.002666052284947927, 0.0025872032641409437, 0.002511259570229456, 0.0024379913078700533, 0.002367189365641321, 0.0022986637551388393, 0.002232242105683408, 0.0021677674675118362, 0.002105098793687801, 0.002044108944363978, 0.0019846835785019175, 0.0019267201454273711, 0.0018701269664052544, 0.0018148223954564905, 0.0017607340487561735, 0.0017077980927113245, 0.0016559585819003197, 0.0016051668392617726, 0.0015553808721257697, 0.001506564818814845, 0.00145868842156587, 0.0014117265224150364, 0.0013656585794374861, 0.0013204682013548206, 0.0012761426990849118, 0.0012326726534906795, 0.0011900514997485146, 0.0011482751308871354, 0.001107341526246918, 0.001067250413293195, 0.0010280029686689934, 0.0009896015507314976, 0.000952049434259676, 0.0009153505122931879, 0.0008795089673195613, 0.00084452896896577, 0.0008104144481968678, 0.0007771689267573391, 0.0007447953613938866, 0.0007132960030624738, 0.0006826722734925748, 0.0006529246597081492, 0.0006240526262908308, 0.0005960545446201777, 0.0005689276378909992, 0.0005426679405593793, 0.0005172702715103255, 0.000492728222582366, 0.00046903416858075104, 0.0004461793086319698, 0.00042415374199089025, 0.00040294655406542744, 0.0003825458580816821, 0.000362938768023767, 0.0003441113834315966, 0.00032604887139261736, 0.00030873556871777744, 0.00029215506317318017, 0.0002762902698638143, 0.00026112350610512454, 0.00024663656464253894, 0.00023281078415080667, 0.0002196271153143759, 0.00020706618020898215, 0.0001951083221858431, 0.00018373364278409, 0.00017292201892981423, 0.00016265307863960502, 0.0001529060676962846, 0.0001436594340953854, 0.00013488966474593198, 0.000126567753355414, 0.00011864764008157661, 0.00011103109079555809, 0.00010347925251659516, 9.544356061662497e-5, 8.585598426507465e-5, 7.305193013582653e-5])
push!(B,0.84)
push!(spotArray2,[0.84, 0.8493086221189583, 0.8584665107698173, 0.8674828263991545, 0.8763665878409831, 0.885126681338104, 0.8937718694308316, 0.9023107997219812, 0.910752013526887, 0.9191039544171035, 0.9273749766663353, 0.9355733536070443, 0.9437072859060931, 0.9517849097677024, 0.9598143050719273, 0.9678035034567951, 0.9757604963521861, 0.9836932429734954, 0.9916096782830719, 0.9995177209273969, 1.0074252811579418, 1.01534026874363, 1.0232706008828143, 1.0312242101226872, 1.0392090522940434, 1.0472331144693323, 1.055304422951963, 1.063431051304848, 1.0716211284262211, 1.0798828466808077, 1.0882244700944763, 1.096654342620573, 1.1051808964862058, 1.1138126606268264, 1.1225582692175486, 1.1314264703097343, 1.140426134581489, 1.1495662642108169, 1.158856001880314, 1.168304639922404, 1.1779216296142665, 1.1877165906317533, 1.1976993206717523, 1.20787980525262, 1.2182682277024899, 1.228874979345444, 1.23971066989574, 1.250786138070489, 1.262112462431399, 1.273700972466433, 1.2855632599224593, 1.2977111904002376, 1.3101569152233332, 1.322912883592835, 1.3359918550400334, 1.3494069121895162, 1.3631714738454472, 1.377299308414117, 1.3918045476761973, 1.4067017009224672, 1.4220056694671572, 1.4377317615534249, 1.4538957076658758, 1.4705136762654387, 1.4876022899623458, 1.5051786421433833, 1.523260314070055, 1.5418653924647516, 1.5610124876025284, 1.5807207519265756, 1.6010098992060116, 1.6219002242551603, 1.6434126232340318, 1.6655686145503203, 1.6883903603838217, 1.7119006888548034, 1.7361231168585023, 1.761081873588585, 1.7868019247731133, 1.8133089976472443, 1.8406296066876486, 1.8687910801343977, 1.897821587326832, 1.9277501668807733, 1.9586067557352487, 1.9904222190977898, 2.0232283813182663, 2.05705805772212, 2.0919450874348575, 2.127924367230632, 2.165031886438754, 2.203304762943085, 2.2427812803102807, 2.283500926084054, 2.3255044312837483, 2.3688338111467315, 2.413532407155371, 2.459644930390622, 2.5072175062555937, 2.5562977206138413, 2.606934667388516, 2.6591789976700024, 2.7130829703811568, 2.7687005045508273, 2.826087233247943, 2.8853005592301195, 2.9463997123624535, 3.0094458088639273, 3.0745019124407054, 3.141633097367446, 3.210906513579758, 3.282391453842894, 3.3561594230638634, 3.4322842098163173, 3.510841960149732, 3.591911253756721, 3.6755731825746882, 3.7619114319003986, 3.8510123640986653, 3.94296510498883, 4.037861632995472, 4.135796871152535, 4.23686878205286, 4.341178465838153, 4.4488302613273625, 4.559931850384629, 4.6745943656312505, 4.79293250160932])
push!(priceArray2,[-1.1721191899010704e-20, 0.004473343258646916, 0.007759085580859213, 0.010079029280869243, 0.011624538341034857, 0.012557509188365895, 0.01301364214480708, 0.013106048957344898, 0.012928430463838415, 0.01255774081630128, 0.01205637035481445, 0.011473989910660959, 0.010849190549905541, 0.010211018218128006, 0.0095804560652756, 0.008971861616129607, 0.008394331413804301, 0.007852952523478835, 0.007349879348055734, 0.006885240095290192, 0.006457841247410681, 0.006065692273896874, 0.005706378409315492, 0.005377311957536564, 0.00507589629950565, 0.00479962598424253, 0.004546143611247967, 0.004313268157046929, 0.004099004569450936, 0.003901542353787835, 0.0037192472875758873, 0.0035506494928288163, 0.00339442993670515, 0.0032494061657787347, 0.00311451874722431, 0.002988818342304294, 0.002871453780883956, 0.0027616612327500114, 0.0026587544819128145, 0.0025621162555555235, 0.0024711905306477825, 0.0023854757299675913, 0.002304518810520709, 0.0022279097628527906, 0.0021552770681193215, 0.002086283600241169, 0.0020206230218066177, 0.0019580166188165322, 0.0018982105290745426, 0.0018409733264372332, 0.0017860939260799299, 0.0017333797734674808, 0.001682655273647102, 0.0016337604125514237, 0.0015865495236789822, 0.0015408901639741243, 0.0014966612776726135, 0.001453753967006374, 0.0014120697475914438, 0.001371519734714795, 0.0013320239034977056, 0.0012935104152484124, 0.0012559150019186515, 0.0012191804008224907, 0.0011832558323928836, 0.001148096514553532, 0.0011136632081513807, 0.0010799217887565537, 0.0010468428409493287, 0.0010144012719532638, 0.0009825759421252483, 0.0009513493103689015, 0.0009207070929986271, 0.0008906379349706875, 0.0008611330927917473, 0.0008321861289912377, 0.0008037926190947454, 0.0007759498738127083, 0.0007486566813145473, 0.0007219130751023565, 0.0006957201284455986, 0.0006700797642593428, 0.0006449945566150673, 0.000620467506255842, 0.0005965018076448819, 0.0005731006531404938, 0.0005502670906145087, 0.0005280039054043366, 0.0005063135116030116, 0.00048519785492758894, 0.00046465832804985124, 0.0004446956985696942, 0.0004253100493570971, 0.00040650073063552443, 0.00038826632292665846, 0.000370604609894521, 0.00035351256051582825, 0.000336986321420718, 0.0003210212230289367, 0.00030561180566556737, 0.00029075186841457187, 0.00027643452701606793, 0.00026265224621809376, 0.0002493968261266468, 0.00023665938815482646, 0.0002244304214545656, 0.00021269985285247482, 0.00020145710176916836, 0.00019069113074489827, 0.0001803904947106111, 0.00017054338904546972, 0.00016113769573998393, 0.00015216102649157325, 0.0001436007609650995, 0.00013544407760915857, 0.00012767797311247946, 0.00012028926387149754, 0.00011326455481553892, 0.00010659013512052627, 0.00010025168121734112, 9.423339822684876e-5, 8.851538088254514e-5, 8.306527512207674e-5, 7.781390583886271e-5, 7.259601519610956e-5, 6.70412090733316e-5, 6.0445828622446606e-5, 5.173841605727247e-5])
push!(B,0.86)
push!(spotArray2,[0.86, 0.868840814890236, 0.8775548773783156, 0.886150608745234, 0.8946363159166412, 0.903020199490666, 0.9113103616629841, 0.9195148140567899, 0.9276414854652398, 0.9356982295138477, 0.9436928322502387, 0.9516330196685949, 0.959526465176067, 0.9673807970083632, 0.9752036056016873, 0.9830024509281446, 0.9907848698017094, 0.9985583831618111, 1.0063305033415793, 1.014108741327773, 1.0219006140194065, 1.0297136514920906, 1.037555404275108, 1.045433450648253, 1.053355403965491, 1.0613289200125142, 1.0693617044053016, 1.0774615200368367, 1.0856361945791766, 1.0938936280481244, 1.1022418004378145, 1.1106887794325884, 1.1192427282036164, 1.1279119132977953, 1.1367047126265497, 1.1456296235622574, 1.1546952711501195, 1.163910416443416, 1.1732839649701974, 1.1828249753396, 1.1925426679960958, 1.2024464341301426, 1.2125458447538418, 1.2228506599503763, 1.2333708383061681, 1.2441165465348685, 1.255098169302482, 1.2663263192631227, 1.2778118473150943, 1.2895658530872136, 1.3015996956655056, 1.313925004570637, 1.3265536909967026, 1.3394979593222152, 1.3527703189044347, 1.3663835961684254, 1.3803509470025301, 1.3946858694722373, 1.4094022168647276, 1.4245142110767095, 1.4400364563584773, 1.4559839534274772, 1.47237211396502, 1.489216775510149, 1.5065342167650577, 1.5243411733268486, 1.5426548538608347, 1.5614929567310156, 1.5808736871038005, 1.6008157745415001, 1.6213384911026032, 1.6424616699663166, 1.664205724599375, 1.6865916684836408, 1.7096411354235586, 1.7333764004530934, 1.7578204013623497, 1.7829967608646804, 1.8089298094257094, 1.8356446087763212, 1.8631669761323524, 1.8915235091443792, 1.9207416116017209, 1.9508495199154976, 1.9818763304063334, 2.0138520274230753, 2.046807512319709, 2.080774633318465, 2.1157862162879804, 2.1518760964662587, 2.1890791511590897, 2.2274313334455194, 2.266969706922953, 2.3077324815254583, 2.3497590504498973, 2.393090028225565, 2.437767289964113, 2.4838340118277173, 2.5313347127545742, 2.5803152974820565, 2.6308231009091125, 2.682906933840769, 2.736617130158959, 2.7920055954652456, 2.849125857242461, 2.908033116583735, 2.968784301538901, 3.0314381221298334, 3.09605512708789, 3.1626977623682775, 3.231430431497909, 3.3023195578150464, 3.375433648660902, 3.4508433615852194, 3.5286215726298176, 3.6088434467560924, 3.6915865104845267, 3.77693072681642, 3.8649585725102327, 3.9557551177872314, 4.049408108543455, 4.146008051147465, 4.245648299905808, 4.348425147280738, 4.454437916947375, 4.56378905978023, 4.676584252861858, 4.792932501609322])
push!(priceArray2,[1.3885057256141252e-20, 0.002917390031770194, 0.005048342194457321, 0.006534777144983986, 0.007503539379466988, 0.008064905509998704, 0.008312898355726938, 0.008326404884872225, 0.008170516594772127, 0.007897930098507035, 0.0075503567633507275, 0.0071599356805920914, 0.00675064530286487, 0.006339692450839175, 0.005938840021544483, 0.005555629040214618, 0.005194441481961717, 0.004857397941203743, 0.0045450665047175294, 0.0042570004406143796, 0.0039921292146317565, 0.003749031748557424, 0.0035261216045749427, 0.0033217676566167933, 0.003134369724443534, 0.002962403425565207, 0.002804444639610416, 0.0026591799225305374, 0.002525409107319752, 0.0024020423907258484, 0.002288094388362179, 0.002182676903465187, 0.0020849900213476508, 0.0019943142712880936, 0.001910002659449959, 0.0018314732025715378, 0.0017582020637723057, 0.0016897173228720454, 0.0016255933714457585, 0.0015654458997709589, 0.0015089275740045906, 0.0014557237522196903, 0.001405549150835375, 0.0013581447386015928, 0.0013132749926028121, 0.0012707254782339372, 0.0012303007208536597, 0.0011918223419599382, 0.001155127435694308, 0.0011200671610876096, 0.0010865055219355334, 0.001054318301821021, 0.00102339212026485, 0.0009936235801321846, 0.0009649184857723418, 0.0009371906290879259, 0.0009103621755193769, 0.0008843625111737647, 0.0008591276892983358, 0.0008345999267553416, 0.0008107271451450785, 0.0007874625511697787, 0.0007647642510673682, 0.0007425948943586319, 0.0007209213426610395, 0.0006997143598694226, 0.000678948320549501, 0.0006586009339104756, 0.0006386529812021124, 0.0006190880648105926, 0.0005998923676998125, 0.000581054422158831, 0.0005625648870776304, 0.000544416333208856, 0.0005266030361533418, 0.000509120777273887, 0.0004919666535871652, 0.0004751388989572666, 0.00045863672002617253, 0.00044246014949684274, 0.0004266099143543114, 0.00041108730810121536, 0.0003958940522167301, 0.00038103214400574733, 0.0003665037109903884, 0.0003523108965479787, 0.0003384557742746455, 0.000324940271612498, 0.00031176610021467954, 0.0002989346941289126, 0.00028644715617945105, 0.00027430421252291735, 0.00026250617510847185, 0.00025105291159666993, 0.00023994382215456782, 0.00022917782249808391, 0.0002187533327584386, 0.00020866827251202834, 0.0001989200638539458, 0.00018950564597835812, 0.00018042150331712382, 0.00017166370058961487, 0.00016322790547124638, 0.00015510938419703235, 0.00014730299118643038, 0.0001398031908741974, 0.0001326041008227306, 0.00012569952592781473, 0.0001190829889941887, 0.00011274776032264687, 0.00010668688648158814, 0.00010089321786629958, 9.535943431276207e-5, 9.007806759451735e-5, 8.504151892681055e-5, 8.024206838884123e-5, 7.567187089886194e-5, 7.132292822570416e-5, 6.718701173977055e-5, 6.325546131958299e-5, 5.951861462138158e-5, 5.5964050494498524e-5, 5.25711704298084e-5, 4.929596472487439e-5, 4.6035332180021305e-5, 4.256400485261191e-5, 3.846475002389029e-5, 3.311732179031847e-5])
push!(B,0.88)
push!(spotArray2,[0.88, 0.8884107681502529, 0.8967175390918259, 0.9049280544407943, 0.913049966106432, 0.9210908434225215, 0.9290581802017034, 0.9369594017194449, 0.944801871634133, 0.952592898849743, 0.9603397443274784, 0.9680496278527287, 0.9757297347636542, 0.983387222647666, 0.9910292280120437, 0.9986628729349076, 1.006295271702743, 1.0139335374406622, 1.0215847887415863, 1.029256156300521, 1.0369547895601108, 1.044687863373667, 1.0524625846918765, 1.0602861992794241, 1.0681659984677891, 1.0761093259505066, 1.0841235846272301, 1.0922162435029725, 1.1003948446489524, 1.1086670102315386, 1.1170404496158375, 1.1255229665505497, 1.134122466440786, 1.1428469637156256, 1.1517045892972786, 1.1607035981788192, 1.1698523771175449, 1.179159452451138, 1.1886334980439097, 1.1982833433705347, 1.2081179817448076, 1.2181465787010937, 1.2283784805362814, 1.2388232230202, 1.2494905402826193, 1.2603903738851125, 1.2715328820862408, 1.282928449308689, 1.2945876958171796, 1.306521487616184, 1.3187409465766518, 1.3312574608012024, 1.3440826952374312, 1.357228602549228, 1.370707434256236, 1.384531752151834, 1.3987144400102829, 1.413268715593945, 1.4282081429717723, 1.4435466451605359, 1.459298517100586, 1.4754784389782274, 1.4921014899071354, 1.509183161981553, 1.5267393747143747, 1.5447864898735681, 1.5633413267307619, 1.5824211777362094, 1.602043824634742, 1.6222275550377219, 1.64299117946645, 1.6643540488829047, 1.6863360727241514, 1.7089577374572325, 1.7322401256718294, 1.756204935728488, 1.7808745019807215, 1.8062718155898376, 1.8324205459518832, 1.8593450627566852, 1.8870704586995366, 1.915622572866698, 1.9450280148165127, 1.9753141893785686, 2.0065093221940313, 2.0386424860209376, 2.071743627828979, 2.105843596709014, 2.140974172623327, 2.1771680960234283, 2.214459098362994, 2.2528819335343817, 2.2924724102580325, 2.333267425454923, 2.3753049986331938, 2.418624307320977, 2.46326572357847, 2.509270851623259, 2.556682566603979, 2.6055450545584318, 2.655903853593406, 2.7078058963245786, 2.7612995536160483, 2.8164346796602695, 2.8732626584403897, 2.931836451618301, 2.9922106478930326, 3.054441513875481, 3.1185870465269017, 3.1847070272100138, 3.2528630774031173, 3.3231187161291285, 3.3955394191530486, 3.4701926800030645, 3.5471480728721203, 3.626477317458601, 3.7082543458065484, 3.792555371207704, 3.879458959229599, 3.9690461009358744, 4.061400288367086, 4.156607592352323, 4.254756742724169, 4.355939211011768, 4.460249295689039, 4.567784210057512, 4.678644172845685, 4.792932501609322])
push!(priceArray2,[-9.81761594514316e-21, 0.0016816507798731485, 0.0029002221854050486, 0.0037376799736308767, 0.004269748414954316, 0.004563697002612139, 0.004677429262246514, 0.004659397407454349, 0.004549041413915467, 0.004377564455831776, 0.004168916087461366, 0.003940882402317868, 0.0037061967276061977, 0.0034735986230987025, 0.0032487787882057167, 0.003035186901589034, 0.00283468050176847, 0.0026480251847109657, 0.0024752653550168444, 0.002315989789467123, 0.0021695169713610986, 0.002035020898050603, 0.0019116144377282623, 0.0017984029876338746, 0.0016945178772250799, 0.0015991354306125949, 0.0015114874684078986, 0.0014308655651286484, 0.0013566214865526507, 0.0012881655074404462, 0.0012249626022758152, 0.0011665287937214198, 0.0011124269026967586, 0.001062262297380663, 0.0010156788059990497, 0.0009723548848833486, 0.0009320000853277878, 0.0008943518330599543, 0.0008591725212744793, 0.0008262468823567912, 0.0007953796530101267, 0.0007663934881559204, 0.0007391271058514494, 0.0007134336415099451, 0.0006891791917953144, 0.000666241531004913, 0.0006445089847018183, 0.0006238794459893927, 0.000604259518775016, 0.000585563770253833, 0.0005677140733522781, 0.0005506390209352579, 0.0005342733974992962, 0.0005185576983505631, 0.000503437645303968, 0.0004888638442691907, 0.00047479137443842223, 0.0004611794453483759, 0.00044799108299203416, 0.0004351928439579358, 0.0004227545550843882, 0.00041064907547442843, 0.0003988520777179424, 0.0003873418454591527, 0.00037609908482432743, 0.00036510674760299814, 0.00035434986442941385, 0.00034381538653124356, 0.0003334920348966636, 0.0003233701559574555, 0.0003134415830942424, 0.00030369950344265973, 0.0002941383296251786, 0.00028475357617774603, 0.00027554174063803996, 0.00026650018960090624, 0.00025762705060299176, 0.0002489211113771163, 0.00024038172826171806, 0.00023200874422685583, 0.000223802413259101, 0.00021576332363489484, 0.00020789231381705865, 0.0002001903850861317, 0.00019265862496726054, 0.00018529814913429424, 0.00017811005442884315, 0.00017109537577937746, 0.0001642550473649541, 0.0001575898684061906, 0.000151100473642362, 0.00014478730840726317, 0.000138650608109197, 0.00013269038183662442, 0.00012690639974899747, 0.00012129818389152222, 0.00011586500216355374, 0.00011060586551020486, 0.00010551952911775431, 0.00010060449921267657, 9.585904667827067e-5, 9.128122497538665e-5, 8.68688835759425e-5, 8.261966831079865e-5, 7.853101540355511e-5, 7.46001588899562e-5, 7.082415263789169e-5, 6.719988873613805e-5, 6.37241132637716e-5, 6.03934413566043e-5, 5.7204371766726355e-5, 5.415330072906852e-5, 5.123653474308054e-5, 4.8450301620674636e-5, 4.579075871751873e-5, 4.325399647253762e-5, 4.0836033864678206e-5, 3.8532799185184396e-5, 3.634008114686307e-5, 3.425340810013887e-5, 3.226771690075134e-5, 3.0376355392356615e-5, 2.856808308324941e-5, 2.6818931441919603e-5, 2.5073885384821135e-5, 2.321587940239516e-5, 2.1033756746060384e-5, 1.8221462531149266e-5])
push!(B,0.9)
push!(spotArray2,[0.9, 0.9080207427556151, 0.9159589715218017, 0.9238218076316096, 0.9316163047838297, 0.9393494553708222, 0.9470281967513448, 0.9546594174740123, 0.9622499634569688, 0.9698066441293156, 0.9773362385398054, 0.9848455014382832, 0.9923411693353272, 0.999829966545529, 1.0073186112198322, 1.014813821372341, 1.0223223209070078, 1.029850845649602, 1.0374061493903766, 1.0449950099428478, 1.0526242352241282, 1.0603006693622647, 1.06803119883606, 1.0758227586528886, 1.0836823385700456, 1.0916169893652137, 1.0996338291616676, 1.1077400498138976, 1.1159429233593747, 1.1242498085422468, 1.1326681574148212, 1.1412055220227513, 1.1498695611799277, 1.1586680473391489, 1.1676088735647396, 1.1767000606133657, 1.1859497641294026, 1.1953662819613107, 1.204958061605582, 1.2147337077849312, 1.2247019901675391, 1.2348718512342622, 1.2452524143008752, 1.2558529917025374, 1.2666830931478268, 1.2777524342498388, 1.2890709452419982, 1.3006487798864064, 1.312496324582716, 1.3246242076857002, 1.337043309039883, 1.3497647697397723, 1.3628000021244677, 1.376160700015591, 1.3898588492077402, 1.4039067382208645, 1.418316969324218, 1.43310246984177, 1.4482765037492258, 1.4638526835730492, 1.4798449826021736, 1.4962677474233461, 1.5131357107913583, 1.5304640048457019, 1.5482681746855145, 1.566564192314984, 1.5853684709717304, 1.6046978798510136, 1.6245697592389776, 1.6450019360685073, 1.6660127399116544, 1.6876210194229757, 1.7098461592485388, 1.7327080974157583, 1.7562273432196727, 1.7804249956216982, 1.8053227621773678, 1.8309429785100364, 1.8573086283480245, 1.8844433641431704, 1.9123715282892875, 1.9411181749595678, 1.9707090925825166, 2.0011708269765847, 2.032530705164245, 2.0648168598868857, 2.0980582548425106, 2.132284710668884, 2.1675269316954253, 2.2038165334878643, 2.241186071210358, 2.2796690688305157, 2.319300049193531, 2.360114564992399, 2.4021492306620034, 2.4454417552256906, 2.4900309761237827, 2.5359568940543946, 2.583260708857794, 2.6319848564765174, 2.6821730470243654, 2.7338703039984624, 2.787123004669546, 2.841978921686712, 2.89848726593394, 2.956698730676858, 3.0166655370393407, 3.0784414808507354, 3.14208198090575, 3.2076441286802906, 3.2751867395478476, 3.3447704055423886, 3.416457549715069, 3.4903124821335454, 3.5664014575741136, 3.6447927349584455, 3.7255566385882046, 3.808765621232521, 3.894494329124905, 3.9828196689278914, 4.0738208767254935, 4.167579589105387, 4.264179916394548, 4.363708518114088, 4.466254680720933, 4.571910397706123, 4.680770452121573, 4.792932501609323])
push!(priceArray2,[6.698251336814615e-21, 0.0008234961228405945, 0.00141384678482797, 0.001812680591724323, 0.00205928779034653, 0.0021888710410374727, 0.002231620771461441, 0.0022124415873769586, 0.0021511497022180678, 0.002062976671492949, 0.001959238728910696, 0.0018480594217146476, 0.0017350621366644058, 0.001623984017205707, 0.0015171840740071333, 0.001416043548407909, 0.0013212687949761238, 0.0012331128431645533, 0.0011515339029146289, 0.0010763061164820974, 0.0010070956472288865, 0.0009435120305626199, 0.0008851422175382649, 0.0008315722844600635, 0.0007824011824773862, 0.0007372487383821864, 0.000695759891977579, 0.0006576065691206039, 0.0006224875662066124, 0.0005901278899514469, 0.000560277347218695, 0.0005327088311198072, 0.0005072164836989773, 0.000483613849453623, 0.0004617320890605095, 0.00044141829307789047, 0.0004225339163627485, 0.00040495337258647783, 0.0003885626571795521, 0.0003732582087415809, 0.00035894584924654836, 0.0003455398357414879, 0.00033296201523371367, 0.0003211410747750953, 0.0003100118791563992, 0.00029951488867834735, 0.0002895956489184126, 0.0002802043433899614, 0.00027129539905774855, 0.00026282713461720886, 0.0002547614427861682, 0.0002470635003849863, 0.00023970138616281265, 0.00023264609659720464, 0.00022587118762695365, 0.00021935257230796328, 0.0002130683380128865, 0.00020699858119161492, 0.00020112525766402035, 0.00019543204648281122, 0.0001899042255407627, 0.00018452855727089048, 0.00017929318297854778, 0.00017418752453674045, 0.0001692021923614644, 0.0001643288987573496, 0.0001595603758824229, 0.00015489029772275718, 0.00015031320559234318, 0.00014582443678037415, 0.0001414200560576573, 0.00013709678982841547, 0.00013285196278054427, 0.00012868343696397178, 0.00012458955334610256, 0.0001205690760971141, 0.00011662114015129411, 0.00011274520283602198, 0.00010894100016892554, 0.00010520850729897599, 0.00010154790062506135, 9.795951806420544e-5, 9.444381636820356e-5, 9.10013297953739e-5, 8.763263635465249e-5, 8.433833192981568e-5, 8.111900765441324e-5, 7.797522922771684e-5, 7.490751843419144e-5, 7.191633694816176e-5, 6.900207241620354e-5, 6.616502674705986e-5, 6.340540649771881e-5, 6.07233152135326e-5, 5.8118747556550294e-5, 5.559158504703241e-5, 5.314159327468737e-5, 5.0768420562523845e-5, 4.8471598344097566e-5, 4.625054387113698e-5, 4.410456583866272e-5, 4.2032872237478445e-5, 4.0034577174677243e-5, 3.81087025208668e-5, 3.6254175413896084e-5, 3.446982985680191e-5, 3.2754415929904605e-5, 3.1106608610151286e-5, 2.952501496004873e-5, 2.800818088737584e-5, 2.6554597634853877e-5, 2.516270792713977e-5, 2.3830911593378e-5, 2.2557570353453676e-5, 2.134101123585685e-5, 2.0179527684467097e-5, 1.9071376607602584e-5, 1.8014767903198996e-5, 1.70078386117352e-5, 1.6048590161485405e-5, 1.5134720427529125e-5, 1.4263132730092722e-5, 1.3428506506531027e-5, 1.2619551866909319e-5, 1.1810901064993079e-5, 1.0949996608758648e-5, 9.944520889322098e-6, 8.66398028979629e-6])

schemes=["MCS","CS"]#,"SC2B","RKL2"] #"LS","ADE"
for (i,barrierLevel) in enumerate(B)
    spotArray = spotArray2[i]
    priceArray = priceArray2[i]
for method in schemes
    if method == "LS" || method=="RKC2" || method == "RKL" || method == "RKL2" || method=="HBDF2" || method == "SC2A" || method=="SC2B"
        damping = "None"
    else
        damping = "DO"
    end
for N in Ns
    price(isCall, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, barrierLevel, N, M, L, damping = damping, method = method)
end
end
end

#case 2 of Foulon(2010)
isCall=true
kappa = 3.0
theta = 0.12
sigma = 0.04
rho = 0.6
r = 0.01
q = 0.04
v0=theta
T=1.0
K=100.0
spotArray=[100.0]
priceArray=[12.025300295511904]
X = [1.0]
A = X
leftSlope = 1.0
rightSlope = 1.0
C = 0.0 * X
B = [1.0 for i = 1:length(C)]
cFunc = CollocationFunction(A, B, C, X, leftSlope, rightSlope, T)
#Idea: add damping to RKL so that img region is wider. Is it better to add damping vs simply increase s? - increase s does not seem to work well on M=4096, L=32, Ns=[56,64]
Ns = [1024, 768,512, 384, 256, 192, 128, 96, 64, 56, 48, 32, 24, 16, 12, 8 ,6,4] #timesteps
Ns = reverse(Ns)
damping = "None"
M = 128
L = 32
schemes = ["LS"]
barrierLevel = 0.0
for method in schemes
    if method == "LS" || method=="RKC2" || method == "RKL" || method == "RKL2" || method=="HBDF2" || method == "SC2A" || method=="SC2B"
        damping = "None"
    else
        damping = "DO"
    end
for N in Ns
    price(isCall, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, barrierLevel, N, M, L, damping = damping, method = "LS",lsSolver="IRDS")
end
end

#caseI
kappa = 1.5
theta = 0.04
sigma = 0.3
rho=-0.9
r=0.025
q=0.0
T=1.0
K=100.0
spotArray=[100.0]
