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

function makeJacobians(ti::Real, cFunc::CollocationFunction{T}, S::Vector{T}, M::Int) where T
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
    @inbounds Threads.@threads  for j = 1:L
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
                @inbounds  for j = 2:L - 1
                    jm = (j - 1) * M
                  @inbounds   @simd for i = 2:M - 1
                        index = i + jm

                Y0ij =  Y0[index] - a * (A2ij[index] * F[index] + A2iju[index] * F[index + M] + A2ijl[index] * F[index - M])
                Y1[index] = Y0ij
            end
        end
    end

#
    function explicitStep(a::Real, rij::Vector{T}, A1ilj::Vector{T}, A1ij::Vector{T}, A1iuj::Vector{T}, A2ijl::Vector{T}, A2ij::Vector{T}, A2iju::Vector{T}, F::Vector{T}, Y0::Vector{T}, Y1::Vector{T}, M::Int, L::Int) where T
        explicitStep1(a, A1ilj, A1ij, A1iuj, F, Y0, Y1, M, L)
        explicitStep2(a, A2ijl, A2ij, A2iju, F, Y1, Y1, M, L)
        @inbounds Threads.@threads for j = 2:L - 1
            jm = (j - 1) * M
            @inbounds @simd for i = 2:M - 1
                index = i + jm
                    Y0ij = a * rij[index] * (F[index + 1 +  M] - F[index + 1 - M] + F[index - 1 - M] - F[index - 1 +  M])
                    Y1[index] -= Y0ij
                end
            end
        end

function RKLStep(s::Int, a::Vector{T}, b::Vector{T}, w0::Real, w1::Real, rij::Vector{T}, A1ilj::Vector{T}, A1ij::Vector{T}, A1iuj::Vector{T}, A2ijl::Vector{T}, A2ij::Vector{T}, A2iju::Vector{T}, F::Vector{T}, Y::Vector{Vector{T}}, useDirichlet::Bool, lbValue::Real, M::Int, L::Int) where T
    mu1b = b[1] * w1
    explicitStep(mu1b, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Y[1], M, L)
    updatePayoffExplicitTrans(Y[1], useDirichlet, lbValue, M, L)
    MY0 = (Y[1] - F) / mu1b
    for j = 2:s
        muu = (2 * j - 1) * b[j] / (j * b[j - 1])
        muj = muu * w0
        mujb = muu * w1
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

function enforceLowerBound(F::Vector{T}, lowerBound::Vector{T},M::Int, L::Int) where {T}
    if length(lowerBound) > 0
        @. F = max(F,lowerBound)
    end
end

function RKLStep(s::Int, a::Vector{T}, b::Vector{T}, w0::Real, w1::Real, rij::Vector{T}, A1ilj::Vector{T}, A1ij::Vector{T}, A1iuj::Vector{T}, A2ijl::Vector{T}, A2ij::Vector{T}, A2iju::Vector{T}, F::Vector{T}, Yjm2::Vector{T}, Yjm::Vector{T}, Yj::Vector{T}, useDirichlet::Bool, lbValue::Real, lowerBound::Vector{T}, M::Int, L::Int) where T
    mu1b = b[1] * w1
    explicitStep(mu1b, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Yjm, M, L)
    updatePayoffExplicitTrans(Yjm, useDirichlet, lbValue, M, L)
    MY0 = (Yjm - F) / mu1b
    enforceLowerBound(Yjm, lowerBound,M,L)
    Yjm2 .= F
    for j = 2:s
        muu = (2 * j - 1) * b[j] / (j * b[j - 1])
        muj = muu * w0
        mujb = muu * w1
        gammajb = -a[j - 1] * mujb
        nuj = - 1.0 * b[2] / (2.0 * b[1]) #b0 = b[1]
        if j > 2
            nuj = -(j - 1) * b[j] / (j * b[j - 2])
        end
       @. Yj = muj * Yjm + nuj * Yjm2 + (1 - nuj - muj) * F + gammajb * MY0 # + mujb*MYjm
       explicitStep(mujb, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, Yjm, Yj, Yj, M, L)
       updatePayoffExplicitTrans(Yj, useDirichlet, lbValue, M, L)
       enforceLowerBound(Yj, lowerBound,M,L)
       Yjm2, Yjm = Yjm, Yjm2
       Yjm, Yj = Yj, Yjm
    end
    return Yjm
end


function RKCStep(s::Int, a::Vector{T}, b::Vector{T}, w0::Real, w1::Real, rij::Vector{T}, A1ilj::Vector{T}, A1ij::Vector{T}, A1iuj::Vector{T}, A2ijl::Vector{T}, A2ij::Vector{T}, A2iju::Vector{T}, F::Vector{T}, Yjm2::Vector{T}, Yjm::Vector{T}, Yj::Vector{T}, useDirichlet::Bool, lbValue::Real, lowerBound::Vector{T}, M::Int, L::Int) where T
    mu1b = b[1] * w1
    explicitStep(mu1b, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Yjm, M, L)
    updatePayoffExplicitTrans(Yjm, useDirichlet, lbValue, M, L)
    MY0 = (Yjm - F) / mu1b
    enforceLowerBound(Yjm, lowerBound,M,L)
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
       enforceLowerBound(Yj, lowerBound,M,L)
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
function legPoly(s::Int, w0::Real)
	tjm = 1.0
	tj = w0
    if s == 1
        return tj, 1.0, 0.0
    end
    dtjm = 0.0
	dtj = 1.0
	d2tjm = 0.0
	d2tj = 0.0

	for j = 2:s
        onej = 1.0 / j
        tjp = (2-onej)*w0*tj - (1-onej)*tjm
        dtjp = (2-onej)*(tj+w0*dtj) - (1-onej)*dtjm
        d2tjp = (2-onej)*(dtj*2+w0*d2tj) - (1-onej)*d2tjm
        tjm = tj
        dtjm = dtj
        d2tjm = d2tj
        tj = tjp
        dtj = dtjp
        d2tj = d2tjp
	end
	return tj, dtj, d2tj
end
function computeRKCStages(dtexplicit,dt,ep)
    # delta = 1 + 4 * (2 + 4 * dt / dtexplicit)
    # s = ceil(Int, (-1 + sqrt(delta)) / 2)
    # if s % 2 == 0
    #     s += 1
    # end
    # s+= Int(floor(s/5))
    #dtexplicit=2/lambdamax. we want beta = (w0+1)*tw0p2/tw0p > lambdamax*dt => dtexplicit*beta < 2dt.
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

function computeRKLStages(dtexplicit,dt,ep)
    s = 1
    betaFunc = function (s::Int)
    w0 = 1 + ep/s^2
    _,tw0p, tw0p2 = legPoly(s, w0)
    beta = (w0+1)*tw0p2/tw0p
    return beta - 2*dt/(dtexplicit)
end
while s < 10000 && betaFunc(s) < 0
    s += 1
end
    #s += Int(ceil(s/10))
    return s
end

function price(isCall, isEuropean, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L; damping = "None", method = "RKL", useVLinear = false, sDisc = "Sinh", useExponentialFitting = true, smoothing = "Kreiss", lambdaS = 0.25 , lambdaV = 2.0, epsilon = 1e-3, Xdev=4, Smax=0, rklStages = 0, epsilonRKL=0, printGamma=false, lsSolver = "SOR")


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
    Xspan = Xdev * sqrt(theta * T)
    logK = log(K)
    Xmin = logK - Xspan + (r - q) * T - 0.5 * v0 * T
    if B != 0
        Xmin = log(B)
    end
    Xmax = logK + Xspan + (r - q) * T - 0.5 * v0 * T
    Smin = exp(Xmin)
    if Smax == 0
    Smax = exp(Xmax)
end
    X = collect(range(Xmin, stop = Xmax, length = M))
    hm = X[2] - X[1]
    Sscale = K * lambdaS
    if sDisc == "Exp"
    S = exp.(X)
    J = exp.(X)
    Jm = @. exp(X - hm / 2)
    elseif sDisc == "Sinh"
        u = collect(range(0.0, stop = 1.0, length = M))
        Smin = B
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
        S = collect(range(B, stop = Smax, length = M))
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
    lowerBoundA = zeros()
    if !isEuropean
        lowerBoundA = zeros(M*L)
        lowerBoundA .= F
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
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
            lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
            updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
            F = (I + A0 + A1 + A2) \ F
            ti -= a * dt
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
            lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
            updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
            F = (I + A0 + A1 + A2) \ F
            N -= 1
        elseif damping == "EU2"
                a = 0.25
                ti -= a * dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
                updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
                F = (I + A0 + A1 + A2) \ F
                ti -= a * dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
                updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
                F = (I + A0 + A1 + A2) \ F
                ti -= a * dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
                updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
                F = (I + A0 + A1 + A2) \ F
                ti -= a * dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                #explicit
            explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
            ti -= a * dt
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
            rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            solveTrapezoidal2(1, 1, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0, F, Y1, M, L)
            solveTrapezoidal1(1, 1, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1, F, Y2, M, L)
            F = Y2
            lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
            updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew
            explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
            ti -= a * dt
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
            Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S,M)
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
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
                updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
                F = (I + A0 + A1 + A2) \ F
            end
        elseif method == "Explicit"
            Y0 = zeros(M * L)
            for n = 1:N
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            for n = 1:N
                explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
                ti -= dt
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti)
                updatePayoffExplicitTrans(Y0, useDirichlet, lbValue, M, L)
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
            # Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
            # rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            # dtexplicit = dt / maximum(A1ij + A2ij)
            # println("Nmin=",T/dtexplicit)
            for n = 1:N
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                explicitStep(0.5, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Y0, M, L)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti - 0.5 * dt)
                updatePayoffExplicitTrans(Y0, useDirichlet, lbValue, M, L)
                Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S,M)
                rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                explicitStep(1.0, rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew, Y0, F, Y1, M, L)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti - dt)
                updatePayoffExplicitTrans(Y1, useDirichlet, lbValue, M, L)
                F = copy(Y1)
                ti -= dt
            end
        elseif method == "RKL"
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
                    Sc, Jc, Jch, Jct = makeJacobians(tj, cFunc, S,M)
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
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
            rij0, A1ilj0, A1ij0, A1iuj0, A2ijl0, A2ij0, A2iju0 = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            rij = copy(rij0); A1ilj = copy(A1ilj0); A1ij= copy(A1ij0); A1iuj = copy(A1iuj0); A2ijl=copy(A2ijl0); A2ij = copy(A2ij0); A2iju = copy(A2iju0)
            Sc, Jc, Jch, Jct = makeJacobians(ti-dt, cFunc, S,M)
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
                Sc, Jc, Jch, Jct = makeJacobians(ti-dt, cFunc, S,M)
                rij1, A1ilj1, A1ij1, A1iuj1, A2ijl1, A2ij1, A2iju1 = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                end
            end
        elseif method == "RKL2"
            Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S,M)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            dtexplicit = dt / max(maximum(A1ij + A2ij))
            if true || (sDisc == "Sinh" && lambdaS >= 1)
                dtexplicit /= 2 #lambdaS
            else
                dtexplicit *= 0.9 #don't be too close to boundary
            end
            s = 0.0
            delta = 1 + 4 * (2 + 4 * dt / dtexplicit)
            s = ceil(Int, (-1 + sqrt(delta)) / 2)
            if s % 2 == 0
                s += 1
            end
            if epsilonRKL > 0
                s = computeRKLStages(dtexplicit, dt, epsilonRKL)
            end
            if rklStages > 0
                s = rklStages
            end
            # println(maximum(A1ij + A2ij), " ", 2 * minimum(A1ilj), " ", 2 * minimum(A2ijl), " ", maximum(A1ij + A2ij + A1iuj + A2iju + A1ilj + A2ijl), " DTE ", dtexplicit, " NE ", T / dtexplicit, " s ", s)
            # println("s=",s)
            a = zeros(s)
            b = zeros(s)
            w0 = 1.0
            w1 = 0.0
            if epsilonRKL == 0
            w1 = 4 / (s^2 + s - 2)
            b[1] = 1.0 / 3
            b[2] = 1.0 / 3
            a[1] = 1.0 - b[1]
            a[2] = 1.0 - b[2]
            for i = 3:s
                b[i] = (i^2 + i - 2.0) / (2 * i * (i + 1.0))
                a[i] = 1.0 - b[i]
            end
        else
            w0 = 1 + epsilonRKL/s^2
            _,tw0p, tw0p2 = legPoly(s, w0)
            w1 = tw0p/tw0p2
            b = zeros(s)
            for jj = 2:s
                _, tw0p, tw0p2 = legPoly(jj, w0)
                b[jj] = tw0p2 / tw0p^2
            end
            b[1] = b[2]
            a = zeros(s)
            for jj = 2:s
                tw0, _, _ = legPoly(jj-1, w0)
                a[jj-1] = (1 - b[jj-1]*tw0)
            end

        end
            Y0 = zeros(L * M)
            Y1 = zeros(L * M)
            Y2 = zeros(L * M)
            for n = 1:N
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti - dt * 0.5)
              #  RKLStep(s, a, b, w1, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y, useDirichlet, lbValue, M, L)
              F .= RKLStep(s, a, b, w0, w1, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, Y1, Y2, useDirichlet, lbValue, lowerBoundA, M, L)
              ti -= dt
                if n < N
                    Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S,M)
                    rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                end
            end
        elseif method == "RKC2"
            Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S,M)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            dtexplicit = dt / max(maximum(A1ij + A2ij))
            if sDisc == "Linear" || (sDisc == "Sinh" && lambdaS >= 1)
                dtexplicit /= 2 #lambdaS
            else

            end
            s = computeRKCStages(dtexplicit, dt, epsilonRKL)

            #s= Int(floor(sqrt(1+6*dt/dtexplicit)))  #Foulon rule as dtExplicit=lambda/2
            if rklStages > 0
                s = rklStages
            end
            #println(maximum(A1ij + A2ij), " ", 2 * minimum(A1ilj), " ", 2 * minimum(A2ijl), " ", maximum(A1ij + A2ij + A1iuj + A2iju + A1ilj + A2ijl), " DTE ", dtexplicit, " NE ", T / dtexplicit, " s ", s)
            # println("s=",s)
            w0 = 1 + epsilonRKL/s^2
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
              F .= RKCStep(s, a, b, w0, w1, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, Y1, Y2, useDirichlet, lbValue, lowerBoundA, M, L)
              ti -= dt
                if n < N
                    Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S,M)
                    rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                end
            end
        elseif method == "ADE0"
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            A0llr, A0uur, A0ulr, A0lur, A2dr, A2dlr, A2dur, A1dr, A1dlr, A1dur = makeSparseSystemLU(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
            # eigenvals, phi = Arpack.eigs(Ar,nev=2)
            # println("largest eigenvalue", eigenvals)

            for n = 1:N
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
            Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S,M)
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
                Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S,M)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                A0ll, A0uu, A0ul, A0lu, A2d, A2dl, A2du, A1d, A1dl, A1du = makeSparseSystemLU(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
            end
        elseif method == "LS"
            a = 1 - sqrt(2) / 2
            F1 = zeros(L * M)
            F2 = zeros(L * M)
            nIter = 0.0
            # Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
            # rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            # A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
            for n = 1:N
                ti1 = ti - a * dt
                Sc, Jc, Jch, Jct = makeJacobians(ti1, cFunc, S,M)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)

                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti1)
                updatePayoffExplicitTrans(F, useDirichlet, lbValue, M, L)
				if lsSolver == "GS"
					F1 = copy(F); nIter+=solveGS(1e-16, 5000, 1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F1, useDirichlet, lbValue, M, L)
	                ti1 -= a * dt
	                Sc, Jc, Jch, Jct = makeJacobians(ti1, cFunc, S,M)
	                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
					lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti1)
	                updatePayoffExplicitTrans(F1, useDirichlet, lbValue, M, L)
					F2 .= F1; nIter+=solveGS(1e-16, 5000, 1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F1, F2, useDirichlet, lbValue, M, L)
                elseif lsSolver == "Jacobi"
                    F1 = copy(F); nIter+=solveJacobi(1e-16, 5000, 1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F1, useDirichlet, lbValue, M, L)
                    ti1 -= a * dt
                    Sc, Jc, Jch, Jct = makeJacobians(ti1, cFunc, S,M)
                    rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                    lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti1)
                    updatePayoffExplicitTrans(F1, useDirichlet, lbValue, M, L)
                    F2 .= F1; nIter+=solveSOR(1e-16, 5000, 1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F1, F2, useDirichlet, lbValue, M, L)
				elseif lsSolver == "SOR"
					F1 = copy(F); nIter+=solveSOR(1e-16, 5000, 1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F1, useDirichlet, lbValue, M, L)
	                ti1 -= a * dt
	                Sc, Jc, Jch, Jct = makeJacobians(ti1, cFunc, S,M)
	                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
					lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti1)
	                updatePayoffExplicitTrans(F1, useDirichlet, lbValue, M, L)
					F2 .= F1; nIter+=solveSOR(1e-16, 5000, 1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F1, F2, useDirichlet, lbValue, M, L)
				elseif lsSolver == "ASOR"
					A0, A1, A2 = makeSparseSystem(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, L, M)
					F1 = copy(F); asor!(F1, I + A0 + A1 + A2,F,1.2,maxiter=5000,tolerance=1e-8)
					ti1 -= a * dt
					Sc, Jc, Jch, Jct = makeJacobians(ti1, cFunc, S,M)
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
					Sc, Jc, Jch, Jct = makeJacobians(ti1, cFunc, S,M)
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
					Sc, Jc, Jch, Jct = makeJacobians(ti1, cFunc, S,M)
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
					Sc, Jc, Jch, Jct = makeJacobians(ti1, cFunc, S,M)
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
                Sc, Jc, Jch, Jct = makeJacobians(ti - 2 * b * dt, cFunc, S,M)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                implicitStep1(b, A1ilj, A1ij, A1iuj, useDirichlet, F, Y1t, M, L)
                Sc, Jc, Jch, Jct = makeJacobians(ti - 4 * b * dt, cFunc, S,M)
                rijt, A1iljt, A1ijt, A1iujt, A2ijlt, A2ijt, A2ijut = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                implicitStep1(b, A1iljt, A1ijt, A1iujt, useDirichlet, Y1t, Y1, M, L)
                Y1 = (1 + sqrt(2)) * Y1 - sqrt(2) * Y1t
                implicitStep2(b, A2ijl, A2ij, A2iju, useDirichlet, Y1, Y2t, M, L)
                implicitStep2(b, A2ijlt, A2ijt, A2ijut, useDirichlet, Y2t, Y2, M, L)
                Y2 = (1 + sqrt(2)) * Y2 - sqrt(2) * Y2t

                Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * dt, cFunc, S,M)
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
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, a * dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
        #expl/2
                Y0 = copy(F)
                for j = 2:L - 1, i = 2:M - 1
                    index = i + (j - 1) * M
                    Y0[index] -= rij[index] * (F[i + 1 + (j) * M] - F[i + 1 + (j - 2) * M] + F[i - 1 + (j - 2) * M] - F[i - 1 + (j) * M])
                end
        #imp = imp1 imp2 imp2 imp1
                Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * b * dt, cFunc, S,M)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                implicitStep1(b, A1ilj, A1ij, A1iuj, useDirichlet, Y0, Y1t, M, L)
                Sc, Jc, Jch, Jct = makeJacobians(ti - b * dt, cFunc, S)
                rijt, A1iljt, A1ijt, A1iujt, A2ijlt, A2ijt, A2ijut = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                implicitStep1(b, A1iljt, A1ijt, A1iujt, useDirichlet, Y1t, Y1, M, L)
                Y1 = (1 + sqrt(2)) * Y1 - sqrt(2) * Y1t
                Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * b * dt, cFunc, S,M)
                rijt, A1iljt, A1ijt, A1iujt, A2ijlt, A2ijt, A2ijut = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                implicitStep2(b, A2ijl, A2ij, A2iju, useDirichlet, Y1, Y2t, M, L)
                Sc, Jc, Jch, Jct = makeJacobians(ti - b * dt, cFunc, S)
                rijt, A1iljt, A1ijt, A1iujt, A2ijlt, A2ijt, A2ijut = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                implicitStep2(b, A2ijlt, A2ijt, A2ijut, useDirichlet, Y2t, Y2, M, L)
                Y2 = (1 + sqrt(2)) * Y2 - sqrt(2) * Y2t
                Sc, Jc, Jch, Jct = makeJacobians(ti - 0.5 * b * dt - 0.5 * dt, cFunc, S,M)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold,  dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                Sc, Jc, Jch, Jct = makeJacobians(ti - b * dt - 0.5 * dt, cFunc, S,M)
                rijt, A1iljt, A1ijt, A1iujt, A2ijlt, A2ijt, A2ijut = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                implicitStep2(b, A2ijl, A2ij, A2iju, useDirichlet, Y2, Y1t, M, L)
                implicitStep2(b, A2ijl, A2ij, A2iju, useDirichlet, Y1t, Y1, M, L)
                Y1 = (1 + sqrt(2)) * Y1 - sqrt(2) * Y1t
                implicitStep1(b, A1ilj, A1ij, A1iuj, useDirichlet, Y1, Y2t, M, L)
                implicitStep1(b, A1ilj, A1ij, A1iuj, useDirichlet, Y2t, Y2, M, L)
                Y2 = (1 + sqrt(2)) * Y2 - sqrt(2) * Y2t
        #expl/2
                Sc, Jc, Jch, Jct = makeJacobians(ti - .5 * dt, cFunc, S,M)
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
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                #explicit
                explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                explicitStep2(a, A2ijl, A2ij, A2iju, F, F, Y0, M, L)
                for j = 2:L - 1, i = 2:M - 1
                    index = i + (j - 1) * M
                    Y0[index] -= rij[index] * (F[i + 1 + (j) * M] - F[i + 1 + (j - 2) * M] + F[i - 1 + (j - 2) * M] - F[i - 1 + (j) * M])
                end
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti - dt)
                updatePayoffExplicitTrans(Y0, useDirichlet, lbValue, M, L)
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            for n = 1:N
                #explicit
                explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                #explicit
                explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                #explicit
                explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
            λb = zeros(M*L)
        #    λh = zeros(M*L)
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
          for n = 1:N
                #explicit
                explicitStep(1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Y0, M, L)
                @. Y0 += dt*λb
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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

                @. F = max(Y2t- dt*λb, lowerBoundA)
                @. λb = max(0, λb + (lowerBoundA - Y2t)/dt)

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
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dth, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
             #DO
            a=0.5
            for n = 1:N
                #CS
                explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
                ti -= dth
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
                 Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                #DO
            explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
            ti -= dt
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
            rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            solveTrapezoidal2(0.5, 0.5, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0, F, Y1, M, L)
            solveTrapezoidal1(0.5, 0.5, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1, F, Y2, M, L)
            F = copy(Y2)
            # a=0.3334
            # explicitStep(1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Y0, M, L)
            # ti -= dt
            # Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                #DO
            explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
            ti -= dt
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
            rijNew, A1iljNew, A1ijNew, A1iujNew, A2ijlNew, A2ijNew, A2ijuNew = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
            solveTrapezoidal2(1, 1, A2ijl, A2ij, A2iju, A2ijlNew, A2ijNew, A2ijuNew, useDirichlet, Y0, F, Y1, M, L)
            solveTrapezoidal1(1, 1, A1ilj, A1ij, A1iuj, A1iljNew, A1ijNew, A1iujNew, useDirichlet, Y1, F, Y2, M, L)
            #MCS
            # a=0.3334
            # explicitStep(1.0, rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, F, Y0, M, L)
            # ti -= dt
            # Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
            Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
            rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
           for n = 1:N
                explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
                lbValue = computeLowerBoundary(isCall, useDirichlet, B, Smin, r, q, ti - dt)
                updatePayoffExplicitTrans(Y0, useDirichlet, lbValue, M, L)
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
                rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju = makeSystem(useExponentialFitting, upwindingThreshold, dt, Sc, Jc, Jch, Jct, S, J, Jm, V, JV, JVm, hm, hl, kappa, theta, sigma, rho, r, q, useDirichlet, M, L)
                explicitStep(rij, A1ilj, A1ij, A1iuj, A2ijl, A2ij, A2iju, F, Y0, M, L)
                ti -= dt
                Sc, Jc, Jch, Jct = makeJacobians(ti, cFunc, S, M)
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
    spl = Spline2D(S, V, Payoff; kx=3,ky=3, s = 0.0)
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
        println(spot, " ", method, " ", N, " ", M, " ", L, " ", price, " ",price-priceArray[i]," ", etime)
    end
    l2 = sqrt(l2/length(spotArray))
    if length(spotArray) > 1
        println(B, " ", method, " ", N, " ", M, " ", L, " ", l2," ",l2," ", etime)
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


A = [0.0]
B = [1.0]
C = [0.0]
X = [0.0]
leftSlope = 1.0
rightSlope = 1.0
kappa = 5.0
theta = 0.16
sigma = 0.9
rho = 0.1
r = 0.1
q = 0.0
v0 = 0.0625
T = 0.25
cFunc = CollocationFunction(A, B, C, X, leftSlope, rightSlope, T)
#M = 128
#L = 128
#N = 32
spotArray = [8.0, 9.0, 10.0,11.0,12.0]
K = 10.0
priceArray = [1.97731, 1.28000, 0.76969] #from cos
priceArray = [1.83887, 1.04835, 0.50147, 0.20819, 0.08043] #euro, v0=0.0625
priceArray = [2.0000, 1.1076, 0.5199, 0.2135, 0.0820] #am, v0=0.0625
priceArray = [2.078372, 1.333640, 0.795983, 0.448277,0.242813] #am v0=0.25
priceArray = [2.000000,1.107629,0.520038,0.213681,  0.082046] #am v0=0.0625
isEuropean = false
B = 0.0
isCall = false
Ns = [1024, 768,512, 384, 256, 192, 128, 96, 64, 56, 48, 32, 24, 16, 12, 8 ,6,4] #timesteps
Ns = reverse(Ns)
damping = "None"
method = "CS"
#N = 16
# M = 512
schemes = ["CS"]
for method in schemes
    if method == "LS" || method == "RKL" || method == "RKLM"
        damping = "None"
    else
        damping = "EU"
    end

for N in Ns
    price(isCall, isEuropean, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L,sDisc="Sinh", useExponentialFitting=true, smoothing="None",damping = damping, method = method, lsSolver="IRDS")
    #price(isCall, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L,sDisc="Linear", useVLinear=true, useExponentialFitting=true, smoothing="Averaging",damping = damping, method = method, lsSolver="IRDS")
    #price(isCall, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L, damping = damping, method = method)
end
end
method="RKL2"
for (M,L,N) in zip([64,128,256,512],[32,64,128,256],[8,16,32,64])
    #price(isCall, isEuropean, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L,sDisc="Sinh", useExponentialFitting=true, smoothing="None",damping = damping, method = method, lsSolver="IRDS")
    price(isCall, isEuropean, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L,sDisc="Linear", useExponentialFitting=true, smoothing="Averaging",damping = "None", method = method, lsSolver="IRDS")
end

#feller not satistfied
kappa = 1.15
theta = 0.0348
sigma = 0.39
rho = -0.64
r = 0.04
q = 0.0
v0 = 0.0348
T=0.25
spotArray = [90.0,100.0,110.0]
K = 100.0
priceArray = [10.0039, 3.2126, 0.9305]
for (M,L,N) in zip([64,128,256,512],[32,64,128,256],[8,16,32,64])
    #price(isCall, isEuropean, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L,sDisc="Sinh", useExponentialFitting=true, smoothing="None",damping = damping, method = method, lsSolver="IRDS")
    price(isCall, isEuropean, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L,sDisc="Linear", useVLinear=true, useExponentialFitting=true, smoothing="Kreiss",damping = "None", method = "RKL2", lsSolver="IRDS")
end
for (M,L,N) in zip([75,150,300],[38,75,150],[120,240,480])
          #price(isCall, isEuropean, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L,sDisc="Sinh", useExponentialFitting=true, smoothing="None",damping = damping, method = method, lsSolver="IRDS")
          price(isCall, isEuropean, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, M, L,sDisc="Sinh", useExponentialFitting=true, smoothing="Kreiss",damping = "None", method = "RKL2", lsSolver="IRDS", lambdaS=0.2, lambdaV=0.2,epsilon=1e-4,Xdev=4)
      end


spotArray = collect(90:110)
priceArray = zeros(length(spotArray))

Ns = [4096,3072,2048,1536, 1024, 768,512, 384, 256, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8] #timesteps
Ns = reverse(Ns)
prevPrice = 0.0
prevDiff = 0.0
for N in Ns
    newPrice = price(isCall, isEuropean, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, 100, 50,sDisc="Linear", useVLinear=true, useExponentialFitting=true, smoothing="Kreiss",damping = "None", method = "RKL2", lsSolver="IRDS")
    #println(N," ",price," ", price-prevPrice," ",prevDiff/(price-prevPrice))
    #prevDiff = price-prevPrice
    #prevPrice = newPrice
end

#case D of Hout
kappa = 0.5
theta = 0.04
sigma = 1.0
rho = -0.9
r = 0.05
q = 0.0
v0 = theta
T=10.0
#case B
kappa = 0.6067
theta = 0.0707
sigma = 0.2928
rho = -0.7571
r = 0.03
q = 0.0
v0 = theta
T=3.0
#case F
kappa = 1.0
theta = 0.09
sigma = 1.0
rho = -0.3
r = 0.03
q = 0.0
v0 = theta
T=5.0
spotArray = collect(7:17)*10
priceArray = zeros(length(spotArray))
for N in Ns
    newPrice = price(isCall, isEuropean, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, B, N, 101, 51,sDisc="Linear", useVLinear=true, useExponentialFitting=true, smoothing="Kreiss",damping = "None", method = "RKL2",Xdev=3,epsilon=1e-3)
end
Base.show(io::IO, x::Union{Float64,Float32}) = Base.Grisu._show(io, x, Base.Grisu.SHORTEST, 0, true, false)
for method in schemes
    if method == "LS" || method=="RKC2" || method == "RKL" || method == "RKL2" || method=="HBDF2" || method == "SC2A" || method=="SC2B"
        damping = "None"
    else
        damping = "DO"
    end
for N in Ns
    price(isCall, isEuropean, spotArray, priceArray, v0, kappa, theta, sigma, rho, r, q, T, cFunc, K, barrierLevel, N, M, L, damping = damping, method = method)
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
#Ns = [1024, 768,512, 384, 256, 192, 128, 96, 64, 56, 48, 32, 24, 16, 12, 8 ,6,4] #timesteps
#Ns = reverse(Ns)
#damping = "None"
#M = 128
#L = 32
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
