using Pkg;Pkg.activate(".");Pkg.instantiate()
using OhMyREPL, Distances, Distributions, Plots
using LinearAlgebra: I
enable_autocomplete_brackets(false)

# Distance metric
dist = Euclidean()

# Number of particles
n = 3
σ_aₑ = 1.
σ_sₑ = 1.
σ_μ = 1.

σ_aₙ = 1.
σ_sₙ = 1.

# Step size and horizon
dt = 1/512
T = 10.0



# Electrochemical states
global aₑ = rand(MvNormal(I(n) * σ_aₑ))
global sₑ = rand(MvNormal(I(n) * σ_sₑ))
global μ = rand(MvNormal(I(n) * σ_μ))

# Newtonian states
global aₙ = reshape(rand(MvNormal(zeros(2n),I(2n) * σ_aₙ)),n,2)
global sₙ = reshape(rand(MvNormal(zeros(2n),I(2n) * σ_aₙ)),n,2)

# Rate parameter
κ = (1 .- exp.(-4 .* rand(n))) / 32

# Noise
ω = MvNormal(I(n) * 1/8)

a_vec = []
#let
#    aₑ = aₑ,
#    sₑ = sₑ,
#    μ= μ,
#    aₙ = aₙ,
#    sₙ = sₙ
for t in 1:dt:T
    # Julia is optimised for columnwise operation. Transpose to leverage this. Result is symmetric anyway. Distance matrix
    Δ_ij = pairwise(dist,aₙ',dims=2)

    # Adjencency matrix from thresholding distances at greater or equal to 1. Remove selfconnections
    A = (Δ_ij .<= 1.) - I(n)

    # Eliminate redundant entries
    Δ_ij = Δ_ij .* A
    C = 1 ./ sqrt.(Δ_ij)

    # Does this need to be an average? Text and equation disagree here
    s_bar = sum(A .* sₑ,dims=2)

    # Compute the inner term in eq 3.2.
    F_inner = Δ_ij .* ((8. .* exp.(-Δ_ij) .- 4.)  ./  ((Δ_ij .^2) )) .- 1. ./ (Δ_ij .^3 )
    # Fix divisions by 0
    F_inner[isnan.(F_inner)] .= 0.

    # Compute F
    F = sum(A .* F_inner,dims=2)

    # Update equations
    sₑ_dot = (10 * (aₑ - sₑ) + s_bar) .* κ .+ rand(ω)
    aₑ_dot = (32 .* sₑ - aₑ - μ .* sₑ) .* κ .+ rand(ω)
    μ_dot   = (sₑ .*aₑ - (7/3) .*μ) .* κ .+ rand(ω)

    aₙ_dot = (1. .+ (μ / 64.)) .* sₙ .+ rand(ω)
    sₙ_dot = 2*F .- 8 * sₙ - aₙ .+ rand(ω)

    # Apply updates
    global sₑ += dt * sₑ_dot
    global aₑ += dt * aₑ_dot
    global μ += dt * μ_dot

    global aₙ += dt .* aₙ_dot
    global sₙ += dt .* sₙ_dot
end




println(aₙ)
aₙ[:,1] - aₙ[:,2]

scatter(aₙ[:,1],aₙ[:,2])
