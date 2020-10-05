using Pkg;Pkg.activate(".");Pkg.instantiate()
using OhMyREPL, Distances, Distributions, Plots,LinearAlgebra
enable_autocomplete_brackets(false)

# This demo is based off of the equations in AFEPFAPP. They differ slightly from the implementation in fep_physics.m

# Distance metric
dist = Euclidean()
sym = issymmetric # Handy for debugging

# Number of particles
n = 128
# Electrochemical state variances
σ_aₑ = 1. /32.
σ_sₑ = 1. /32.
σ_μ = 1. /32.

# Newtonian state variances
σ_aₙ = 4.
σ_sₙ = 1.

# Random fluctuations
ω = MvNormal(I(n) * 1/8)

# Step size and horizon
dt = 1/512 # fep_physics demo runs 1/32
T = 20.0

# Electrochemical states
global aₑ = rand(MvNormal(I(n) * σ_aₑ))
global sₑ = rand(MvNormal(I(n) * σ_sₑ))
global μ = rand(MvNormal(I(n) * σ_μ))

# Newtonian states
global aₙ = reshape(rand(MvNormal(zeros(2n),I(2n) * σ_aₙ)),n,2)
global sₙ = reshape(rand(MvNormal(zeros(2n),I(2n) * σ_aₙ)),n,2)

# Rate parameter. fep_physics demo does not divide by 32. That happens when setting initial states for some reason
κ = (1. .- exp.(-4. .* rand(n))) / 32.


for t in 1:dt:T
    # Distance matrix
    Δ_ij = pairwise(dist,aₙ',dims=2)

    # Adjacency matrix without self connections
    A = (Δ_ij .<= 1.) - I(n)

    # Eliminate redundant entries
    Δ_ij = Δ_ij .* A

    # Get absolute difference between all particles aₑ.
    aₑⁱ= ones(n) * aₑ'
    Δaₑⁱ = -abs.(aₑⁱ - aₑⁱ')

    # Compute the inner term in eq 3.2.
    # Q = 8. * exp.(Δaₑⁱ .* 2) .-2 - this line is LAWKI
    Q = (8. * exp.(Δaₑⁱ) .-4.)

    # This is how fep_physics.m implements the Δᵢⱼscaling
    #C = 1 ./ sqrt.(Δ_ij)
    C1 = 1 ./ (Δ_ij .^2)
    C2 = 1 ./ (Δ_ij .^3)

    # Compute F
    f = (Q .* C1)  - C2
    # Fix nans/infs. They happen earlier, but we catch them here
    f[isinf.(f)] .= 0.
    f[isnan.(f)] .= 0.
    F_mag = sum(f,dims=2)

    # Differences by dimensions to set direction of the F vector
    # Thanks to Casper Hesp for this detail :)
    aₙ¹= ones(n) * aₙ[:,1]'
    Δaₙ¹ = (aₙ¹ - aₙ¹') .* A
    aₙ²= ones(n) * aₙ[:,2]'
    Δaₙ² = (aₙ² - aₙ²') .* A
    F1 = Δaₙ¹ * F_mag
    F2 = Δaₙ² * F_mag

    # F vector.
    F = hcat(F1,F2)

    # Average state of neighbouring particles
    s_bar = (A * sₑ) ./ sum(A, dims=2)
    # Fix nans/infs
    s_bar[isinf.(s_bar)] .= 0.
    s_bar[isnan.(s_bar)] .= 0.

    # Update equations
    # Electrochemical
    sₑ_dot = (10 * (aₑ - sₑ) + s_bar) .* κ .+ rand(ω)
    aₑ_dot = (32 .* sₑ - aₑ - μ .* sₑ) .* κ .+ rand(ω)
    μ_dot   = (sₑ .*aₑ - (8/3) .*μ) .* κ .+ rand(ω)

    # Newtonian
    aₙ_dot = (1. .+ (μ ./ 64.)) .* sₙ .+ rand(ω)
    sₙ_dot = 2*F .- 8 * sₙ - aₙ .+ rand(ω)

    # Apply updates
    global sₑ += dt * sₑ_dot
    global aₑ += dt * aₑ_dot
    global μ += dt * μ_dot

    global aₙ += dt .* aₙ_dot
    global sₙ += dt .* sₙ_dot

    # Plot results
    p = scatter(aₙ[:,1],aₙ[:,2],markersize=4.5,legend=false,title=t)
    display(p)
end


# Final adjacency matrix
B = (A + A' + A'*A) .> 0
