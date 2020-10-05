using Pkg;Pkg.activate(".");Pkg.instantiate()
using OhMyREPL, Distances, Distributions, Plots,LinearAlgebra
enable_autocomplete_brackets(false)

# Distance metric
dist = Euclidean()
sym = issymmetric

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
dt = 1/512 # This is from the fep_physics SPM demo. Differs from monograph
T = 2048.0

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

    # This is what? Taken from the SPM demo
    # Think it's the inverse square scaling for F...?
    C = 1 ./ sqrt.(Δ_ij)
   # # Fix sqrt(0) issues
    C[isinf.(C)] .= 0.
    C[isnan.(C)] .= 0.

    # Get absolute difference between all particles aₑ.
    aₑⁱ= ones(n) * aₑ' # Make a handy matrix
    Δaₑⁱ = -abs.(aₑⁱ - aₑⁱ')

    Q = 8. * exp.(Δaₑⁱ) .-4.
    Q = Q - I(n) .* diag(Q)
    Q = Q ./ Δ_ij.^2
    Q[isinf.(Q)] .= 0.
    Q[isnan.(Q)] .= 0.
    # Compute the inner term in eq 3.2.
    # Q = 8. * exp.(Δaₑⁱ .* 2) .-2 - this line is LAWKI
    Q = ((8. * exp.(Δaₑⁱ) .-4.) .* Δ_ij.^2)

    #scaling  = 1. .* (Δ_ij.^3)
    #scaling[isinf.(scaling)] .= 0.

    F_inner = C .^2 .* (Q -C)#Q - scaling
    F_inner[isinf.(F_inner)] .= 0.
    F_inner[isnan.(F_inner)] .= 0.
    # Compute F
    F = sum(F_inner,dims=2)

    # Differences by dimensions to set direction of the F vector
    aₙ¹= ones(n) * aₙ[:,1]' # Make a handy matrix
    Δaₙ¹ = (aₙ¹ - aₙ¹') .* A
    aₙ²= ones(n) * aₙ[:,2]' # Make a handy matrix
    Δaₙ² = (aₙ² - aₙ²') .* A
    F1 = Δaₙ¹ * F
    F2 = Δaₙ² * F

    # F vector.
    f = hcat(F1,F2)

    #CQ = C.^2 .* (Q-C) # I think this is F?

    # Does this need to be an average? Text and equation disagree here
    s_bar = A * sₑ
    #F_inner = Δ_ij .* (Q  ./  (Δ_ij .^2) ) .- 1. ./ (Δ_ij .^3 )

    #F_inner = Δ_ij .* ((8. .* exp.(Δaₑⁱ) .- 4.)  ./  ((Δ_ij .^2) )) .- 1. ./ (Δ_ij .^3 )
    # Fix divisions by 0

    # Update equations
    sₑ_dot = (10 * (aₑ - sₑ) + s_bar) .* κ .+ rand(ω)
    aₑ_dot = (32 .* sₑ - aₑ - μ .* sₑ) .* κ .+ rand(ω)
    μ_dot   = (sₑ .*aₑ - (8/3) .*μ) .* κ .+ rand(ω)

    aₙ_dot = (1. .+ (μ ./ 64.)) .* sₙ .+ rand(ω)
    sₙ_dot = 2*f .- 8 * sₙ - aₙ .+ rand(ω)

    # Apply updates
    global sₑ += dt * sₑ_dot
    global aₑ += dt * aₑ_dot
    global μ += dt * μ_dot

    global aₙ += dt .* aₙ_dot
    global sₙ += dt .* sₙ_dot

    p = scatter(aₙ[:,1],aₙ[:,2],markersize=4.5,legend=false)
    display(p)
end



#,xlim=(-5,5),ylim=(-5,5),
