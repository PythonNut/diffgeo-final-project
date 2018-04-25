using DataFrames
using CSV
using Dierckx
using DifferentialEquations
using Plots
using StaticArrays
using ProgressMeter

pyplot()

file = "100669/100669/100669_session_9/Gyroscope.csv"
df = CSV.read(file, header=collect(1:7))

T, ωx, ωy, ωz = Int64[], Float64[], Float64[], Float64[]
for i in 1:size(df, 1)
    (t, _, _, x, y, z, _) = Array(df[i,:])
    if length(T) > 0 && T[end] == t
        continue
    end

    push!(ωx, x)
    push!(ωy, y)
    push!(ωz, z)
    push!(T, t)
end

T = (T - T[1])/1000
ωxs, ωys, ωzs = Spline1D.([T], [ωx, ωy, ωz])

function W(t)
    x, y, z = ωxs(t), ωys(t), ωzs(t)
    return @SMatrix [ 0 -z  y;
                      z  0 -x;
                     -y  x  0]
end

u0 = @SMatrix eye(3)
tspan = (0.0, T[end])
f(u, p, t) = W(t)*u
prob = ODEProblem(f, u0, tspan)
sol = solve(prob)

plt = path3d(1, xlim=(-1,1), ylim=(-1,1), zlim=(-1,1),
             xlab="x", ylab="y", zlab="z",
             title="Rotation", marker=1)

n = 5000
prog = Progress(n,1)

anim = @animate for t in linspace(0, 100, n)
    ω = sol(t)[:, 1]
    push!(plt, ω...)
    next!(prog)
end every 10

mp4(anim, "omega.mp4")
