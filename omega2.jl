using DataFrames
using CSV
using Dierckx
using Plots
using RecursiveArrayTools
using ProgressMeter

pyplot()

file = "100669/100669/100669_session_9/Gyroscope.csv"
df = CSV.read(file, header=collect(1:7))
df[1] -= df[1,1]
df[1] /= 1000

t = 0
C = eye(3)
ts, Cs = [0.0], [C]

for i in 1:size(df, 1)
    (tdt, _, _, x, y, z, _) = Array(df[i,:])
    if t == tdt
        continue
    end
    dt = tdt - t
    B = [0 -z y; z 0 -x; -y x 0] * dt
    σ = norm([x, y, z] * dt)
    C *= eye(3) + sin(σ)/σ*B + (1 - cos(σ))/σ^2*B^2
    push!(ts, tdt)
    push!(Cs, C)
    t = tdt
end

# This is a faster cat(3, Cs...)
Cs = Array(VectorOfArray(Cs))
Cspline = mapslices(x->Spline1D(ts, x), Cs, 3)

plt = path3d(1, xlim=(-1,1), ylim=(-1,1), zlim=(-1,1),
             xlab="x", ylab="y", zlab="z",
             title="Rotation", marker=1)

n = 5000
prog = Progress(n,1)
anim = @animate for t in linspace(0, 100, n)
    ω = (x->x(t)).(Cspline)[:, 1]
    push!(plt, ω...)
    next!(prog)
end every 10

mp4(anim, "omega2.mp4")
