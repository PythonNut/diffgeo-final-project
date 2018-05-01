using DataFrames
using CSV
using Dierckx
using Plots
using RecursiveArrayTools
using ProgressMeter

pyplot()

file = "100669/100669/100669_session_9/Magnetometer.csv"
df = CSV.read(file, header=collect(1:7))
df[1] -= df[1,1]
df[1] /= 1000

t = 0
ts, Ms = [], []

for i in 1:size(df, 1)
    (tdt, _, _, x, y, z, _) = Array(df[i,:])
    if t == tdt
        continue
    end
    push!(ts, tdt)
    push!(Ms, [x,y,z])
    t = tdt
end

# This is a faster cat(3, Cs...)
Ms = Array(VectorOfArray(Ms))
Mspline = mapslices(x->Spline1D(ts, x), Ms, 2)

plt = path3d(1,
             xlab="x", ylab="y", zlab="z",
             title="Magnetic field", marker=1)

n = 5000
prog = Progress(n,1)
anim = @animate for t in linspace(0, 100, n)
    ω = normalize((x->x(t)).(Mspline)[:])
    push!(plt, ω...)
    next!(prog)
end every 10

mp4(anim, "mag.mp4")
