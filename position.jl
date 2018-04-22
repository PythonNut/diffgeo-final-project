using DataFrames
using CSV
using Plots
using ProgressMeter

pyplot()

@everywhere begin
    using Dierckx
    using DifferentialEquations
    using DSP
end

file = "100669/100669/100669_session_9/Accelerometer.csv"
df = CSV.read(file, header=collect(1:7))
@everywhere fs = 1000

T, Xa, Ya, Za = Vector{Int64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
for i in 1:size(df, 1)
    (t, _, _, x, y, z, _) = Array(df[i,:])
    if length(T) > 0 && T[end] == t
        continue
    end
    push!(Xa, x)
    push!(Ya, y)
    push!(Za, z)
    push!(T, t)
end

@everywhere function ode_integrate(ix, tspan)
    prob = ODEProblem((u, p, t) -> ix(t), 0.0, tspan)
    return solve(prob)
end

@everywhere function high_pass(T, ix)
    filt = digitalfilter(Highpass(0.3, fs=fs), FIRWindow(hamming(10001)))
    Tms = T[1]:0.001:T[end]
    ixu = ix(Tms)
    ixf = filtfilt(filt, ixu)
    return Spline1D(Tms, ixf)
end

T = (T - T[1])/fs
tspan = (0.0, T[end])

tic()

println("Interpolating accelerations...")
Xai, Yai, Zai = pmap(Spline1D, fill(T, 3), [Xa, Ya, Za])
println("Filtering accelerations...")
Xafi, Yafi, Zafi = pmap(high_pass, fill(T, 3), [Xai, Yai, Zai])
println("Computing velocities...")
Xv, Yv, Zv = pmap(ode_integrate, [Xafi, Yafi, Zafi], fill(tspan, 3))
println("Filtering velocities...")
Xvfi, Yvfi, Zvfi = pmap(high_pass, fill(T, 3), [Xv, Yv, Zv])
println("Computing positions...")
Xx, Yx, Zx = pmap(ode_integrate, [Xvfi, Yvfi, Zvfi], fill(tspan, 3))
println("Filtering positions...")
Xxf, Yxf, Zxf = pmap(high_pass, fill(T, 3), [Xx, Yx, Zx])

toc()

plt = path3d(1,
             xlab="x", ylab="y", zlab="z",
             title="Location", marker=1)

println("Rendering animation...")
n = 5000
prog = Progress(n,1)

anim = @animate  for t in linspace(0, 100, n)
    x = Xxf(t), Yxf(t), Zxf(t)
    push!(plt, x...)
    next!(prog)
end every 10

mp4(anim, "position.mp4")
