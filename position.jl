using DataFrames
using Plots
using ProgressMeter

pyplot()

@everywhere begin
    using Dierckx
    using DifferentialEquations
    using DSP
end

file = "100669/100669/100669_session_9/Accelerometer.csv"
df = readcsv(file)
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

@everywhere function cumtrapz(x::Array{Float64, 1}, y::Array{Float64}, dim::Integer=1)
    perm = [dim:max(length(size(y)),dim); 1:dim-1];
    y = permutedims(y, perm);
    if ndims(y) == 1
        n = 1;
        m = length(y);
    else
        m, n = size(y);
    end

    if n == 1
        dt = diff(x)/2.0;
        z = [0; cumsum(dt.*(y[1:(m-1)] + y[2:m]))];
    else
        dt = repmat(diff(x)/2.0,1,n);
        z = [zeros(1,n); cumsum(dt.*(y[1:(m-1), :] + y[2:m, :]),1)];
        z = ipermutedims(z, perm);
    end

    return z
end

@everywhere function supersample(T, Ts, xs)
    xsi = Spline1D(T, xs)
    return xsi(Ts)
end

@everywhere function ode_integrate(ix, tspan)
    prob = ODEProblem((u, p, t) -> ix(t), 0.0, tspan)
    return solve(prob)
end

@everywhere function high_pass(T, xs)
    filt = digitalfilter(Highpass(0.3, fs=fs), FIRWindow(hamming(10001)))
    xsf = filtfilt(filt, xs)
    return xsf
end

T = (T - T[1])/fs
Tms = collect(T[1]:0.001:T[end])
tspan = (0.0, T[end])

tic()

println("Interpolating accelerations...")
Xas, Yas, Zas = pmap(supersample, fill(T, 3), fill(Tms, 3), [Xa, Ya, Za])
println("Filtering accelerations...")
Xaf, Yaf, Zaf = pmap(high_pass, fill(Tms, 3), [Xas, Yas, Zas])
println("Computing velocities...")
Xv, Yv, Zv = pmap(cumtrapz, fill(Tms, 3), [Xaf, Yaf, Zaf])
println("Filtering velocities...")
Xvf, Yvf, Zvf = pmap(high_pass, fill(Tms, 3), [Xv, Yv, Zv])
println("Computing positions...")
Xx, Yx, Zx = pmap(cumtrapz, fill(Tms, 3), [Xvf, Yvf, Zvf])
println("Filtering positions...")
Xxf, Yxf, Zxf = pmap(high_pass, fill(Tms, 3), [Xx, Yx, Zx])
Xxfi, Yxfi, Zxfi = pmap(Spline1D, fill(Tms, 3), [Xxf, Yxf, Zxf])

toc()

plt = path3d(1,
             xlab="x", ylab="y", zlab="z",
             title="Location", marker=1)

println("Rendering animation...")
n = 5000
prog = Progress(n,1)

anim = @animate  for t in linspace(0, 100, n)
    x = Xxfi(t), Yxfi(t), Zxfi(t)
    push!(plt, x...)
    next!(prog)
end every 10

mp4(anim, "position.mp4")
