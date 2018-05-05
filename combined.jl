using DataFrames
using CSV
using Dierckx
using PyPlot
using PyCall
using RecursiveArrayTools
using ProgressMeter
using StatsBase
using DSP

@pyimport mpl_toolkits
@pyimport mpl_toolkits.mplot3d as mplot3d
@pyimport matplotlib.animation as anim

function fcat(mats)
    # This is a faster cat(3, Cs...)
    return Array(VectorOfArray(mats))
end

function read_file_sampled(file)
    df = CSV.read(file, datarow=1)
    t = -1
    ts, vs = Int64[], Vector{Float64}[]
    for row in eachrow(df)
        tdt, x, y, z = row[1], row[4], row[5], row[6]
        if t == tdt
            continue
        end
        push!(ts, tdt)
        push!(vs, [x,y,z])
        t = tdt
    end
    vmat = fcat(vs)
    ts -= ts[1]
    return ts, vmat
end

function spline_sample(ts, vs, dt)
    tsample = 0:dt:ts[end]
    vspline = mapslices(cs->Spline1D(ts, cs, k=1), vs, 2)
    vsample = (s->s(tsample)).(vspline)
    return tsample, hcat(vsample...)
end

function orth(V)
    F = svdfact(mapslices(normalize, V, 1))
    return F[:U] * F[:Vt]
end

function low_pass(T, xs, fs)
    filt = digitalfilter(Lowpass(0.3, fs=fs), FIRWindow(hanning(10001)))
    xsf = filtfilt(filt, xs)
    return xsf
end

function high_pass(T, xs, fs)
    filt = digitalfilter(Highpass(0.3, fs=fs), FIRWindow(hanning(10001)))
    xsf = filtfilt(filt, xs)
    return xsf
end

function draw_vec(line, o, v)
    line[:set_data]([o[1], o[1] + v[1]], [o[2], o[2] + v[2]])
    line[:set_3d_properties]([o[3], o[3] + v[3]])
end

draw_vec(line, v) = draw_vec(line, [0,0,0], v)

function make_lines(marks)
    return [ax[:plot]([],[],[], mark)[1] for mark in marks]
end

function proj(u, v)
    return dot(u, v)/dot(u, u) * u
end

function cumtrapz(x::Array{Float64, 1}, y::Array{Float64}, dim::Integer=1)
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

@printf("Loading files...\n")
Tmag_raw, Vmag_raw = read_file_sampled("public_dataset/100669/100669_session_9/Magnetometer.csv")
Tgyr_raw, Vgyr_raw = read_file_sampled("public_dataset/100669/100669_session_9/Gyroscope.csv")
Tacc_raw, Vacc_raw = read_file_sampled("public_dataset/100669/100669_session_9/Accelerometer.csv")

sample_rate_ms = mode(vcat(diff(Tmag_raw), diff(Tgyr_raw), diff(Tacc_raw)))

Tmag, Vmag = spline_sample(Tmag_raw/1000, Vmag_raw, sample_rate_ms/1000)
Tgyr, Vgyr = spline_sample(Tgyr_raw/1000, Vgyr_raw, sample_rate_ms/1000)
Tacc, Vacc = spline_sample(Tacc_raw/1000, Vacc_raw, sample_rate_ms/1000)

@assert Tacc == Tgyr == Tacc
T = Tmag #WLOG

function integrate3d(xs)
    return mapslices(cs->cumtrapz(collect(T), cs), xs, 1)
end

@printf("Filtering gravitational and magnetic fields...\n")
Vgvt = low_pass(T, Vacc, 100)
Vmag = low_pass(T, Vmag, 100)


@printf("Unifying gravitational and magnetic fields...\n")
Rs = Matrix{Float64}[]
for (i, t) in enumerate(T)
    gvt, mag = Vgvt[i,:], Vmag[i, :]
    # R = hcat(gvt, mag, cross(gvt, mag))
    # push!(Rs, orth(R))
    gvtn = -normalize(gvt)
    magn = normalize(mag - proj(gvtn, mag))
    R = hcat(magn, cross(gvtn, magn), gvtn)
    push!(Rs, R)
end

Rs = fcat(Rs)
Rspline = mapslices(x->Spline1D(T, x), Rs, 3)
accspline = mapslices(x->Spline1D(T, x), Vacc, 1)
gvtspline = mapslices(x->Spline1D(T, x), Vgvt, 1)
magspline = mapslices(x->Spline1D(T, x), Vmag, 1)

@printf("Fusing gyroscopic data w/ absolute sensors...\n")
t = 0
C = Rs[:,:,1]
ts, Cs = [0.0], [C]
for i in 2:size(T, 1)
    (tdt, (x, y, z)) = T[i], Vgyr[i,:]
    dt = tdt - t
    B = [0 -z y; z 0 -x; -y x 0] * dt
    σ = norm([x, y, z] * dt)
    C *= eye(3) + sin(σ)/σ*B + (1 - cos(σ))/σ^2*B^2
    mix = 0.99
    C = mix*C + (1 - mix)*Rs[:,:,i]
    push!(ts, tdt)
    push!(Cs, C)
    t = tdt
end

Cs = fcat(Cs)
Cspline = mapslices(x->Spline1D(ts, x), Cs, 3)


@printf("Translating accelerations into global frame...\n")
Cinv = permutedims(Cs, [2,1,3])

Vacc_global = zeros(Vacc)
for (i, t) in enumerate(T)
    Vacc_global[i,:] = Cinv[:,:,i] * Vacc[i,:]
end

Cinv_spline = mapslices(x->Spline1D(ts, x), Cinv, 3)

@printf("Filtering global accelerations...\n")

Vacc_global_transient = high_pass(T, Vacc_global, 100)

@printf("Integrating accelerations to get velocity and filtering...\n")

Vvel_global = integrate3d(Vacc_global_transient)
Vvel_global_transient = high_pass(T, Vvel_global, 100)

@printf("Integrating velocities to get position and filtering...\n")

Vpos_global = integrate3d(Vvel_global_transient)
Vpos_global_transient = high_pass(T, Vpos_global, 100)

@printf("Plotting results...\n")
fig = figure("Combined")
ax = axes(projection="3d", xlim=(-1,1), ylim=(-1,1), zlim=(-1,1))

n = 1000
tspan = linspace(0, 30, n)

lines = make_lines(["r-", "g-", "b-", "r--", "g--", "b--", "k-"])

function init()
    for i = 1:length(lines)
        lines[i][:set_data]([],[])
        lines[i][:set_3d_properties]([])
    end
    return tuple(lines..., Union{})
end

function animate(i)
    ω = (x->x(tspan[i+1])).(Cinv_spline)
    ω2 = reshape((x->x(tspan[i+1])).(Rspline), Val{2})'
    acc = Vpos_global_transient[i+1,:] 
    # mag = normalize(Vmag[i+1,:])

    draw_vec(lines[1], acc, ω[:,1])
    draw_vec(lines[2], acc, ω[:,2])
    draw_vec(lines[3], acc, ω[:,3])

    draw_vec(lines[4], acc, ω2[:,1])
    draw_vec(lines[5], acc, ω2[:,2])
    draw_vec(lines[6], acc, ω2[:,3])

    draw_vec(lines[7], acc)

    return tuple(lines..., Union{})
end

myanim = anim.FuncAnimation(fig, animate, init_func=init, frames=n, interval=1)
