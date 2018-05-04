using DataFrames
using CSV
using Dierckx
using Plots
using RecursiveArrayTools
using ProgressMeter

pyplot()

function read_file_sampled(file)
    df = CSV.read(file, header=collect(1:7))
    t = -1
    ts, vs = Int64[], Vector{Float64}[]
    for row in eachrow(df)
        (tdt, _, _, x, y, z, _) = Array(row)
        if t == tdt
            continue
        end
        push!(ts, tdt)
        push!(vs, [x,y,z])
        t = tdt
    end
    # This is a faster cat(3, Cs...)
    vmat = Array(VectorOfArray(vs))
    ts -= ts[1]
    tsample = 0:0.002:ts[end]/1000
    vspline = mapslices(cs->Spline1D(ts/1000, cs), vmat, 2)
    vsample = (s->s(tsample)).(vspline)
    return tsample, hcat(vsample...)
end

function orth(V)
    F = svdfact(mapslices(normalize, V, 2))
    return F[:U] * F[:Vt]
end

@printf("Loading files...\n")
Tmag, Vmag = read_file_sampled("public_dataset/100669/100669_session_10/Magnetometer.csv")
Tgyr, Vgyr = read_file_sampled("public_dataset/100669/100669_session_10/Gyroscope.csv")
Tacc, Vacc = read_file_sampled("public_dataset/100669/100669_session_10/Accelerometer.csv")

@assert Tacc == Tgyr == Tacc

T = Tmag #WLOG

@printf("Unifying gravitational and magnetic fields...\n")

Rs = Matrix{Float64}[]
for (i, t) in enumerate(T)
    acc, mag = Vacc[i,:], Vmag[i, :]
    R = hcat(acc, mag, cross(acc, mag))
    push!(Rs, orth(R))
end

Rs = Array(VectorOfArray(Rs))

@printf("Integrating gyroscope w.r.t. absolute sensors...\n")
t = 0
C = Rs[:,:,1]
ts, Cs = [0.0], [C]
for i in 2:size(T, 1)
    (tdt, (x, y, z)) = T[i], Vgyr[i,:]
    dt = tdt - t
    B = [0 -z y; z 0 -x; -y x 0] * dt
    σ = norm([x, y, z] * dt)
    C *= eye(3) + sin(σ)/σ*B + (1 - cos(σ))/σ^2*B^2
    C = 0.98*C + 0.02*Rs[:,:,i]
    push!(ts, tdt)
    push!(Cs, C)
    t = tdt
end

Cs = Array(VectorOfArray(Cs))
Cspline = mapslices(x->Spline1D(ts, x), Cs, 3)

n = 1000
prog = Progress(n, 1)
anim = @animate for t in linspace(0, 10, n)
    ω = (x->x(t)).(Cspline)
    # ω = Cs[:,:,i]
    # acc = normalize(Vacc[i,:])
    # mag = normalize(Vmag[i,:])
    plot((x->[0,x]).(ω[:,1])..., color="red", label="x")
    plot!((x->[0,x]).(ω[:,2])..., color="green", label="y")
    plot!((x->[0,x]).(ω[:,3])..., color="blue", label="z")
    # plot!((x->[0,x]).(acc)..., color="teal", label="acc")
    # plot!((x->[0,x]).(mag)..., color="orange", label="mag")
    ylims!(-1, 1)
    xlims!(-1, 1)
    zlims!(-1, 1)
    next!(prog)
end

mp4(anim, "fused.mp4")
