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
Tmag, Vmag = read_file_sampled("100669/100669/100669_session_9/Magnetometer.csv")
Tgyr, Vgyr = read_file_sampled("100669/100669/100669_session_9/Gyroscope.csv")
Tacc, Vacc = read_file_sampled("100669/100669/100669_session_9/Accelerometer.csv")

@assert Tacc == Tgyr == Tacc

T = Tmag #WLOG

@printf("Unifying gravitational and magnetic fields...\n")

Rs = Matrix{Float64}[]
for (i, t) in enumerate(T)
    gyr, acc = Vgyr[i,:], Vacc[i, :]
    R = hcat(gyr, acc, cross(gyr, acc))
    push!(Rs, orth(R))
end

Rs = Array(VectorOfArray(Rs))
