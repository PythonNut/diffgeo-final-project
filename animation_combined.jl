using DataFrames
using CSV
using Dierckx
using PyPlot
using PyCall
using RecursiveArrayTools
using ProgressMeter
@pyimport mpl_toolkits
@pyimport mpl_toolkits.mplot3d as mplot3d
@pyimport matplotlib.animation as anim

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
    F = svdfact(mapslices(normalize, V, 1))
    return F[:U] * F[:Vt]
end

@printf("Loading files...\n")
Tmag, Vmag = read_file_sampled("public_dataset/100669/100669_session_9/Magnetometer.csv")
Tgyr, Vgyr = read_file_sampled("public_dataset/100669/100669_session_9/Gyroscope.csv")
Tacc, Vacc = read_file_sampled("public_dataset/100669/100669_session_9/Accelerometer.csv")

@assert Tacc == Tgyr == Tacc

global T = Tmag #WLOG

@printf("Unifying gravitational and magnetic fields...\n")

Rs = Matrix{Float64}[]
for (i, t) in enumerate(T)
    acc, mag = Vacc[i,:], Vmag[i, :]
    R = hcat(acc, mag, cross(acc, mag))
    push!(Rs, orth(R))
end

Rs = Array(VectorOfArray(Rs))

prog = Progress(1000,1)

fig = figure("MyFigure")
ax = axes(projection="3d", xlim=(-1,1), ylim=(-1,1), zlim=(-1,1))
n = 5

global line = []
for i = 1:n
    push!(line, ax[:plot]([],[],[])[1])
end

function init()
    global line
    for i = 1:n
        line[i][:set_data]([],[])
        line[i][:set_3d_properties]([])
    end
    return tuple(line..., Union{})
end

function animate(i)
    global line
    global Rs
    global Vacc
    global Vmag
    println(i)
    ω = Rs[:,:,i+1]
    acc = normalize(Vacc[i+1,:])
    mag = normalize(Vmag[i+1,:])

    line[1][:set_data]([0, ω[1,1]], [0, ω[2,1]])
    line[1][:set_3d_properties]([0, ω[3,1]])

    line[2][:set_data]([0, ω[1,2]], [0, ω[2,2]])
    line[2][:set_3d_properties]([0, ω[3,2]])

    line[3][:set_data]([0, ω[1,3]], [0, ω[2,3]])
    line[3][:set_3d_properties]([0, ω[3,3]])

    line[4][:set_data]([0, mag[1]], [0, mag[2]])
    line[4][:set_3d_properties]([0, mag[3]])

    line[5][:set_data]([0, acc[1]], [0, acc[2]])
    line[5][:set_3d_properties]([0, acc[3]])
    
    return tuple(line..., Union{})
end

myanim = anim.FuncAnimation(fig, animate, init_func=init, frames=100, interval=20)
