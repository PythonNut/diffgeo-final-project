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

include("util.jl")

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
