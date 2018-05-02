using DataFrames
using CSV
using Dierckx
using PyPlot
using PyCall
using RecursiveArrayTools
using ProgressMeter
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
    F = svdfact(mapslices(normalize, V, 2))
    return F[:U] * F[:Vt]
end

@printf("Loading files...\n")
Tmag, Vmag = read_file_sampled("public_dataset/100669/100669_session_9/Magnetometer.csv")
Tgyr, Vgyr = read_file_sampled("public_dataset/100669/100669_session_9/Gyroscope.csv")
Tacc, Vacc = read_file_sampled("public_dataset/100669/100669_session_9/Accelerometer.csv")

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

prog = Progress(1000,1)
# function animate(i)
#     ω = Rs[:,:,i]
#     acc = normalize(Vacc[i,:])
#     mag = normalize(Vmag[i,:])
#     plot((x->[0,x]).(ω[:,1])..., color="red", label="x")
#     plot!((x->[0,x]).(ω[:,2])..., color="green", label="y")
#     plot!((x->[0,x]).(ω[:,3])..., color="blue", label="z")
#     plot!((x->[0,x]).(acc)..., color="teal", label="acc")
#     plot!((x->[0,x]).(mag)..., color="orange", label="mag")
#     ylims!(-1,1)
#     xlims!(-1,1)
#     zlims!(-1,1)
#     next!(prog)
# end

fig = figure()
ax = fig[:add_subplot](111, projection="3d")
n = 5

global line = Array(Any,n)
for i = 1:n
    line[i]   = ax[:plot]([],[],[])[1]
end

function init()
    global line
    for i = 1:n
        line[i][:set_data]([],[])
        line[i][:set_3d_properties]([])
    end
    result = tuple(tuple([line[i] for i = 1:n]...)...)
    return result
end

#To speed up the animation, we will only plot every 'step'th data point
step = 125

function animate(i)
    global line

    ω = Rs[:,:,i]
    acc = normalize(Vacc[i,:])
    mag = normalize(Vmag[i,:])

    line[0][:set_data](ω[0][0],ω[1][0])
    line[0][:set_3d_properties](ω[2][0])

    line[1][:set_data](ω[0][1],ω[1][1])
    line[1][:set_3d_properties](ω[2][1])

    line[2][:set_data](ω[0][2],ω[1][2])
    line[2][:set_3d_properties](ω[2][2])

    line[3][:set_data](mag[0],mag[1])
    line[3][:set_3d_properties](mag[2])

    line[4][:set_data](acc[0],acc[1])
    line[4][:set_3d_properties](acc[2])
    
    # result = tuple(tuple([line[i] for i = 1:n]...)...)
    # return result
end

# call the animator.  
myanim = anim.FuncAnimation(fig, animate, init_func=init, frames=1000, interval=20)

# mp4(anim1, "orientation.mp4")









