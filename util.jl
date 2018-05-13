using DataFrames
using CSV
using Dierckx
using RecursiveArrayTools
using DSP
using Glob
using Rotations

function fcat(mats)
    # This is a faster cat(3, Cs...)
    return Array(VectorOfArray(mats))
end

function orth(V)
    F = svdfact(mapslices(normalize, V, 2))
    return F[:U] * F[:Vt]
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

function get_sessions()
    return glob("public_dataset/*/*")
end

function db_key(session)
    return replace(session, "/", "_")
end

function get_session_activity(session)
    csv = joinpath(session, "Activity.csv")
    @assert isfile(csv)
    df = CSV.read(csv, datarow=1)
    activities = unique(df[:9])
    @assert length(activities) == 1
    return activities[1] % 6
end

function filter_sessions(sessions, db)
    keys = names(db)
    return filter(session->db_key(session) in keys, sessions)
end

function read_session(session)
    key = db_key(session)
    R, x = readmmap(db[key * "/rot"]), readmmap(db[key * "/pos"])
    activity = get_session_activity(session)
    return R, x, activity
end

function walking(id)
    return id in [0, 2, 4]
end

function spline_sample(ts, vs, dt)
    tsample = 0:dt:ts[end]
    vspline = mapslices(cs->Spline1D(ts, cs, k=1), vs, 2)
    vsample = (s->s(tsample)).(vspline)
    return tsample, hcat(vsample...)
end

function low_pass(T, xs, fs)
    filt = digitalfilter(Lowpass(0.3, fs=fs), FIRWindow(hanning(1001)))
    xsf = filtfilt(filt, xs)
    return xsf
end

function high_pass(T, xs, fs)
    filt = digitalfilter(Highpass(0.3, fs=fs), FIRWindow(hanning(1001)))
    xsf = filtfilt(filt, xs)
    return xsf
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

function integrate3d(T, xs)
    Tc = collect(T)
    return mapslices(cs->cumtrapz(Tc, cs[1:length(Tc)]), xs, 1)
end

function normalize_batch(R, x)
    R0 = inv(SPQuat(R[1,:]...))
    Rn = zeros(R)
    for i in 1:size(R, 1)
        q=R0*SPQuat(R[i,:]...)
        Rn[i,1] = q.x
        Rn[i,2] = q.y
        Rn[i,3] = q.z
    end
    xn = zeros(x)
    for i in 1:size(xn, 1)
        xn[i,:] = R0 * x[i,:]
    end
    return Rn, xn
end

batch_zip(channel, batchsize) = Channel() do c
    while true
        batch = collect(Iterators.take(channel, batchsize))
        if length(batch) == 0
            break
        end
        push!(c, (getindex.(batch, 1), getindex.(batch, 2)))
    end
end

batch(channel, batchsize) = Channel() do c
    while true
        batch = collect(Iterators.take(channel, batchsize))
        if length(batch) == 0
            break
        end
        push!(c, batch)
    end
end

macro ifund(exp)
    local e = :($exp)
    isdefined(e.args[1]) ? :($(e.args[1])) : :($(esc(exp)))
end

macro interrupts(ex)
    :(try $(esc(ex))
      catch e
      e isa InterruptException || rethrow()
      throw(e)
      end)
end

runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)
