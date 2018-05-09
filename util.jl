using DataFrames
using CSV
using Dierckx
using RecursiveArrayTools
using StatsBase
using DSP

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
    dataset = "public_dataset"
    users = readdir(dataset)
    session_list = []

    # loop through all of the users
    for i = 1:length(users)
        user = string(dataset, "/", users[i])
        # check that the user is a directory
        if isdir(user)
            sessions = readdir(user)
            # loop through all sessions for a user
            for child in sessions
                session = joinpath(user, child)
                # check that the session is a directory
                if isdir(session)
                    push!(session_list, session)
                end
            end
        end
    end

    return session_list
end

function get_session_activity(session)
    csv = joinpath(session, "Activity.csv")
    @assert isfile(csv)
    df = CSV.read(csv, datarow=1)
    activities = unique(df[:9])
    @assert length(activities) == 1
    return activities[1] % 6
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
