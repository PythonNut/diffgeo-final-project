using DataFrames
using CSV
using Dierckx
using RecursiveArrayTools
using ProgressMeter
using StatsBase
using DSP
using HDF5
using Rotations

include("util.jl")

function process(acc_csv, gyr_csv, mag_csv)
    # Loading files
    Tacc_raw, Vacc_raw = read_file_sampled(acc_csv)
    Tgyr_raw, Vgyr_raw = read_file_sampled(gyr_csv)
    Tmag_raw, Vmag_raw = read_file_sampled(mag_csv)

    sample_rate_ms = 10

    Tmag, Vmag = spline_sample(Tmag_raw/1000, Vmag_raw, sample_rate_ms/1000)
    Tgyr, Vgyr = spline_sample(Tgyr_raw/1000, Vgyr_raw, sample_rate_ms/1000)
    Tacc, Vacc = spline_sample(Tacc_raw/1000, Vacc_raw, sample_rate_ms/1000)

    mlen = minimum(length.([Tmag, Tgyr, Tacc]))

    # @assert Tacc == Tgyr == Tacc
    T = Tmag[1:mlen]

    # Filtering gravitational and magnetic fields
    Vgvt = low_pass(T, Vacc, 100)
    Vmag = low_pass(T, Vmag, 100)

    # Unifying gravitational and magnetic fields
    Rs = zeros(3, 3, length(T))
    for (i, t) in enumerate(T)
        gvt, mag = Vgvt[i,:], Vmag[i, :]
        gvtn = -normalize(gvt)
        magn = normalize(mag - proj(gvtn, mag))
        Rs[:,1,i] = magn
        Rs[:,2,i] = cross(gvtn, magn)
        Rs[:,3,i] = gvtn
    end

    # Fusing gyroscopic data w/ absolute sensors
    t = 0.0
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

    # Translating accelerations into global frame
    Cinv = permutedims(Cs, [2,1,3])
    Vacc_global = zeros(Vacc)
    for (i, t) in enumerate(T)
        Vacc_global[i,:] = Cinv[:,:,i] * Vacc[i,:]
    end

    # Filtering global accelerations
    Vacc_global_transient = high_pass(T, Vacc_global, 100)

    # Integrating accelerations to get velocity and filtering
    Vvel_global = integrate3d(T, Vacc_global_transient)
    Vvel_global_transient = high_pass(T, Vvel_global, 100)

    # Integrating velocities to get position and filtering
    Vpos_global = integrate3d(T, Vvel_global_transient)
    Vpos_global_transient = high_pass(T, Vpos_global, 100)
    return Cinv, Vpos_global_transient
end

function process_session(session)
    accel_csv = joinpath(session, "Accelerometer.csv")
    gyro_csv = joinpath(session, "Gyroscope.csv")
    mag_csv = joinpath(session, "Magnetometer.csv")

    @assert isfile(accel_csv)
    @assert isfile(gyro_csv)
    @assert isfile(mag_csv)

    R, x = process(accel_csv, gyro_csv, mag_csv)
    return R, x
end

function quaternize(Rs)
    ret = mapslices(x-> begin q=SPQuat(x); [q.x q.y q.z] end, Rs, (1,2))
    return reshape(ret, (3, size(Rs, 3)))'
end

function main()
    db = h5open("data.hdf5", "w")
    sessions = get_sessions()

    @showprogress 1 "Computing..." for session in sessions
        try
            R, x = process_session(session)
            db[replace(session, "/", "_") * "/rot"] = quaternize(R)
            db[replace(session, "/", "_") * "/pos"] = x
        catch e
            if !isa(e, InterruptException)
                println(session)
                println(e)
            else
                close(db)
                throw(e)
            end
        end
    end
    close(db)
end
