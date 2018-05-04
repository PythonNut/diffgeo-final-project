using DataFrames
using CSV

dataset = string(pwd(),"/","public_dataset")
users = readdir(dataset)

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

# loop through all of the users
for i = 1:length(users)
	user = string(dataset, "/", users[i])
	# check that the user is a directory
	if isdir(user)
		sessions = readdir(user)
		# loop through all sessions for a user
		for j = 1:length(sessions)
			session = string(user, "/", sessions[j])
			# check that the session is a directory
			if isdir(session)
				accel_csv = string(session, "/", "Accelerometer.csv")
				gyro_csv = string(session, "/", "Gyroscope.csv")
				mag_csv = string(session, "/", "Magnetometer.csv")
			end
		end
	end
end




