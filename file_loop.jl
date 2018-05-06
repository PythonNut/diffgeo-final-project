using DataFrames
using CSV

dataset = string(pwd(),"/","public_dataset")
users = readdir(dataset)

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
            if !contains(session, "session")
                for d in readdir(session)
                    mv(joinpath(session, d), joinpath(user, d))
                end
                rm(session)
            end
            if isdir(session)
                accel_csv = string(session, "/", "Accelerometer.csv")
                gyro_csv = string(session, "/", "Gyroscope.csv")
                mag_csv = string(session, "/", "Magnetometer.csv")
                # println(accel_csv)
            end
        end
    end
end
