using DataFrames
using CSV
using Plots
using ProgressMeter
using DSP
using Dierckx

pyplot()

file = "public_dataset/100669/100669_session_9/Accelerometer.csv"
df = CSV.read(file, header=collect(1:7))
fs = 1000;

T, Xa, Ya, Za = Int64[], Float64[], Float64[], Float64[]

# read the data from the csv file
for i in 1:size(df, 1)
    (t, _, _, x, y, z, _) = Array(df[i,:])
    if length(T) > 0 && T[end] == t
        continue
    end
    push!(Xa, x)
    push!(Ya, y)
    push!(Za, z)
    push!(T, t)
end

function supersample(T, Ts, xs)
    xsi = Spline1D(T, xs)
    return xsi(Ts)
end

function low_pass(T, xs)
    filt = digitalfilter(Lowpass(.1, fs=fs), FIRWindow(hamming(10001)))
    xsf = filtfilt(filt, xs)
    return xsf
end

T = (T - T[1])/fs
Tms = collect(T[1]:0.001:T[end])

Xas = supersample(T, Tms, Xa)
Yas = supersample(T, Tms, Ya)
Zas = supersample(T, Tms, Za)

Xaf = low_pass(Tms, Xas)
Yaf = low_pass(Tms, Yas)
Zaf = low_pass(Tms, Zas)

plot(T, Xa)
plot!(Tms, Xas)
plot!(Tms,Xaf)

