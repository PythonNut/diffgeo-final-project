using HDF5
using Iterators
using ProgressMeter
using ValueHistories
using BSON: @save, @load
using StatsBase: mode

using Flux
using Flux.Tracker
using Flux: binarycrossentropy, @epochs, throttle, testmode!, loadparams!

include("util.jl")
include("merge_layer.jl")

db = h5open("data.hdf5", "r")
sessions = shuffle(filter_sessions(get_sessions(), db))

j = 100
k = 15

encoders = [LSTM(j, k) for _ in 1:6]
fc1 = Dense(6 * k, 6 * k)
out = Dense(6 * k, 1, σ)

model = Chain(Merge(encoders), fc1, out, x->x[1])

function loss(xs, ys, m=model)
    result = mean(binarycrossentropy(m(xs), ys))
    Flux.reset!(m)
    return result
end

function accuracy(xs, ys, m=model)
    result = mean((m(xs) > 0.5) == ys)
    Flux.reset!(m)
    return result
end

train_sessions = sessions[1:1700]
valid_sessions = sessions[1701:end]

function session_accuracy()
    mean(mean(accuracy(t..., predict)
              for t in data_generator([session], false)) > 0.5
         for session in valid_sessions)
end

data_generator(sessions, progress=true) = Channel() do c
    t = 10
    pairs = []
    for session in sessions
        R, x, _ = read_session(session)
        for i in 1:j*t:size(R, 1) - j*t
            push!(pairs, (session, i))
        end
    end

    p = Progress(length(pairs))
    for (session, i) in shuffle(pairs)
        R, x, activity = read_session(session)
        Rn, xn = normalize_batch(R[i:t:i+j*t-1,:], x[i:t:i+j*t-1,:])
        push!(c, (hcat(Rn, xn), walking(activity)))
        if progress
            next!(p)
        end
    end
end

opt = ADAM(params(model))
batchsize = 10
@ifund data = collect(data_generator(train_sessions))
@ifund valid = collect(data_generator(valid_sessions))

function mytrain!(loss, data, opt; cb = () -> ())
    cb = runall(cb)
    opt = runall(opt)
    @showprogress for d in data
        l = loss(d...)
        isinf(l) && error("Loss is Inf")
        isnan(l) && error("Loss is NaN")
        @interrupts back!(l)
        opt()
        cb() == :stop && break
    end
end

predict = mapleaves(Flux.Tracker.data, model)
testmode!(predict, true)

function save()
    weights = Tracker.data.(params(model))
    fname = "models/model-$(now()).bson"
    info("Saving to $(fname)...")
    @save fname opt weights
end

function load(fname)
    global opt
    global model
    info("Loading $(fname)...")
    @load fname opt weights
    Flux.loadparams!(model, weights)
end

function resume()
    fname = glob("models/*")[end]
    load(fname)
end

function main(epochs=30)
    @epochs epochs begin
        mytrain!(loss, data, opt, cb=throttle(save, 300))

        info("Calculating loss")
        losses = mean(loss(t..., predict) for t in valid)
        @show losses

        info("Calculating accuracy")
        accuracies = mean(accuracy(t..., predict) for t in valid)
        @show accuracies

        info("Calculating session accuracy")
        @show session_accuracy()
    end
end
