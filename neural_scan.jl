using DataFrames
using CSV
using HDF5
using Iterators
using ProgressMeter

@everywhere using Flux
@everywhere using Flux.Tracker
@everywhere using Flux: onehot, crossentropy, @epochs, throttle, argmax

import Flux: children
import Flux: mapchildren

include("util.jl")

db = h5open("data.hdf5", "r")
sessions = shuffle(filter_sessions(get_sessions(), db))

@everywhere j = 100
@everywhere k = 20

@everywhere rscanner = Chain(Dense(3, k), LSTM(k, k))
@everywhere xscanner = Chain(Dense(3, k), LSTM(k, k))
@everywhere fc1 = Dense(2*k, 2*k)
@everywhere encoder = Dense(2*k, 6)

# struct Merge
#     encoders
# end

# (m::Merge)(x) = vcat((e(x[:, i]) for (i, e) in enumerate(m.encoders))...)
# Flux.children(x::Merge) = (x.encoders...,)
# Flux.mapchildren(f, x::Merge) = Merge([Flux.mapchildren(f, e) for e in x.encoders])

@everywhere function model(R::Matrix{Float64}, x::Matrix{Float64})
    rstate, xstate = zeros(k), zeros(k)
    for i in size(R, 1)
        rstate = rscanner(R[i,:])
    end
    for i in size(x, 1)
        xstate = xscanner(x[i,:])
    end

    Flux.reset!(rscanner)
    Flux.reset!(xscanner)
    return softmax(encoder(fc1(vcat(rstate, xstate))))
end

function loss(R::Matrix{Float64},
              x::Matrix{Float64},
              y::Flux.OneHotVector)
    return crossentropy(model(R, x), y)
end

function loss(Rs::Vector{Matrix{Float64}},
              xs::Vector{Matrix{Float64}},
              ys::Vector{Flux.OneHotVector})
    err = @parallel (+) for i in 1:length(Rs)
        crossentropy(model(Rs[i], xs[i]), ys[i])
    end

    return err/length(Rs)
end

function accuracy(xs, ys)
    err = @parallel (+) for i in 1:length(Rs)
        crossentropy(model(Rs[i], xs[i]), ys[i])
    end
    return err/length(xs)
end

train_sessions = sessions[1:1700]
valid_sessions = sessions[1701:end]

data_generator(sessions) = Channel() do c
    pairs = []
    t = 10
    keys = names(db)
    for session in sessions
        R, x, _ = read_session(session)
        for i in 1:j*t:size(R, 1) - j*t
            push!(pairs, (session, i))
        end
    end

    @showprogress for (session, i) in shuffle(pairs)
        R, x, activity = read_session(session)
        Rn, xn = normalize_batch(R[i:t:i+j*t-1,:], x[i:t:i+j*t-1,:])
        push!(c, (Rn, xn, onehot(activity, 0:5)))
    end
end

batch(channel, batchsize) = Channel() do c
    while true
        batch = collect(Iterators.take(channel, batchsize))
        if length(batch) == 0
            break
        end
        push!(c, (getindex.(batch, 1), getindex.(batch, 2), getindex.(batch, 3)))
    end
end

opt = ADAM(params(xscanner, rscanner, fc1, encoder))
batchsize = 32

function batch_loss(sessions)
    return mean(loss(t...) for t in batch(data_generator(sessions), batchsize))
end

function main(epochs=1000)
    @epochs epochs begin
        data = batch(data_generator(train_sessions), batchsize)
        # valid = batch(data_generator(valid_sessions), 1)
        Flux.train!(loss, data, opt)
        # accuracies = [accuracy([x], [y]) for (x, y) in data_generator(valid_sessions)]
        @show batch_loss(valid_sessions)
        @show batch_loss(shuffle(train_sessions)[1:600])
        # @show mean(accuracies)
    end
end
