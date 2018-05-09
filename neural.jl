using DataFrames
using CSV
using HDF5
using Iterators
using Rotations

using Flux
using Flux.Tracker
using Flux: onehot, crossentropy, @epochs, throttle, argmax

import Flux: children
import Flux: mapchildren

include("util.jl")

db = h5open("data.hdf5", "r")
sessions = shuffle(get_sessions())

j = 100
k = 10

scanners = [LSTM(j, k) for _ in 1:6]
fc1 = Dense(6 * k, 6 * k)
out = Dense(6 * k, 6)

struct Merge
    encoders
end

(m::Merge)(x) = vcat((e(x[:, i]) for (i, e) in enumerate(m.encoders))...)
Flux.children(x::Merge) = (x.encoders...,)
Flux.mapchildren(f, x::Merge) = Merge([Flux.mapchildren(f, e) for e in x.encoders])

model = Chain(Merge(encoders),
              fc1,
              out,
              softmax)

function loss(xs, ys)
    l = mean(crossentropy.(model.(xs), ys))
    Flux.truncate!(model)
    return l
end

function accuracy(xs, ys)
    l = mean(argmax.(model.(xs)) .== argmax.(ys))
    Flux.truncate!(model)
    return
end

train_sessions = sessions[1:1700]
valid_sessions = sessions[1701:end]

function normalize_batch(R, x)
    R0 = inv(SPQuat(R[1,:]...))
    Rn = mapslices(r -> begin q=R0*SPQuat(r...); [q.x, q.y, q.z] end, R, 2)
    xn = mapslices(p->R0*p, x, 2)
    x0 = xn[1,:]
    xn = mapslices(x->x-x0, xn, 2)
    return Rn, xn
end

data_generator(sessions) = Channel() do c
    pairs = []
    t = 10
    keys = names(db)
    for session in sessions
        key = replace(session, "/", "_")
        if !(key in keys)
            continue
        end
        R, x = read(db[key * "/rot"]), read(db[key * "/pos"])
        for i in 1:j*t:size(R, 1) - j*t
            push!(pairs, (session, i))
        end
    end

    for (session, i) in shuffle(pairs)
        key = replace(session, "/", "_")
        R, x = read(db[key * "/rot"]), read(db[key * "/pos"])
        Rn, xn = normalize_batch(R[i:t:i+j*t-1,:], x[i:t:i+j*t-1,:])
        push!(c, (hcat(Rn, xn),
                  onehot(get_session_activity(session), 0:5)))
    end
end

batch(channel, batchsize) = Channel() do c
    while true
        batch = collect(Iterators.take(channel, batchsize))
        if length(batch) == 0
            break
        end
        push!(c, (getindex.(batch, 1), getindex.(batch, 2)))
    end
end

opt = ADAM(params(model))
batchsize = 10
evalcb = () -> @show loss(first(valid)...)

function main(epochs=1000)
    @epochs epochs begin
        data = batch(data_generator(train_sessions), batchsize)
        valid = batch(data_generator(valid_sessions), 1)
        Flux.train!(loss, data, opt)
        losses = [loss([x], [y]) for (x, y) in data_generator(valid_sessions)]
        accuracies = [accuracy([x], [y]) for (x, y) in data_generator(valid_sessions)]
        @show mean(losses)
        @show mean(accuracies)
    end
end

