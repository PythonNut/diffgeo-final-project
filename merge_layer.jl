using Flux
import Flux: children, mapchildren

struct Merge
    encoders
end

(m::Merge)(x) = vcat((e(x[:, i]) for (i, e) in enumerate(m.encoders))...)
Flux.children(x::Merge) = (x.encoders...,)
Flux.mapchildren(f, x::Merge) = Merge([Flux.mapchildren(f, e) for e in x.encoders])
