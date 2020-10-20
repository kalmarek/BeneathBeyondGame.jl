using AbstractAlgebra
import AlphaZero.GI

mutable struct BBCube{N} <: GI.AbstractGame
    board::BitArray{N}
    history::Vector{UInt16}
    # bb::Polymake.BeneathBeyond
end

function BBCube{N}() where N
    history = UInt16[]
    sizehint!(history, 2^N)
    board = falses(ntuple(_->2, N))
    return BBCube{N}(board, history) #, ...)
end

BBCube{N}(state::NamedTuple) where N = BBCube{N}(state.board, state.history)

board(g::BBCube) = g.board
history(g::BBCube) = g.history

GI.State(::Type{BBCube{N}}) where N = NamedTuple{(:board, :history),Tuple{BitArray{N},Vector{UInt16}}}

GI.Action(::Type{<:BBCube}) = Int

GI.two_players(::Type{<:BBCube}) = false

GI.actions(::Type{BBCube{N}}) where {N} = 1:2^N

GI.actions_mask(g::BBCube) = vec(.~(board(g)))

GI.current_state(g::BBCube) = (board=board(g), history=history(g))

GI.white_playing(::Type{<:BBCube}, state) = true

GI.game_terminated(g::BBCube) = all(board(g))

GI.white_reward(g::BBCube) = isempty(history(g)) ? Int(0) : -Int(sum(x->x^2, history(g)))

Base.@propagate_inbounds function Base.push!(g::BBCube{N}, n::Integer) where N
    checkbounds(board(g), n)

    g.board[n] = true
    push!(g.history, n)
    # Polymake.add_point!(g.bb, n)

    return g
end

GI.play!(g::BBCube, action::Integer) = push!(g, action)

GI.heuristic_value(g::BBCube) = isempty(history(g)) ? Int(0) : -Int(sum(history(g))) # Polymake.triangulation_size(g.bb)

#####
##### Machine Learning API
#####

function GI.vectorize_state(::Type{BBCube{N}}, state) where N
    res = zeros(Float32, 2^(N+1))
    @inbounds res[1:2^N] .= vec(state.board)
    @inbounds res[2^N+1:2^N+length(state.history)] .= state.history;
    return res
end

#####
##### Symmetries
#####

Base.@propagate_inbounds function Base.permutedims(cidx::CartesianIndex, perm::AbstractVector{<:Integer})
    @boundscheck all(i -> 0 < perm[i] <=length(cidx), 1:length(cidx))
    return CartesianIndex(ntuple(i->cidx[perm[i]], length(cidx)))
end

Base.permutedims(A::AbstractArray, σ::AbstractAlgebra.Perm) = permutedims(A, σ.d)
Base.permutedims(cidx::CartesianIndex, σ::AbstractAlgebra.Perm) = permutedims(cidx, σ.d)

Base.isone(p::AbstractAlgebra.Perm) = p.d == 1:length(p.d)

function action_on_gameactions(σ::AbstractAlgebra.Perm, cids, lids)
    return [lids[permutedims(cids[a], σ)] for a in vec(lids)]
end

function action_on_gamestate(
    state,
    σ::AbstractAlgebra.Perm;
    cids=CartesianIndices(state.board),
    lids=LinearIndices(state.board)
)
    p = action_on_gameactions(σ, cids, lids)
    return (
        (
            board = permutedims(state.board, σ),
            history = UInt16[p[h] for h in state.history]
        ),
        p
    )
end

function GI.symmetries(::Type{BBCube{N}}, state) where N
    cids = CartesianIndices(state.board)
    lids = LinearIndices(state.board)

    return [ action_on_gamestate(state, σ, cids=cids, lids=lids)
        for σ in AbstractAlgebra.Generic.SymmetricGroup(N) if !isone(σ)
    ]
end

#####
##### Interaction API
#####

function GI.action_string(::Type{BBCube{N}}, action) where N
    ci = CartesianIndices(ntuple(_->2, N))[action]
    return join(Tuple(ci) .- 1, "")
end

function GI.parse_action(::Type{BBCube{N}}, str::String) where N
    @assert length(str) == N
    ci = ntuple(n -> str[n] == "0" ? 1 : 2, N)
    return LinearIndices(ntuple(_->2, N))[ci]
end

function GI.read_state(::Type{BBCube{N}}) where N
    throw("Not Implemented")
end

function GI.render(g::BBCube{N}; with_position_names=true, botmargin=true) where N
    print(board(g))
    botmargin && print("\n")
end


