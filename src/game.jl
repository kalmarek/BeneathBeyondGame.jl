import AbstractAlgebra
import AlphaZero.GI

struct CubeSpec{N} <: GI.AbstractGameSpec end

board_shape(::Type{CubeSpec{N}}) where {N} = ntuple(_ -> 2, N)

mutable struct CubeEnv{N} <: GI.AbstractGameEnv
    board::BitArray{N}
    history::Vector{UInt16}
    # bb::Polymake.BeneathBeyond
end

board(g::CubeEnv) = g.board
history(g::CubeEnv) = g.history

function GI.init(
    ::CubeSpec{N},
    state = (board = falses(board_shape(CubeSpec{N})), history = UInt16[]),
) where {N}
    sizehint!(state.history, 2^N)
    return CubeEnv{N}(state.board, state.history)
end

GI.spec(::CubeEnv{N}) where {N} = CubeSpec{N}()

GI.two_players(::CubeSpec) = false

#####
##### Game API
#####

GI.actions(::CubeSpec{N}) where {N} = 1:2^N

GI.actions_mask(g::CubeEnv) = vec(.~(board(g)))

GI.current_state(g::CubeEnv) = (board = board(g), history = history(g))

GI.white_playing(::CubeEnv) = true

GI.game_terminated(g::CubeEnv) = all(board(g))

GI.white_reward(g::CubeEnv) =
    isempty(history(g)) ? 0.0 : -Float64(sum(x -> x^2, history(g)))

Base.@propagate_inbounds function Base.push!(g::CubeEnv, n::Integer)
    @boundscheck checkbounds(board(g), n)

    g.board[n] = true
    push!(g.history, n)
    # Polymake.add_point!(g.bb, n)

    return g
end

GI.play!(g::CubeEnv, action) = push!(g, action)

GI.heuristic_value(g::CubeEnv) = isempty(history(g)) ? Int(0) : -Int(sum(history(g))) # Polymake.triangulation_size(g.bb)

#####
##### Machine Learning API
#####

function GI.vectorize_state(::CubeSpec{N}, state) where {N}
    res = zeros(Float32, 2^(N + 1))
    @inbounds res[1:2^N] .= vec(state.board)
    @inbounds res[2^N+1:2^N+length(state.history)] .= state.history
    return res
end

#####
##### Symmetries
#####

function Base.permutedims(
    cidx::CartesianIndex,
    perm::AbstractVector{<:Integer},
)
    @boundscheck all(i -> 0 < perm[i] <= length(cidx), 1:length(cidx))
    return CartesianIndex(ntuple(i -> cidx[perm[i]], length(cidx)))
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
    cids = CartesianIndices(state.board),
    lids = LinearIndices(state.board),
)
    p = action_on_gameactions(σ, cids, lids)
    state_p =
        (board = permutedims(state.board, σ), history = UInt16[p[h] for h in state.history])
    return (state_p, p)
end

function GI.symmetries(::CubeSpec{N}, state) where {N}
    cids = CartesianIndices(state.board)
    lids = LinearIndices(state.board)

    return [
        @inbounds action_on_gamestate(state, σ, cids = cids, lids = lids)
        for σ in AbstractAlgebra.SymmetricGroup(N) if !isone(σ)
    ]
end

#####
##### Interaction API
#####

using Crayons

function GI.action_string(::CubeSpec{N}, action) where {N}
    ci = CartesianIndices(board_shape(CubeSpec{N}))[action]
    return join(Tuple(ci) .- 1, "")
end

function GI.parse_action(::CubeSpec{N}, str) where {N}
    @assert length(str) == N
    ci = tuple(map(x -> (x == "0" ? 1 : 2), str))
    return LinearIndices(board_shape(CubeSpec{N}))[ci]
end

function GI.read_state(::CubeSpec{N}) where {N}
    throw("Not Implemented")
end

function GI.render(g::CubeEnv{N}; with_position_names = true, botmargin = true) where {N}
    st = GI.current_state(g)
    amask = GI.actions_mask(g)
    k = ceil(Int, log10(2^N))
    for action in GI.actions(GI.spec(g))
        color =
        amask[action] ? crayon"bold fg:light_gray" : crayon"fg:dark_gray"
        with_position_names && print(color, rpad("$action", k + 2), " | ")
        println(color, GI.action_string(GI.spec(g), action), crayon"default")
    end
    botmargin && print("\n")
end
