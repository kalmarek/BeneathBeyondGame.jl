import AlphaZero.GI

mutable struct BBCube{N} where N <: GI.AbstractGame
    board::BitArray{N}
    history::Vector{Int}
    bb::Polymake.BeneathBeyond
end

function BBCube{N}() where N
    history = Int[]
    sizehint!(h, 2^N)
    board = falses(ntuple(x->2, N))
    return BBCube{N}(board, history, ...)
end

function BBCube{N}(state::Tuple) where N
    history, bb = state
    board = falses(ntuple(x->2, N))
    board[history] = true
    return BBCube{N}(board, history, bb)
end

history(g::BBCube) = g.history
board(g::BBCube) = g.board


GI.two_players(::Type{<:BBCube}) = false

GI.State(::Type{<:BBCube}) = Vector{Int}

GI.Action(::Type{<:BBCube}) = Int

GI.actions(::Type{BBCube{N}}) where {N} = 1:2^n

GI.white_playing(::Type{<:BBCube}, state) = true

GI.white_reward(g::BBCube) = throw("Not Implemented")

GI.current_state(g::BBCube) = (copy(history(g)), copy(bb))

GI.game_terminated(g::BBCube) = all(board(g))
GI.actions_mask(g::BBCube) = .~(board(g))

function GI.play!(g::BBCube, action::Integer)
    g.board[action] = true
    push!(g.history, action)
    Polymake.add_point!(g.bb, action)
    return g
end

GI.heuristic_value(g::BBCube) = Polymake.triangulation_size(g.bb)

# TODO: GI.symmetries

#####
##### ML interface
#####

function GI.vectorize_state(::Type{BBCube{N}}, state) where N
    res = zeros(Float32, 2^N)
    res[1:length(state)] .= state;
    return res
end



function GI.action_string(::Type{BBCube{N}}, action) where N
    board = trues(ntuple(_->2, N))
    ci = CartesianIndices(board)[action]
    return join(Tuple(ci) .- 1, "")
end

function GI.parse_action(::Type{BBCube{N}}, str::String) where N
    @assert length(str) == N
    board = trues(ntuple(_->2, N))
    ci = ntuple(n -> str[n] == "0" ? 1 : 2, N)
    return LinearIndices(board)[Tuple(ci)]
end

function GI.read_state(::Type{BBCube{N}}) where N
    throw("Not Implemented")
end




