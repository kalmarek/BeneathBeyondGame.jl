import AlphaZero.GI

struct CapsSpec{N, P} <: GI.AbstractGameSpec end

board_shape(::Type{CapsSpec{N, P}}) where {N, P} = ntuple(_ -> P, N)

mutable struct CapsEnv{N, P} <: GI.AbstractGameEnv
    board::BitArray{N}
    history::Vector{UInt16}
    # possible_moves::Vector{UInt16}
end

board(g::CapsEnv) = g.board
history(g::CapsEnv) = g.history

function GI.init(
    ::CapsSpec{N, P},
    state = (board = falses(board_shape(CapsSpec{N, P})), history = UInt16[]),
) where {N, P}
   # Todo: Better upper bound
    sizehint!(state.history, P^N-N)
    return CapsEnv{N, P}(copy(state.board), copy(state.history))
end

GI.spec(::CapsEnv{N, P}) where {N, P} = CapsSpec{N, P}()

GI.two_players(::CapsSpec) = false

function GI.set_state!(g::CapsEnv, state)
    g.board = copy(state.board)
    g.history = copy(state.history)
end

#####
##### Game API
#####

GI.actions(::CapsSpec{N}) where {N} = 1:3^N

GI.actions_mask(g::CapsEnv) = vec(.~(board(g)))

GI.current_state(g::CapsEnv) = (board = copy(board(g)), history = copy(history(g)))

GI.white_playing(::CapsEnv) = true

GI.game_terminated(g::CapsEnv) = all(board(g))

GI.white_reward(g::CapsEnv{N,P}) where {N,P} =
    Float64(length(history(g))) + log(1 + P^N - sum(board(g)))

function third_point_on_line(p1, p2, P)
   # q1 = p1 .- 1;
   # q2 = p2 .- 1;
   # qr = ((q1 .+ q2) .* (-1))
   # qr = qr .+ 3
   # qr = qr .% 3
   # qr = qr .+ 1
   # return qr
   return mod.(-1 .*(p1 .+ p2), Ref(Base.OneTo(P)))
end

Base.@propagate_inbounds function Base.push!(g::CapsEnv{N, P}, n::Integer) where {N, P}
    @boundscheck checkbounds(board(g), n)

    g.board[n] = true
    q = Tuple(CartesianIndices(board(g))[n])
    for pIndex in history(g)
       p = Tuple(CartesianIndices(board(g))[pIndex])
       pq = third_point_on_line(p, q, P)
       g.board[pq...] = true
    end

    push!(g.history, n)
    return g
end

GI.play!(g::CapsEnv, action) = push!(g, action)

GI.heuristic_value(g::CapsEnv) = GI.white_reward(g)

#####
##### Machine Learning API
#####

function GI.vectorize_state(::CapsSpec{N,P}, state) where {N, P}
    res = zeros(Float32, 2*(P^N))
    @inbounds res[1:P^N] .= vec(state.board)
    @inbounds res[P^N+1:P^N+length(state.history)] .= state.history
    return res
end

#####
##### Symmetries
#####
#=
struct AllPerms{T<:Integer}
    all::Int
    c::Vector{T}
    elts::Vector{T}

    AllPerms(n::T) where T = new{T}(factorial(n), ones(T, n), collect(1:n))
end

Base.eltype(::Type{AllPerms{T}}) where T = Vector{T}
Base.length(A::AllPerms) = A.all

@inline Base.iterate(A::AllPerms) = (A.elts, 1)

@inline function Base.iterate(A::AllPerms, count)
    count >= A.all && return nothing

    k,n = 0,1

    @inbounds while true
        if A.c[n] < n
            k = ifelse(isodd(n), 1, A.c[n])
            A.elts[k], A.elts[n] = A.elts[n], A.elts[k]
            A.c[n] += 1
            return A.elts, count + 1
        else
            A.c[n] = 1
            n += 1
        end
    end
end

Base.@propagate_inbounds function Base.permutedims(
    cidx::CartesianIndex{N},
    perm::AbstractVector{<:Integer},
) where N
    @boundscheck length(perm) == N
    @boundscheck all(i -> 0 < perm[i] <= length(cidx), 1:length(cidx))
    return CartesianIndex(ntuple(i -> @inbounds(cidx[perm[i]]), Val(N)))
end

function action_homomorphism(σ::AbstractVector{<:Integer}, cids, lids)
    return Int[lids[permutedims(cids[a], σ)] for a in vec(lids)]
end

function action_on_gamestate(
    state,
    σ::AbstractVector{<:Integer};
    cids = CartesianIndices(state.board),
    lids = LinearIndices(state.board),
)
    p = action_homomorphism(invperm(σ), cids, lids)
    state_p =
        (board = permutedims(state.board, σ), history = UInt16[p[h] for h in state.history])
    return (state_p, convert(Vector{Int}, p))
end

function GI.symmetries(::CapsSpec{N}, state) where {N}
    cids = CartesianIndices(state.board)
    lids = LinearIndices(state.board)

    return Tuple{typeof(state), Vector{Int}}[
        action_on_gamestate(state, σ, cids = cids, lids = lids) for σ in Iterators.rest(AllPerms(N), 1)
        ]
end

=#
#####
##### Interaction API
#####

using Crayons

function GI.action_string(::CapsSpec{N,P}, action) where {N, P}
    ci = CartesianIndices(board_shape(CapsSpec{N, P}))[action]
    return join(Tuple(ci) .- 1, "")
end

function GI.parse_action(::CapsSpec{N, P}, str) where {N, P}
    if length(str) <= ceil(log10(P^N))
        k = parse(Int, str)
        return k
    else
        ci = map(c -> parse(Int, c) + 1, collect(str)[1:N])
        k = LinearIndices(board_shape(CapsSpec{N, P}))[ci...]
        return k
    end
end

function GI.read_state(::CapsSpec{N}) where {N}
    throw("Not Implemented")
end

function GI.render(g::CapsEnv{N, P}; with_position_names = true, botmargin = true) where {N, P}

    st = GI.current_state(g)
    amask = GI.actions_mask(g)
    k = ceil(Int, log10(P^N))
    for action in GI.actions(GI.spec(g))
        color =
        amask[action] ? crayon"bold fg:light_gray" : crayon"fg:dark_gray"
        with_position_names && print(color, rpad("$action", k + 2), " | ")
        println(color, GI.action_string(GI.spec(g), action), crayon"reset")
    end
    println("current value: ", GI.heuristic_value(g), "\n")
    botmargin && print("\n")
end
