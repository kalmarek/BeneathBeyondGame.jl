import AlphaZero.GI

struct CubeSpec{N} <: GI.AbstractGameSpec end

board_shape(::Type{CubeSpec{N}}) where {N} = ntuple(_ -> 2, N)

mutable struct CubeEnv{N, B<:Polymake.BeneathBeyond} <: GI.AbstractGameEnv
    board::BitArray{N}
    history::Vector{UInt16}
    algo::B
end

board(g::CubeEnv) = g.board
history(g::CubeEnv) = g.history
algo(g::CubeEnv) = g.algo

function vertices_cube(n::Integer, lo=-1, up=1)
    verts = Iterators.product(ntuple(_->lo:up-lo:up, n)...)
    res = Array{Int}(undef, 2^n, n+1)
    res[:, 1] .= 1
    for (i,v) in enumerate(verts)
        res[i, 2:end] .= v
    end
    return res
end

function GI.init(
    ::CubeSpec{N},
    state = (board = falses(board_shape(CubeSpec{N})),
             history = UInt16[],
             algo = Polymake.BeneathBeyond(Polymake.Matrix{Polymake.Rational}(vertices_cube(N)))
             ),
) where {N}
    sizehint!(state.history, 2^N)
    return CubeEnv(copy(state.board), copy(state.history), deepcopy(state.algo))
end

GI.spec(::CubeEnv{N}) where {N} = CubeSpec{N}()

GI.two_players(::CubeSpec) = false

function GI.set_state!(g::CubeEnv, state)
    g.board = copy(state.board)
    g.history = copy(state.history)
    g.algo = deepcopy(state.algo)
end

#####
##### Game API
#####

GI.actions(::CubeSpec{N}) where {N} = 1:2^N

GI.actions_mask(g::CubeEnv) = vec(.~(board(g)))

GI.current_state(g::CubeEnv) =
    (board = copy(board(g)), history = copy(history(g)), algo = deepcopy(algo(g)))

GI.white_playing(::CubeEnv) = true

GI.game_terminated(g::CubeEnv) = all(board(g))

GI.white_reward(g::CubeEnv) =
    isempty(history(g)) ? Inf : -log(Polymake.triangulation_size(algo(g)))

Base.@propagate_inbounds function Base.push!(g::CubeEnv, n::Integer)
    @boundscheck checkbounds(board(g), n)

    g.board[n] = true
    push!(g.history, n)
    # @info "adding point $n:" g.algo.rays[n, :]
    @inbounds Polymake.add_point!(g.algo, n)

    return g
end

GI.play!(g::CubeEnv, action) = push!(g, action)

GI.heuristic_value(g::CubeEnv) =
    isempty(history(g)) ? Inf : -log(Polymake.triangulation_size(algo(g)))

#####
##### Machine Learning API
#####

function GI.vectorize_state(::CubeSpec{N}, state) where {N}
    res = zeros(Float32, 2^(N + 1) + 1)
    @inbounds res[1:2^N] .= vec(state.board)
    @inbounds res[2^N+1:2^N+length(state.history)] .= state.history
    @inbounds res[2^(N + 1) + 1] = Polymake.triangulation_size(state.algo)
    return res
end

#####
##### Symmetries
#####

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
    # algo_p = deepcopy(state.algo)
    # TODO : What to do with state? I don't want to compute triangulation again...

    state_p =
        (board = permutedims(state.board, σ), history = UInt16[p[h] for h in state.history])
    return (state_p, convert(Vector{Int}, p))
end

function GI.symmetries(::CubeSpec{N}, state) where {N}
    cids = CartesianIndices(state.board)
    lids = LinearIndices(state.board)

    return Tuple{typeof(state), Vector{Int}}[
        # action_on_gamestate(state, σ, cids = cids, lids = lids) for σ in Iterators.rest(AllPerms(N), 1)
        ]
end

#####
##### Interaction API
#####

using Crayons

function GI.action_string(::CubeSpec{N}, action) where {N}
    ci = CartesianIndices(board_shape(CubeSpec{N}))[action]
    k = ceil(Int, log10(2^N))
    return rpad("$action", k + 2) * " | " * join(Tuple(ci) .- 1, "")
end

function GI.parse_action(::CubeSpec{N}, str) where {N}
    if length(str) <= ceil(log10(2^N))
        k = parse(Int, str)
        return k
    else
        ci = map(x -> (x == '0' ? 1 : 2), collect(str)[1:N])
        k = getindex(LinearIndices(board_shape(CubeSpec{N})), ci...)
        return k
    end
end

function GI.read_state(::CubeSpec{N}) where {N}
    throw("Not Implemented")
end

function GI.render(g::CubeEnv{N}; botmargin = true) where {N}

    st = GI.current_state(g)
    amask = GI.actions_mask(g)

    for action in GI.actions(GI.spec(g))
        color = amask[action] ? crayon"bold fg:light_gray" : crayon"fg:dark_gray"
        println(color, GI.action_string(GI.spec(g), action), crayon"reset")
    end

    println("current value: ", GI.heuristic_value(g))
    println("triangulation size: ", Polymake.triangulation_size(algo(g)))

    botmargin && print("\n")
end
