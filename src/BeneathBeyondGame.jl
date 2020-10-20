module BeneathBeyondGame
    export GameEnv, GameSpec

    include("game.jl")
    const GameEnv = CubeEnv{5}
    const GameSpec = CubeSpec{5}

    module Training
        using AlphaZero
        import ..GameSpec
        # include("params.jl")
    end
end
