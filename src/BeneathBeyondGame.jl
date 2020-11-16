module BeneathBeyondGame
    export GameEnv, GameSpec

    import Polymake

    include("game.jl")
    const GameEnv = CubeEnv{5}
    const GameSpec = CubeSpec{5}

    module Training
        using AlphaZero
        import ..GameSpec
        import ..CubeSpec
        include("params.jl")
    end
end
