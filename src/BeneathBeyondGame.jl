module BeneathBeyondGame
    export GameEnv, GameSpec

    include("game.jl")
    const GameEnv = CubeEnv{5}
    const GameSpec = CubeSpec{5}

    module Training
        using AlphaZero
        import ..GameSpec
        import ..CubeSpec
        include("params.jl")
    end
    
    export CapsEnv, CapsSpec
    include("caps_game.jl")
    const CapsGameEnv = CapsEnv{5}
    const CapsGameSpec = CapsSpec{5}
    module Training2
        using AlphaZero
        import ..CapsGameSpec
        import ..CapsSpec
        include("caps_params.jl")
    end
end
