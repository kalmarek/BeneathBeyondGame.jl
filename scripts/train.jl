length(ARGS) != 1 && throw("You need to specify the dimension")
const N = parse(Int, ARGS[1])

using Dates
using AlphaZero
using BeneathBeyond

session = let N = N
    experiment = BeneathBeyondGame.Training.experiment(N)
    session_dir = joinpath("sessions", "bb_$N", string(now()))
    @info session_dir

    session = UserInterface.Session(
        experiment,
        dir = session_dir,
        autosave = true,
        save_intermediate = false,
    )

    @time UserInterface.resume!(session)
    session
end
