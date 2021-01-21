Network = NetLib.SimpleNet

netparams() = NetLib.SimpleNetHP(width = 100, depth_common = 4, use_batch_norm = false)

self_play() = SelfPlayParams(
    sim = SimParams(
        num_games = 2000,
        num_workers = 4,
        use_gpu = false,
        reset_every = 16,
        flip_probability = 0.0,
        alternate_colors = false,
    ),
    mcts = MctsParams(
        num_iters_per_turn = 50,
        cpuct = 1.0,
        temperature = ConstSchedule(0.0),
        dirichlet_noise_ϵ = 0.0,
        dirichlet_noise_α = 1.0,
    ),
)

arena() = ArenaParams(
    sim = SimParams(
        num_games = 100,
        num_workers = 10,
        use_gpu = false,
        reset_every = 1,
        flip_probability = 0.0,
        alternate_colors = true,
    ),
    mcts = self_play().mcts,
    update_threshold = 0.00,
)

learning(; l2_regulariation = 1e-3, lr = 5e-3, batch_size = 32) = LearningParams(
    use_gpu = false,
    samples_weighing_policy = LOG_WEIGHT,
    l2_regularization = l2_regulariation,
    optimiser = Adam(lr = lr),
    batch_size = batch_size,
    loss_computation_batch_size = 2048,
    nonvalidity_penalty = 1.0,
    min_checkpoints_per_epoch = 0,
    max_batches_per_checkpoint = 5_000,
    num_checkpoints = 1,
)

params(; num_iters = 10, kwargs...) = Params(
    arena = arena(),
    self_play = self_play(),
    learning = learning(; kwargs...),
    num_iters = num_iters,
    memory_analysis = nothing,
    ternary_rewards = false,
    use_symmetries = true,
    mem_buffer_size = PLSchedule(800_000),
)

benchmark_sim() = SimParams(arena().sim; num_games = 400, num_workers = 100)

benchmark() = [Benchmark.Duel(
    Benchmark.Full(self_play().mcts),
    Benchmark.MctsRollouts(self_play().mcts),
    benchmark_sim(),
)]

experiment(N::Integer; num_iters = 10) = Experiment(
    "cube_$N",
    CubeSpec{N}(),
    params(; num_iters = num_iters, l2_regulariation = 1e-3, lr = 5e-3, batch_size = 64),
    Network,
    netparams(),
    benchmark = benchmark(),
)
