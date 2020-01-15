from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_commnet_args


if __name__ == '__main__':
    for i in range(8):
        args = get_common_args()
        if args.alg == 'coma':
            args = get_coma_args(args)
        elif args.alg == 'commnet_coma':
            args = get_commnet_args(args)
        else:
            args = get_mixer_args(args)
        env = StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir)
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        runner = Runner(env, args)
        if args.learn:
            runner.run(i)
        else:
            win_rate = runner.evaluate_sparse()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
