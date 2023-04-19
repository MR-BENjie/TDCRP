import argparse
import rlbench.gym
import gym
import torch
from spinup import sac_pytorch
from spinup.utils.run_utils import setup_logger_kwargs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="plug_charger_in_power_supply")

    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()



    task = "%s-state-v0"%args.env
    logger_kwargs = setup_logger_kwargs(args.env+"-"+args.exp_name, args.seed, data_dir= "./checkpoint")

    torch.set_num_threads(torch.get_num_threads())
    sac_pytorch(env_fn= lambda : gym.make(task, render_mode='human'), ac_kwargs= dict(hidden_sizes=[args.hid]*args.l), gamma= args.gamma, seed = args.seed, epochs= args.epochs, logger_kwargs= logger_kwargs)

    print('Done')

