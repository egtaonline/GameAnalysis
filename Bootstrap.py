#! /usr/bin/env python2.7

import RandomGames as RG

from GameIO import read, to_JSON_str
from Subgames import subgame
from Regret import regret
from Nash import mixed_nash, replicator_dynamics, pure_nash
from Reductions import deviation_preserving_reduction as DPR

from sys import stdin
from os import listdir
from os.path import join, exists, dirname
from argparse import ArgumentParser
from random import sample
from copy import copy
from json import dumps
from functools import partial

import numpy as np

def subsample(game, num_samples):
	"""
	Returns a game with a random subset of the input game's payoff samples.

	Note: this is not intended for use on games with different numbers of 
	samples per profile.
	"""
	sg = copy(game)
	sg.sample_values = map(lambda p: p[:,:,sample(range(p.shape[-1]), \
			num_samples)], sg.sample_values)
	sg.reset()
	sg.max_samples = num_samples
	return sg


def pre_aggregate(game, count):
	"""
	Aggregates samples to produce a game with |samples| / count samples.

	Note: this is not intended for use on games with different numbers of 
	samples per profile.
	"""
	agg = copy(game)
	sv = agg.sample_values
	sv = np.swapaxes(sv, 0, -1)
	np.random.shuffle(sv)
	sv = np.swapaxes(sv, 0, -1)
	shape = list(sv.shape)
	samples = shape[-1]
	if samples % count != 0:
		sv = sv[:,:,:,:samples - (samples % count)]
	shape.append(count)
	shape[-2] /= count
	sv = np.reshape(sv, shape)
	sv = np.average(sv, axis=-1)
	agg.sample_values = sv
	agg.reset()
	agg.min_samples = agg.max_samples = shape[-2]
	return agg


def bootstrap(game, equilibrium, statistic=regret, method="resample", \
				method_args=[], points=1000):
	"""
	Returns a bootstrap distribution for the statistic.

	To run a resample regret boostrap:
		bootstrap(game, eq)
	To run a single-sample regret bootstrap: 
		bootstrap(game, eq, method='single_sample')
	To run a replicator dynamics bootstrap:
		bootstrap(game, eq, replicator_dynamics)
	"""
	boot_dstr = []
	method = getattr(game, method)
	for i in range(points):
		method(*method_args)
		boot_dstr.append(statistic(game, equilibrium))
	game.reset()
	boot_dstr.sort()
	return boot_dstr


def bootstrap_experiment(base_game_func, noisy_game_func, statistic=regret, \
		num_games=1000, stdevs=[.2,1.,5.,25.], sample_sizes=[5,10,20,50,100, \
		200,500], equilibrium_search=mixed_nash, bootstrap_args=[]):
	results = [{s:{} for s in stdevs} for i in range(num_games)]
	for i in range(num_games):
		base_game = base_game_func()
		for stdev in stdevs:
			sample_game = noisy_game_func(base_game, stdev, sample_sizes[-1])
			for sample_size in sample_sizes:
				subsample_game = subsample(sample_game, sample_size)
				equilibria = equilibrium_search(subsample_game)
				results[i][stdev][sample_size] = [ \
					{ \
						"profile" : eq,
						"statistic" : statistic(base_game, eq),
						"bootstrap" : bootstrap(subsample_game, eq, statistic, \
												*bootstrap_args)
					} for eq in equilibria]
	return results


def parse_args():
	parser = ArgumentParser()
	parser.add_argument("game_func", type=str, default="", choices=\
						["","congestion","local_effect","uniform_symmetric"], \
						help="Specifies the function generating random base "+\
						"games. If empty, the script will look for a file "+\
						"with simulated game data on stdin.")
	parser.add_argument("noise_func", type=str, default="", choices=["", \
						"normal","gaussian_mixture"], help="Noise model to "+\
						"perturb sample payoffs around the base game payoff. "+\
						"May only be empty if first argument is also empty.")
	parser.add_argument("-game_args", type=str, nargs="*", default=[], help=\
						"Arguments to pass to game_func. Usually players "+\
						"and strategies.")
	parser.add_argument("-noise_args", type=str, nargs="*", default=[], help=\
						"Arguments to pass to noise_func. Should not include"+\
						"noise magnitude or number of samples.")
	parser.add_argument("-num_games", type=int, default=1000, help="Number "+\
						"of games to generate per stdev/subsample "+\
						"combination. Default: 1000")
	parser.add_argument("-points", type=int, default=1000, help="Number of "+\
						"bootstrap resamples per equilibrium. Default: 1000")
	parser.add_argument("-pair", type=str, default="game", choices=["game", \
						"profile", "payoff"], help="Pairing level to use for "+\
						"bootstrap resampling. Default: game")
	parser.add_argument("-agg", type=int, default=0, help="Number of samples "+\
						"to pre-aggregate. Default: 0")
	parser.add_argument("-stdevs", type=float, nargs="*", default=\
						[.2,1.,5.,25.], help="Noise magnitude parameters "+\
						"passed to the noise model. Default: .2 1. 5. 25.")
	parser.add_argument("-sample_sizes", type=int, nargs="*", default=\
						[5,10,20,100,200,500], help="Numbers of samples "+\
						"per profile at which to test the bootstrap. "+\
						"Default: 5 10 20 100 200 500")
	parser.add_argument("--single", action="store_true", help="Set to use "+\
						"the single_sample_regret function.")
	parser.add_argument("--rd", action="store_true", help="Set to compute "+\
						"bootstrap distributions of equilibrium movement by "+\
						"replicator dynamics.")
	parser.add_argument("--pure", action="store_true", help="Find an compute "+\
						"bootstrap regret distributions for pure-strategy "+\
						"(rather than role-symmetric mixed-strategy) Nash "+\
						"equilibria.")
	args = parser.parse_args()

	if args.game_func == "none":
		game = read(stdin)
		args.game_func = lambda: game
		args.noise_func = lambda g,s,c: game.subsample(c)
	else:
		assert args.noise_func != "none", "Must specify a noise model."
		game_args = []
		for a in args.game_args:
			try:
				game_args.append(int(a))
			except ValueError:
				game_args.append(float(a))
		noise_args = []
		for a in args.noise_args:
			try:
				noise_args.append(int(a))
			except ValueError:
				noise_args.append(float(a))
		args.game_func = partial(getattr(RG, args.game_func), *game_args)
		noise_func = getattr(RG, args.noise_func + "_noise")
		args.noise_func = lambda g,s,c: noise_func(g,s,c, *noise_args)

	if args.agg > 0:
		args.noise_func = lambda g,s,c: pre_aggregate(args.noise_func(g,s,c), \
											args.agg)
	assert not (args.rd and args.pure), "Must use mixed_nash for rd bootstrap"
	args.bootstrap_args = ["resample" if not args.single else "singleSample", \
						[args.pair] if not args.single else [], args.points]
	return args


def main():
	args = parse_args()
	results = bootstrap_experiment(args.game_func, args.noise_func, \
					replicator_dynamics if args.rd else regret, \
					args.num_games, args.stdevs, args.sample_sizes, \
					pure_nash if args.pure else mixed_nash, args.bootstrap_args)
	print to_JSON_str(results, indent=None)


if __name__ == "__main__":
	main()

