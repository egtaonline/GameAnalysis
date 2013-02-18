#! /usr/bin/env python2.7

from numpy.linalg import norm

from BasicFunctions import call
from GameIO import to_NFG_asym, read_NE
from RoleSymmetricGame import tiny, is_constant_sum
from Regret import regret

def pure_nash(game, epsilon=0):
	"""
	Finds all pure-strategy epsilon-Nash equilibria.
	"""
	return filter(lambda profile: regret(game, profile) <= epsilon, game)


def min_regret_profile(game):
	"""
	Finds the profile with the confirmed lowest regret.
	"""
	return min(game.knownProfiles(), key=lambda p: regret(game, p))


def mixed_nash(game, regret_thresh=1e-4, dist_thresh=1e-2, random_restarts=0, \
		at_least_one=False, *RD_args, **RD_kwds):
	"""
	Runs replicator dynamics from multiple starting mixtures.
	"""
	equilibria = []
	all_eq = []
	for m in game.biasedMixtures() + [game.uniformMixture()] + \
			[game.randomMixture() for __ in range(random_restarts)]:
		eq = replicator_dynamics(game, m, *RD_args, **RD_kwds)
		distances = map(lambda e: norm(e-eq,2), equilibria)
		if regret(game, eq) <= regret_thresh and all([d >= dist_thresh \
				for d in distances]):
			equilibria.append(eq)
		all_eq.append(eq)
	if len(equilibria) == 0 and at_least_one:
		return [min(all_eq, key=lambda e: regret(game, e))]
	return equilibria


def replicator_dynamics(game, mix, iters=10000, converge_thresh=1e-8, \
		verbose=False):
	"""
	Replicator dynamics.
	"""
	for i in range(iters):
		old_mix = mix
		mix = (game.expectedValues(mix) - game.minPayoffs + tiny) * mix
		mix = mix / mix.sum(1).reshape(mix.shape[0],1)
		if norm(mix - old_mix) <= converge_thresh:
			break
	if verbose:
		print i+1, "iterations ; mix =", mix, "; regret =", regret(game, mix)
	mix[mix < 0] = 0 #occasionally one of the probabilities is barely negative
	return mix


def gambit_lp(game):
	assert is_constant_sum(game), "game must be constant sum for LP"
	eq = read_NE(call("gambit-lp -q", to_NFG_asym(game)))
	return game.toProfile(eq.reshape(len(game.roles), game.maxStrategies))


from GameIO import io_parser, to_JSON_str

def parse_args():
	parser = io_parser()
	parser.add_argument("-r", metavar="REGRET", type=float, default=1e-3, \
			help="Max allowed regret for approximate Nash equilibria. " + \
			"default=1e-3")
	parser.add_argument("-d", metavar="DISTANCE", type=float, default=1e-3, \
			help="L2-distance threshold to consider equilibria distinct. " + \
			"default=1e-3")
	parser.add_argument("-c", metavar="CONVERGENCE", type=float, default=1e-8, \
			help="Replicator dynamics convergence thrshold. default=1e-8")
	parser.add_argument("-i", metavar="ITERATIONS", type=int, default=10000, \
			help="Max replicator dynamics iterations. default=1e4")
	parser.add_argument("-s", metavar="SUPPORT", type=float, default=1e-3, \
			help="Min probability for a strategy to be considered in " + \
			"support. default=1e-3")
	parser.add_argument("-type", choices=["mixed", "pure", "mrp"], default= \
			"mixed", help="Type of approximate equilibrium to compute: " + \
			"role-symmetric mixed-strategy Nash, pure-strategy Nash, or " + \
			"min-regret profile. default=mixed")
	parser.add_argument("-p", metavar="POINTS", type=int, default=0, \
			help="Number of random points from which to initialize " + \
			"replicator dynamics in addition to the default set of uniform " +\
			"and heavily-biased mixtures.")
	parser.add_argument("--one", action="store_true", help="Always report " +\
			"at least one equilibrium per game.")
	args = parser.parse_args()
	games = args.input
	if not isinstance(games, list):
		games = [games]
	return games, args


def main():
	games, args = parse_args()
	if args.type == "pure":
		equilibria = [pure_nash(g, args.r) for g in games]
		if args.one:
			for i in range(len(games)):
				if len(equilibria[i]) == 0:
					equilibria[i] = min_regret_profile(games[i])
	elif args.type == "mixed":
		equilibria = [[g.toProfile(eq, args.s) for eq in mixed_nash(g, \
				args.r, args.d, args.p, args.one, iters=args.i, \
				converge_thresh=args.c)] for g in games]
	elif args.type == "mrp":
		equilibria = map(min_regret_profile, games)
	if len(equilibria) > 1:
		print to_JSON_str(equilibria)
	else:
		print to_JSON_str(equilibria[0])


if __name__ == "__main__":
	main()


