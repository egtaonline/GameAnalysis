#! /usr/bin/env python2.7

import GameIO as IO

from BasicFunctions import leading_zeros
from RoleSymmetricGame import Game, SampleGame, PayoffData, Profile

from functools import partial
from itertools import combinations
from bisect import bisect
from numpy.random import uniform as U, normal, multivariate_normal, beta
from random import choice
from numpy import array, arange, zeros, fill_diagonal, cumsum
from sys import argv


def __make_asymmetric_game(player_count, strategy_count):
	roles = map(str, range(player_count))
	players = {r:1 for r in roles}
	strategies = {r:map(str, range(strategy_count)) for r in roles}
	return Game(roles, players, strategies)

def independent(N, S, dstr=partial(U,-1,1)):
	"""
	All payoff values drawn independently according to specified distribution.

	N: number of players
	S: number of strategies
	dstr: distribution from which payoff values are independently drawn
	"""
	g = __make_asymmetric_game(N, S)
	for prof in g.allProfiles():
		g.addProfile({r:[PayoffData(prof[r].keys()[0], 1, dstr())] \
				for r in prof})
	return g


def covariant(N, S, mean_func=lambda:0, var=1, covar_func=partial(U,0,1)):
	"""
	Payoff values for each profile drawn according to multivariate normal.

	The multivariate normal for each profile has a constant mean-vector with 
	value drawn from mean_func, constant variance=var, and equal covariance 
	between all pairs of players, drawn from covar_func.

	N: number of players
	S: number of strategies
	mean_func: distribution from which mean payoff for each profile is drawn
	var: diagonal entries of covariance matrix
	covar_func: distribution from which the value of the off-diagonal 
				covariance matrix entries for each profile is drawn

	Both mean_func and covar_func should be numpy-style random number generators
	that can return an array.
	"""
	g = __make_asymmetric_game(N, S)
	mean = zeros(S)
	covar = zeros([S,S])
	for prof in g.allProfiles():
		mean.fill(mean_func())
		payoffs = multivariate_normal(mean, covar)
		covar.fill(covar_func())
		fill_diagonal(covar, var)
		g.addProfile({r:[PayoffData(prof[r].keys()[0], 1, payoffs[i])] \
				for i,r in enumerate(g.roles)})
	return g
	

def uniform_zero_sum(S, min_val=-1, max_val=1):
	"""
	2-player zero-sum game; player 1 payoffs drawn from a uniform distribution.

	S: number of strategies
	min_val: minimum for the uniform distribution
	max_val: maximum for the uniform distribution
	"""
	roles = ["row", "column"]
	players = {r:1 for r in roles}
	strategies = {"row":["r" + leading_zeros(i,S) for i in range(S)], \
			"column":["c" + leading_zeros(i,S) for i in range(S)]}
	g = Game(roles, players, strategies)
	for prof in g.allProfiles():
		row_strat = prof["row"].keys()[0]
		row_val = U(min_val, max_val)
		col_strat = prof["column"].keys()[0]
		p = {"row":[PayoffData(row_strat, 1, row_val)], \
				"column":[PayoffData(col_strat, 1, -row_val)]}
		g.addProfile(p)
	return g


def uniform_symmetric(N, S, min_val=-1, max_val=-1):
	"""
	Symmetric game with each payoff value drawn from a uniform distribution.

	N: number of players
	S: number of strategies
	min_val: minimum for the uniform distribution
	max_val: maximum for the uniform distribution
	"""
	roles = ["All"]
	players = {"All":N}
	strategies = {"All":["s" + leading_zeros(i,S) for i in range(S)]}
	g = Game(roles, players, strategies)
	for prof in g.allProfiles():
		payoffs = []
		for strat, count in prof["All"].items():
			payoffs.append(PayoffData(strat, count, U(min_val, max_val)))
		g.addProfile({"All":payoffs})
	return g


def congestion(N, facilities, required):
	"""
	Generates random congestion games with N players and nCr(f,r) strategies.

	Congestion games are symmetric, so all players belong to role All. Each 
	strategy is a subset of size #required among the size #facilities set of 
	available facilities. Payoffs for each strategy are summed over facilities.
	Each facility's payoff consists of three components:

	-constant ~ U[0,#facilities]
	-linear congestion cost ~ U[-#required,0]
	-quadratic congestion cost ~ U[-1,0]
	"""
	roles = ["All"]
	players = {"All":N}
	strategies = {'+'.join(["f"+str(f) for f in strat]):strat for strat in \
			combinations(range(facilities), required)}
	facility_values = [array([U(facilities), U(-required), U(-1)]) for __ in \
			range(facilities)]
	g = Game(roles, players, {"All":strategies.keys()})
	for prof in g.allProfiles():
		payoffs = []
		useage = [0]*facilities
		for strat, count in prof["All"].items():
			for facility in strategies[strat]:
				useage[facility] += count
		for strat, count in prof["All"].items():
			payoffs.append(PayoffData(strat, count, [sum(useage[f]**arange(3) \
					* facility_values[f]) for f in strategies[strat]]))
		g.addProfile({"All":payoffs})
	return g


def local_effect(N, S):
	"""
	Generates random congestion games with N players and S strategies.

	Local effect games are symmetric, so all players belong to role All. Each
	strategy corresponds to a node in the G(N,2/S) local effect graph. Payoffs
	for each strategy consist of constant terms for each strategy, and
	interaction terms for the number of players choosing that strategy and each
	neighboring strategy.

	The one-strategy terms are drawn as follows:
	-constant ~ U[-N-S,N+S]
	-linear ~ U[-N,N]

	The neighbor strategy terms are drawn as follows:
	-linear ~ U[-S,S]
	-quadratic ~ U[-1,1]
	"""
	roles = ["All"]
	players = {"All":N}
	strategies = ["s"+str(i) for i in range(S)]
	local_effects = {s:{} for s in strategies}
	for s in strategies:
		for d in strategies:
			if s == d:
				local_effects[s][d] = [U(-N-S,N+S),U(-N,N)]
			elif U(0,S) > 2:
				local_effects[s][d] = [U(-S,S),U(-1,1)]
	g = Game(roles, players, {"All":strategies})
	for prof in g.allProfiles():
		payoffs = []
		for strat, count in prof["All"].items():
			value = local_effects[strat][strat][0] + \
					local_effects[strat][strat][1] * count
			for neighbor in local_effects[strat]:
				if neighbor not in prof["All"]:
					continue
				nc = prof["All"][neighbor]
				value += local_effects[strat][neighbor][0] * count
				value += local_effects[strat][neighbor][1] * count**2
			payoffs.append(PayoffData(strat, count, value))
		g.addProfile({"All":payoffs})
	return g


def polymatrix(N, S, matrix_game=partial(independent,2)):
	"""
	Creates a polymatrix game using the specified 2-player matrix game function.

	Each player's payoff in each profile is a sum over independent games played
	against each opponent. Each pair of players plays an instance of the
	specified random 2-player matrix game.

	N: number of players
	S: number of strategies
	matrix_game: a function of one argument (S) that returns 2-player, 
					S-strategy games.
	"""
	roles = map(str, range(N))
	players = {r:1 for r in roles}
	strategies = {r : map(str, range(S)) for r in roles}
	matrices = {pair : matrix_game(2, S) for pair in combinations(roles, 2)}
	g = Game(roles, players, strategies)
	
	for prof in g.allProfiles():
		payoffs = {r:0 for r in roles}
		for role in roles:
			role_strat = prof[role].keys()[0]
			for other in roles:
				if role < other:
					m = matrices[(role, other)]
					p0 = sorted(m.players.keys())[0]
					p1 = sorted(m.players.keys())[1]
				elif role > other:
					m = matrices[(other, role)]
					p0 = sorted(m.players.keys())[1]
					p1 = sorted(m.players.keys())[0]
				else:
					continue
				other_strat = prof[other].keys()[0]
				s0 = m.strategies[p0][strategies[role].index(role_strat)]
				s1 = m.strategies[p1][strategies[other].index(other_strat)]
				m_prof = Profile({p0:{s0:1},p1:{s1:1}})
				payoffs[role] += m.getPayoff(m_prof, p0, s0)
		g.addProfile({r:[PayoffData(prof[r].keys()[0], 1, payoffs[r])] \
						for r in roles})
	return g


def add_noise(game, model, spread, samples):
	"""
	Generate sample game with random noise added to each payoff.
	
	game: a RSG.Game or RSG.SampleGame
	model: a 2-parameter function that generates mean-zero noise
	spread, samples: the parameters passed to the noise function
	"""
	sg = SampleGame(game.roles, game.players, game.strategies)
	for prof in game.knownProfiles():
		sg.addProfile({r:[PayoffData(s, prof[r][s], game.getPayoff(prof,r,s) + \
				model(spread, samples)) for s in prof[r]] for r in game.roles})
	return sg


def gaussian_mixture_noise(max_stdev, samples, modes=2):
	"""
	Generate Gaussian mixture noise to add to one payoff in a game.

	max_stdev: maximum standard deviation for the mixed distributions (also 
				affects how widely the mixed distributions are spaced)
	samples: numer of samples to take of every profile
	modes: number of Gaussians to mix
	"""
	multipliers = range((-modes+1)/2,0) + [0]*(modes%2) + range(1,modes/2+1)
	offset = normal(0, max_stdev)
	stdev = beta(2,1) * max_stdev
	return normal(choice(multipliers)*offset, stdev, samples)


def uniform_noise(max_half_width, samples):
	"""
	Generate uniform random noise to add to one payoff in a game.

	max_range: maximum half-width of the uniform distribution
	samples: numer of samples to take of every profile
	"""
	hw = beta(2,1) * max_half_width
	return U(-hw, hw, samples)


normal_noise = partial(normal, 0)
unimodal_noise = partial(gaussian_mixture_noise, modes=1)
bimodal_noise = partial(gaussian_mixture_noise, modes=2)


def mix_models(models, rates, spread, samples):
	"""
	Generate SampleGame with noise drawn from several models.

	models: a list of 2-parameter noise functions to draw from
	rates: the probabilites with which a payoff will be drawn from each model
	spread, samples: the parameters passed to the noise functions
	"""
	cum_rates = cumsum(rates)
	m = models[bisect(cum_rates, U(0,1))]
	return m(spread, samples)


u80b20_noise = partial(mix_models, [unimodal_noise, bimodal_noise], [.8,.2])


def parse_args():
	parser = IO.io_parser(description="Generate random games.")
	parser.add_argument("type", choices=["uZS", "uSym", "CG", "LEG"], help= \
			"Type of random game to generate. uZS = uniform zero sum. " +\
			"uSym = uniform symmetric. CG = congestion game.")
	parser.add_argument("count", type=int, help="Number of random games " +\
			"to create.")
	parser.add_argument("-noise", choices=["none", "normal", \
			"gauss_mix"], default="None", help="Noise function.")
	parser.add_argument("-noise_args", nargs="*", default=[], \
			help="Arguments to be passed to the noise function.")
	parser.add_argument("-game_args", nargs="*", default=[], \
			help="Additional arguments for game generator function.")
	assert "-input" not in argv, "no input JSON required"
	argv = argv[:3] + ["-input", None] + argv[3:]
	return parser.parse_args()


def main():
	args = parse_args()

	if args.type == "uZS":
		game_func = uniform_zero_sum
		assert len(args.game_args) == 1, "game_args must specify strategy count"
		game_args = map(int, args.game_args)
	elif args.type == "uSym":
		game_func = uniform_symmetric
		assert len(args.game_args) == 2, "game_args must specify player and "+\
									"strategy counts"
	elif args.type == "CG":
		game_func = congestion
		assert len(args.game_args) == 3, "game_args must specify player, "+\
									"facility, and required facility counts"
		game_args = map(int, args.game_args)
	elif args.type == "LEG":
		game_func = local_effect
		assert len(args.game_args) == 2, "game_args must specify player and "+\
									"strategy counts"
		game_args = map(int, args.game_args)
	
	games = [game_func(*game_args) for __ in range(args.count)]

	if args.noise == "normal":
		assert len(args.noise_args) == 2, "noise_args must specify stdev "+\
											"and sample count"
		noise_args = [float(args.noise_args[0]), int(args.noise_args[1])]
		games = map(lambda g: normal_noise(g, *noise_args), games)	
	elif args.noise == "gauss_mix":
		assert len(args.noise_args) == 3, "noise_args must specify max "+\
									"stdev, sample count, and number of modes"
		noise_args = [float(args.noise_args[0]), int(args.noise_args[1]), \
						int(args.noise_args[2])]
		games = map(lambda g: gaussian_mixture_noise(g, *noise_args), games)	

	if len(games) == 1:
		print IO.to_JSON_str(games[0])
	else:
		print IO.to_JSON_str(games)


if __name__ == "__main__":
	main()
