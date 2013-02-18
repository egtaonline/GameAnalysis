#! /usr/bin/env python2.7

from HashableClasses import h_dict
from RoleSymmetricGame import SampleGame, PayoffData

def translate(arr, source_game, target_game):
	"""
	Translates a mixture, profile, count, or payoff array between related
	games based on role/strategy indices.

	Useful for testing full-game regret of subgame equilibria.
	"""
	a = target_game.zeros()
	for role in target_game.roles:
		for strategy in source_game.strategies[role]:
			a[target_game.index(role), target_game.index(role, strategy)] = \
					arr[source_game.index(role), source_game.index(role, \
					strategy)]
	return a


def subgame(game, strategies={}, require_all=False):
	"""
	Creates a game with a subset each role's strategies.

	default settings result in a subgame with no strategies
	"""
	if not strategies:
		strategies = {r:[] for r in game.roles}
	sg = type(game)(game.roles, game.players, strategies)
	if sg.size <= len(game):
		for p in sg.allProfiles():
			if p in game:
				add_subgame_profile(game, sg, p)
			elif require_all:
				raise KeyError("Profile missing")
	elif require_all:
		raise KeyError("Profile missing")
	else:
		for p in game:
			if is_valid_profile(sg, p):
				add_subgame_profile(game, sg, p)
	return sg


def add_subgame_profile(game, subgame, prof):
	if isinstance(game, SampleGame):
		subgame.addProfile({role:[PayoffData(strat, prof[role][strat], \
				game.sample_values[game[prof]][game.index(role), game.index( \
				role, strat)]) for strat in prof[role]] for role in prof})
	else:
		subgame.addProfile({role:[PayoffData(strat, prof[role][strat], \
				game.getPayoff(prof, role, strat)) for strat in prof[role]] \
				for role in prof})


def is_valid_profile(game, prof):
	if set(prof.keys()) != set(game.roles):
		return False
	for role in prof:
		for strat in prof[role]:
			if strat not in game.strategies[role]:
				return False
	return True


def is_subgame(small_game, big_game):
	if any((r not in big_game.roles for r in small_game.roles)):
		return False
	if any((small_game.players[r] != big_game.players[r] for r \
			in small_game.roles)):
		return False
	for r in small_game.roles:
		if any((s not in big_game.strategies[r] for s in \
				small_game.strategies[r])):
			return False
	return True


def cliques(full_game, known_subgames=[]):
	"""
	Finds maximal subgames for which all profiles are known.

	input:
	subgames = known complete subgames to be expanded (any payoff data in
	the known subgames is ignored, so for faster loading, give only the
	header information).
	"""
	new_profiles = set(full_game.knownProfiles()) - set().union(*( \
			subgame(full_game, s).allProfiles() for s in known_subgames))
	new_strategies = {r:set() for r in full_game.roles}
	for prof in new_profiles:
		for role in full_game.roles:
			new_strategies[role] |= set(prof[role].keys())
	subgames = {subgame(full_game).strategies}
	for sg in known_subgames:
		try:
			sg = subgame(full_game, sg.strategies)
		except AttributeError:
			sg =  subgame(full_game, sg)
		if sg.isComplete():
			subgames.add(sg.strategies)
	maximal_subgames = set()
	explored_subgames = set()
	while(subgames):
		sg_strat = subgames.pop()
		empty_role = not all(map(len, sg_strat.values()))
		explored_subgames.add(sg_strat)
		maximal = True
		for role in full_game.roles:
			if empty_role and len(sg_strat[role]) > 0:
				continue
			for s in new_strategies[role] - set(sg_strat[role]):
				strategies = h_dict({r : tuple(sorted(list(sg_strat[r]) + \
						([s] if r == role else []))) for r in full_game.roles})
				if strategies in explored_subgames:
					maximal=False
					continue
				try:
					new_sg = subgame(full_game, strategies, True)
				except KeyError:
					continue
				maximal=False
				subgames.add(new_sg.strategies)
		if maximal:
			sg = subgame(full_game, sg_strat)
			if len(sg) > 0:
				maximal_subgames.add(sg.strategies)
	return sorted(maximal_subgames, key=len)


from GameIO import to_JSON_str, io_parser


def main():
	parser = io_parser()
	args = parser.parse_args()
	print to_JSON_str(map(lambda s: subgame(args.input, s), \
					cliques(args.input)))


if __name__ == "__main__":
	main()
