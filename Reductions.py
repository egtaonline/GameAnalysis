#! /usr/bin/env python2.7

from RoleSymmetricGame import Game, Profile, PayoffData, is_symmetric

def hierarchical_reduction(game, players={}):
	if not players:
		players = {r : game.players[r] / 2 for r in game.roles}
	HR_game = type(game)(game.roles, players, game.strategies)
	for reduced_profile in HR_game.allProfiles():
		try:
			full_profile = Profile({r:full_game_profile(reduced_profile[r], \
					game.players[r]) for r in game.roles})
			HR_game.addProfile({r:[PayoffData(s, reduced_profile[r][s], \
					game.getPayoffData(full_profile, r, s)) for s in \
					full_profile[r]] for r in full_profile})
		except KeyError:
			continue
	return HR_game


def full_game_profile(HR_profile, N):
	"""
	Returns the symmetric full game profile corresponding to the given
	symmetric reduced game profile.
	"""
	n = sum(HR_profile.values())
	full_profile = {s : c * N / n  for s,c in HR_profile.items()}
	while sum(full_profile.values()) < N:
		full_profile[max([(float(N) / n * HR_profile[s] \
				- full_profile[s], s) for s in full_profile])[1]] += 1
	return full_profile


def deviation_preserving_reduction(game, players={}):
	if not players:
		return twins_reduction(game)

	#it's often faster to go through all of the full-game profiles
	DPR_game = type(game)(game.roles, players, game.strategies)
	if len(game) <= DPR_game.size and is_symmetric(game):
		reduced_profile_data = {}
		divisible = True
		for full_prof in game.knownProfiles():
			try:
				rp, role, strat = dpr_profile(full_prof, players, game.players)
			except NotImplementedError:
				divisible = False
				break
			if rp not in reduced_profile_data:
				reduced_profile_data[rp] = {r:[] for r in game.roles}
			count = rp[role][strat]
			value = game.getPayoffData(full_prof, role, strat)
			reduced_profile_data[rp][role].append(PayoffData(strat,count,value))
		if divisible:
			for prof_data in reduced_profile_data.values():
				valid_profile = True
				for r,l in prof_data.items():
					if sum([d.count for d in l]) != players[r]:
						valid_profile = False
						break
				if valid_profile:
					DPR_game.addProfile(prof_data)
			return DPR_game
	
	#it's conceptually simpler to go through all of the reduced-game profiles
	DPR_game = type(game)(game.roles, players, game.strategies)
	for reduced_profile in DPR_game.allProfiles():
		try:
			role_payoffs = {}
			for role in game.roles:
				role_payoffs[role] = []
				for s in reduced_profile[role]:
					full_profile = {}
					for r in game.roles:
						if r == role:
							opp_prof = reduced_profile.asDict()[r]
							opp_prof[s] -= 1
							full_profile[r] = full_game_profile(opp_prof, \
									game.players[r] - 1)
							full_profile[r][s] += 1
						else:
							full_profile[r] = full_game_profile( \
									reduced_profile[r], game.players[r])
					role_payoffs[r].append(PayoffData(s,reduced_profile[r][s],\
							game.getPayoffData(Profile(full_profile), r, s)))
			DPR_game.addProfile(role_payoffs)
		except KeyError:
			continue
	return DPR_game


def dpr_profile(full_profile, reduced_players, full_players=None):
	"""
	Gives the DPR profile, role, and strategy that f_p's payoffs determine

	Providing full_players is optional, but saves some work.
	"""
	role = None
	strat = None
	if full_players == None:
		full_players = {r:sum(v.values()) for r,v in full_profile.items()}
	reduction_factors = {}
	for r in full_profile:
		if full_players[r] % reduced_players[r] == 0:
			reduction_factors[r] = full_players[r] / reduced_players[r]
		else:
			role = r
			reduction_factors[r] = (full_players[r]-1) / (reduced_players[r]-1)
	for s,c in full_profile[role].items():
		if c % reduction_factors[r] != 0:
			if strat != None:
				raise NotImplementedError("This function doesn't handle "+\
										"non-divisible player counts yet.")
			strat = s
	reduced_profile = {r:{} for r in reduced_players}
	for r in full_profile:
		for s,c in full_profile[role].items():
			if r == role and s == strat:
				reduced_profile[r][s] = (c-1) / reduction_factors[r] + 1
			else:
				reduced_profile[r][s] = c / reduction_factors[r]
	return Profile(reduced_profile), role, strat


def twins_reduction(game):
	return deviation_preserving_reduction(game, {r:2 for r in game.roles})


def DPR_profiles(game, players={}):
	"""Returns the profiles from game that contribute to the DPR game."""
	if not players:
		players = {r:2 for r in game.roles}
	DPR_game = Game(game.roles, players, game.strategies)
	profiles = []
	for reduced_profile in DPR_game.allProfiles():
		for role in game.roles:
			for s in reduced_profile[role]:
				full_profile = {}
				for r in game.roles:
					if r == role:
						opp_prof = reduced_profile.asDict()[r]
						opp_prof[s] -= 1
						full_profile[r] = full_game_profile(opp_prof, \
								game.players[r] - 1)
						full_profile[r][s] += 1
					else:
						full_profile[r] = full_game_profile(reduced_profile[\
								r], game.players[r])
				profiles.append(Profile(full_profile))
	return profiles


from GameIO import to_JSON_str, io_parser

def parse_args():
	parser = io_parser()
	parser.add_argument("type", choices=["DPR", "HR", "TR"], help="Type " + \
			"of reduction to perform.")
	parser.add_argument("players", type=int, default=[], nargs="*", help= \
			"Number of players in each reduced-game role.")
	return parser.parse_args()


def main():
	args = parse_args()
	game = args.input
	players = dict(zip(game.roles, args.players))
	if args.type == "DPR":
		print to_JSON_str(deviation_preserving_reduction(game, players))
	elif args.type == "HR":
		print to_JSON_str(hierarchical_reduction(game, players))
	elif args.type == "TR":
		print to_JSON_str(twins_reduction(game))


if __name__ == "__main__":
	main()
