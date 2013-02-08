import numpy as np

from itertools import product, chain, combinations_with_replacement as CwR
from collections import namedtuple
from math import isinf
from string import join
from random import choice

from HashableClasses import *
from BasicFunctions import *

PayoffData = namedtuple("payoff", "strategy count value")

tiny = float(np.finfo(np.float64).tiny)

class Profile(h_dict):
	def __init__(self, role_payoffs):
		try:
			d = {}
			for role, payoffs in role_payoffs.items():
				d[role] = h_dict({p.strategy:p.count for p in payoffs})
			h_dict.__init__(self, d)
		except AttributeError:
			h_dict.__init__(self, {r:h_dict(p) for r,p in role_payoffs.items()})

	def remove(self, role, strategy):
		p = self.asDict()
		p[role][strategy] -= 1
		if p[role][strategy] == 0:
			del p[role][strategy]
		return Profile(p)

	def add(self, role, strategy):
		p = self.asDict()
		p[role][strategy] = p[role].get(strategy, 0) + 1
		return Profile(p)

	def deviate(self, role, strategy, deviation):
		p = self.asDict()
		p[role][strategy] -= 1
		if p[role][strategy] == 0:
			del p[role][strategy]
		p[role][deviation] = p[role].get(deviation, 0) + 1
		return Profile(p)

	def asDict(self):
		return {r:{s:self[r][s] for s in self[r]} for r in self}

	def __repr__(self):
		return join([role +": "+ join([str(count) +" "+ strategy for strategy, \
			count in self[role].items()], ", ") for role in self], "; ")

	def toJSON(self):
		return {"type":"GA_Profile", "data":self}


class Game(dict):
	def __init__(self, roles=[], players={}, strategies={}, payoff_data=[]):
		"""
		Role-symmetric game representation.

		__init__ parameters:
		roles: collection of role-name strings
		players: mapping from roles to number of players per role
		strategies: mapping from roles to per-role strategy sets
		payoff_data: collection of data objects mapping roles to collections
				of (strategy, count, value) tuples
		"""
		self.roles = sorted(set(map(str, roles)))
		self.players = h_dict({r : int(players[r]) for r in self.roles})
		self.strategies = h_dict({r : tuple(sorted(set(map(str, \
				strategies[r])))) for r in self.roles})

		self.numStrategies = [len(self.strategies[r]) for r in self.roles]
		self.maxStrategies = max(self.numStrategies)
		self.minPayoffs = self.zeros(dtype=float, masked=False)
		self.minPayoffs.fill(float('inf'))

		self.mask = np.array([[False]*s + [True]*(self.maxStrategies - s) for \
				s in self.numStrategies])
		self.size = prod([game_size(self.players[r], self.numStrategies[ \
				i]) for i,r in enumerate(self.roles)])
		self.role_index = {r:i for i,r in enumerate(self.roles)}
		self.strategy_index = {r : {s:i for i,s in enumerate( \
				self.strategies[r]) } for r in self.roles}
		self.values = []
		self.counts = []
		self.dev_reps = []
		for profile_data_set in payoff_data:
			self.addProfile(profile_data_set)

	def addProfile(self, role_payoffs):
		prof = Profile(role_payoffs)
		if prof in self:
			raise IOError("duplicate profile: " + str(prof))
		self.makeLists()
		self.addProfileArrays(role_payoffs)
		self.addDevReps(role_payoffs)
		self[prof] = len(self.values) - 1

	def addProfileArrays(self, role_payoffs):
		counts = self.zeros(dtype=int)
		values = self.zeros(dtype=float)
		for r, role in enumerate(self.roles):
			for strategy, count, value in role_payoffs[role]:
				s = self.index(role, strategy)
				min_value = np.min(value)
				if min_value < self.minPayoffs[r][0]:
					self.minPayoffs[r] = min_value
				values[r,s] = np.average(value)
				counts[r,s] = count
		self.values.append(values)
		self.counts.append(counts)

	def addDevReps(self, role_payoffs):
		devs = self.zeros(dtype=int)
		counts = self.counts[-1]
		for i, r in enumerate(self.roles):
			for j, s in enumerate(self.strategies[r]):
				if counts[i,j] > 0:
					opp_prof = counts - self.array_index(r,s)
					try:
						devs[i,j] = profile_repetitions(opp_prof)
					except OverflowError:
						devs = np.array(devs, dtype=object)
						devs[i,j] = profile_repetitions(opp_prof)
				else:
					devs[i,j] = 0
		self.dev_reps.append(devs)

	def makeLists(self):
		if isinstance(self.values, np.ndarray):
			self.values = list(self.values)
			self.counts = list(self.counts)
			self.dev_reps = list(self.dev_reps)

	def makeArrays(self):
		if isinstance(self.values, list):
			self.values = np.array(self.values)
			self.counts = np.array(self.counts)
			self.dev_reps = np.array(self.dev_reps)

	def __hash__(self):
		return hash((self.players, self.strategies))

	def index(self, role, strategy=None):
		"""
		index(r) returns the role-index of r
		index(r,s) returns the strategy index of s (for role r)
		"""
		if strategy != None:
			return self.strategy_index[role][strategy]
		return self.role_index[role]

	def zeros(self, dtype=float, masked=False):
		z = np.zeros([len(self.roles), self.maxStrategies], dtype=dtype)
		return np.ma.array(z, mask=self.mask) if masked else z

	def array_index(self, role, strategy=None, dtype=bool):
		"""
		array_index(r,s) returns a boolean ndarray version of index(r,s)
		"""
		a = self.zeros(dtype=dtype)
		if strategy == None:
			a[self.index(role)] += 1
		else:
			a[self.index(role), self.index(role, strategy)] += 1
		return a

	def getPayoff(self, profile, role, strategy):
		v = self.values[self[profile]]
		return v[self.index(role), self.index(role,strategy)]

	def getPayoffData(self, profile, role, strategy):
		return self.getPayoff(profile, role, strategy)

	def getExpectedPayoff(self, mix, role=None):
		if role == None:
			return (mix * self.expectedValues(mix)).sum(1)
		return (mix * self.expectedValues(mix)).sum(1)[self.index(role)]

	def getSocialWelfare(self, profile):
		if is_pure_profile(profile):
			return self.values[self[profile]].sum()
		if is_mixture_array(profile):
			players = np.array([self.players[r] for r in self.roles])
			return (self.getExpectedPayoff(profile) * players).sum()
		if is_profile_array(profile):
			return self.getSocialWelfare(self.toProfile(profile))
		if is_mixed_profile(profile):
			return self.getSocialWelfare(self.toArray(profile))

	def expectedValues(self, mix):
		"""
		Computes the expected value of each pure strategy played against
		all opponents playing mix.

		The result is normalized by the sum of all profile weights to cope
		with missing profiles.
		"""
		self.makeArrays()
		try:
			weights = ((mix+tiny)**self.counts).prod(1).prod(1).reshape( \
					self.values.shape[0], 1, 1) * self.dev_reps / (mix+tiny)
		except ValueError: #this happens if there's only one strategy
			weights = ((mix+tiny)**self.counts).prod(1).reshape( \
					self.values.shape[0], 1) * self.dev_reps / (mix+tiny)
		return (self.values * weights).sum(0) / (weights.sum(0) + tiny)

	def allProfiles(self):
		return [Profile({r:{s:p[self.index(r)].count(s) for s in set(p[ \
				self.index(r)])} for r in self.roles}) for p in product(*[ \
				CwR(self.strategies[r], self.players[r]) for r in self.roles])]

	def knownProfiles(self):
		return self.keys()

	def isComplete(self):
		return len(self) == self.size

	def uniformMixture(self):
		return np.array(1-self.mask, dtype=float) / \
				(1-self.mask).sum(1).reshape(len(self.roles),1)

	def randomMixture(self):
		m = np.random.uniform(0, 1, size=self.mask.shape)
		return m / m.sum(1).reshape(m.shape[0], 1)

	def biasedMixtures(self, role=None, strategy=None, bias=.9):
		"""
		Gives mixtures where the input strategy has %bias weight for its role.

		Probability for that role's remaining strategies is distributed
		uniformly, as is probability for all strategies of other roles.

		Returns a list even when a single role & strategy are specified, since
		the main use case is starting replicator dynamics from several mixtures.
		"""
		assert 0 <= bias <= 1, "probabilities must be between zero and one"
		if self.maxStrategies == 1:
			return [self.uniformMixture()]
		if role == None:
			return flatten([self.biasedMixtures(r, strategy, bias) for r \
					in filter(lambda r: self.numStrategies[self.index(r)] \
					> 1, self.roles)])
		if strategy == None:
			return flatten([self.biasedMixtures(role, s, bias) for s in \
					self.strategies[role]])
		i = self.array_index(role, strategy, dtype=float)
		m = 1. - self.mask - i
		m /= m.sum(1).reshape(m.shape[0], 1)
		m[self.index(role)] *= (1. - bias)
		m += i*bias
		return [m]

	def toProfile(self, arr, supp_thresh=1e-3):
		arr = np.array(arr)
		if is_mixture_array(arr):
			arr[arr < supp_thresh] = 0
			sums = arr.sum(1).reshape(arr.shape[0], 1)
			if np.any(sums == 0):
				raise ValueError("no probability greater than threshold.")
			arr /= sums
		p = {}
		for r in self.roles:
			i = self.index(r)
			p[r] = {}
			for s in self.strategies[r]:
				j = self.index(r, s)
				if arr[i,j] > 0:
					p[r][s] = arr[i,j]
		return Profile(p)

	def toArray(self, prof):
		if is_mixed_profile(prof):
			a = self.zeros(dtype=float)
		elif is_pure_profile(prof):
			a = self.zeros(dtype=int)
		else:
			raise TypeError(one_line("unrecognized profile type: " + \
					str(prof), 71))
		for role in prof.keys():
			i = self.index(role)
			for strategy in prof[role].keys():
				j = self.index(role, strategy)
				a[i,j] = prof[role][strategy]
		return a

	def __cmp__(self, other):
		"""does not compare payoffs"""
		return cmp(self.roles, other.roles) or \
				cmp(self.players, other.players) or \
				cmp(self.strategies, other.strategies) or \
				cmp(sorted(self.keys()), sorted(other.keys()))

	def __eq__(self, other):
		return not self.__cmp__(other)

	def __ne__(self, other):
		return self.__cmp__(other)

	def __lt__(self, other):
		return min(self.__cmp__(other), 0)

	def __gt__(self, other):
		return max(self.__cmp__(other), 0)

	def __le__(self, other):
		return self == other or self < other

	def __ge__(self, other):
		return self == other or self > other

	def __repr__(self):
		return (str(self.__class__.__name__) + ":\n\troles: " + \
				join(self.roles, ",") + "\n\tplayers:\n\t\t" + \
				join(map(lambda x: str(x[1]) + "x " + str(x[0]), \
				sorted(self.players.items())), "\n\t\t") + \
				"\n\tstrategies:\n\t\t" + join(map(lambda x: x[0] + \
				":\n\t\t\t" + join(x[1], "\n\t\t\t"), \
				sorted(self.strategies.items())), "\n\t\t") + \
				"\npayoff data for " + str(len(self)) + " out of " + \
				str(self.size) + " profiles").expandtabs(4)

	def to_TB_JSON(self):
		"""
		Convert to JSON according to the EGTA-online v3 default game spec.
		"""
		game_dict = {}
		game_dict["roles"] = [{"name":role, "count":self.players[role], \
					"strategies": list(self.strategies[role])} for role \
					in self.roles]
		game_dict["profiles"] = []
		for prof in self:
			p = self[prof]
			sym_groups = []
			for r, role in enumerate(self.roles):
				for strat in prof[role]:
					s = self.index(role, strat)
					sym_groups.append({"role":role, "strategy":strat, \
							"count":self.counts[p][r,s], \
							"payoff":float(self.values[p][r,s])})
			game_dict["profiles"].append({"symmetry_groups":sym_groups})
		return game_dict

	def toJSON(self):
		"""
		Convert to JSON according to the EGTA-online v3 default game spec.
		"""
		game_dict = {}
		game_dict["players"] = self.players
		game_dict["strategies"] = self.strategies
		game_dict["profiles"] = []
		for prof in self:
			game_dict["profiles"].append({role:[(strat, prof[role][strat], \
					self.getPayoff(prof, role, strat)) for strat in \
					prof[role]] for role in prof})
		return game_dict


def is_pure_profile(prof):
	if not isinstance(prof, h_dict):
		return False
	flat = flatten([v.values() for v in prof.values()])
	return all([isinstance(count, int) and count >= 0 for count in flat])


def is_mixed_profile(prof):
	if not isinstance(prof, h_dict):
		return False
	flat = flatten([v.values() for v in prof.values()])
	return all([prob >= 0 for prob in flat]) and \
			np.allclose(sum(flat), len(prof))


def is_profile_array(arr):
	return isinstance(arr, np.ndarray) and np.all(arr >= 0) and \
			arr.dtype == int


def is_mixture_array(arr):
	return isinstance(arr, np.ndarray) and np.all(arr >= 0) and \
			np.allclose(arr.sum(1), 1)


def is_symmetric(game):
	return len(game.roles) == 1


def is_asymmetric(game):
	return all([p == 1 for p in game.players.values()])


def is_zero_sum(game):
	game.makeArrays()
	return np.allclose(game.values.sum(1).sum(1), 0)


def is_constant_sum(game):
	game.makeArrays()
	s = game.values[0].sum()
	return np.allclose(game.values.sum(1).sum(1), s)


class SampleGame(Game):
	def __init__(self, *args, **kwargs):
		self.sample_values = []
		self.min_samples = float("inf")
		self.max_samples = 0
		Game.__init__(self, *args, **kwargs)

	def addProfile(self, role_payoffs):
		Game.addProfile(self, role_payoffs)
		self.addSamples(role_payoffs)

	def addSamples(self, role_payoffs):
		samples = map(list, self.zeros())
		for r, role in enumerate(self.roles):
			played = []
			for strat, count, values in role_payoffs[role]:
				s = self.index(role, strat)
				samples[r][s] = values
				self.min_samples = min(self.min_samples, len(values))
				self.max_samples = max(self.max_samples, len(values))
				played.append(strat)
			for strat in set(self.strategies[role]) - set(played):
				s = self.index(role, strat)
				p = self.index(role, played[0])
				samples[r][s] = [0]*len(samples[r][p])
			for s in range(self.numStrategies[r], self.maxStrategies):
				p = self.index(role, played[0])
				samples[r][s] = [0]*len(samples[r][p])
		self.sample_values.append(np.array(samples))
	
	def getPayoffData(self, profile, role, strategy):
		v = self.sample_values[self[profile]]
		return v[self.index(role), self.index(role,strategy)]


	def resample(self, pair="game"):
		"""
		Overwrites self.values with a bootstrap resample of self.sample_values.

		pair = payoff: resample all payoff observations independently
		pair = profile: resample paired profile observations
		pair = game: resample paired game observations
		"""
		if pair == "payoff":
			raise NotImplementedError("TODO")
		elif pair == "profile":#TODO: handle ragged arrays
			self.values = map(lambda p: np.average(p, 2, weights= \
					np.random.multinomial(p.shape[2], np.ones( \
					p.shape[2])/p.shape[2])), self.sample_values)
		elif pair == "game":#TODO: handle ragged arrays
			if isinstance(self.sample_values, list):
				self.sample_values = np.array(self.sample_values, dtype=float)
			s = self.sample_values.shape[3]
			self.values = np.average(self.sample_values, 3, weights= \
					np.random.multinomial(s, np.ones(s)/s))

	def singleSample(self):
		"""Makes self.values be a single sample from each sample set."""
		if self.max_samples == self.min_samples:
			self.makeArrays()
			vals = self.sample_values.reshape([prod(self.values.shape), \
												self.max_samples])
			self.values = np.array(map(choice, vals)).reshape(self.values.shape)
		else:
			self.values = np.array([[[choice(s) for s in r] for r in p] for \
								p in self.sample_values])

	def reset(self):#TODO: handle ragged arrays
		self.values = map(lambda p: np.average(p,2), self.sample_values)

	def makeArrays(self):
		self.sample_values = np.array(self.sample_values)
		Game.makeArrays(self)

	def toJSON(self):
		"""
		Convert to JSON according to the EGTA-online v3 default game spec.
		"""
		game_dict = {}
		game_dict["players"] = self.players
		game_dict["strategies"] = self.strategies
		game_dict["profiles"] = []
		for prof in self:
			game_dict["profiles"].append({role:[(strat, prof[role][strat], \
					list(self.sample_values[self[prof]][self.index(role), \
					self.index(role, strat)])) for strat in prof[role]] for \
					role in prof})
		return game_dict

	def to_TB_JSON(self):
		"""
		Convert to JSON according to the EGTA-online v3 sample-game spec.
		"""
		game_dict = {}
		game_dict["roles"] = [{"name":role, "count":self.players[role], \
					"strategies": list(self.strategies[role])} for role \
					in self.roles]
		game_dict["profiles"] = []
		for prof in self:
			p = self[prof]
			obs = {"observations":[]}
			for i in range(self.sample_values[self[prof]].shape[2]):
				sym_groups = []
				for r, role in enumerate(self.roles):
					for strat in prof[role]:
						s = self.index(role, strat)
						sym_groups.append({"role":role, "strategy":strat, \
								"count":self.counts[p][r,s], \
								"payoff":float(self.sample_values[p][r,s,i])})
				obs["observations"].append({"symmetry_groups":sym_groups})
			game_dict["profiles"].append(obs)
		return game_dict

	def __repr__(self):
		if self.min_samples < self.max_samples:
			return Game.__repr__(self) + "\n" + str(self.min_samples) + \
				"-" + str(self.max_samples) + " samples per profile"
		return Game.__repr__(self) + "\n" + str(self.max_samples) + \
			" samples per profile"
