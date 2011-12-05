import numpy as np

from math import factorial, isnan
from operator import mul, add
from random import choice
from itertools import combinations_with_replacement as CwR, permutations, \
		product

def nCr(n, k):
	"""
	Number of combinations: n choose k.
	"""
	return reduce(mul, range(n-k+1,n+1)) / factorial(k)

def game_size(n,s):
	"""
	Number of profiles in a symmetric game with n players and s strategies.
	"""
	return nCr(n+s-1,n)

def list_repr(l, sep=", "):
	try:
		return reduce(lambda x,y: str(x) + sep + str(y), l)
	except TypeError:
		return ""

class SymmetricProfile(tuple):
	"""
	Profile object for symmetric games: a sorted tuple of strategies.
	"""
	def __new__(self, strategies):
		return tuple.__new__(self, sorted(strategies))

	def getStrategies(self):
		"""without repetitions"""
		return sorted(set(self))

	def remove(self, strategy):
		"""n-1 player profile excluding strategy"""
		i = self.index(strategy)
		return SymmetricProfile(self[:i] + self[i+1:])

	def add(self, strategy):
		"""n+1 player profile including strategy"""
		return SymmetricProfile(list(self) + [strategy])

	def probability(self, profile):
		"""only makes sense for mixed strategy profiles"""
		try:
			return reduce(add, [reduce(mul, [self[i][s] for i,s in \
					enumerate(p)]) for p in set(permutations(profile))])
		except TypeError:
			return 1

	def dist(self, other):
		return sum([s.dist(o) for s,o in zip(self, other)])

	def isMixed(self):
		return any([isinstance(s, mixture) for s in self])

	def isFullyMixed(self):
		return all([isinstance(s, mixture) for s in self])

	def __repr__(self):
		return "(" + list_repr((repr(self.count(s)) + "x" + \
				repr(s) for s in sorted(set(self)))) + ")"

	def repetitions(self):
		"""number of orderings of the profile that could occur"""
		try:
			return self._reps
		except:
			self._reps = factorial(len(self)) / reduce(mul, (factorial( \
					self.count(s)) for s in sorted(set(self))))
			return self._reps


class Profile(dict):
	"""
	Profile object for RoleSymmetricGames: maps roles to SymmetricProfiles.
	"""
	def _blocked_attribute(obj):
		raise TypeError("profiles are immutable.")
	_blocked_attribute = property(_blocked_attribute)

	__delitem__ = clear = pop = popitem = setdefault = _blocked_attribute

	def __new__(self, *args, **kwargs):
		new = dict.__new__(self, *args, **kwargs)
		self.__setitem__ = self.update = self._blocked_attribute
		return new

	def remove(self, role, strategy):
		return Profile({r:self[r] if r != role else self[r].remove(strategy) \
				for r in self})

	def add(self, role, strategy):
		return Profile({r:self[r] if r != role else self[r].add(strategy) \
				for r in self})

	def probability(self, profile):
		"""only makes sense for mixed strategy profiles"""
		return reduce(mul, [sp.probability(profile[r]) for r, sp \
				in self.items()])

	def dist(self, other):
		return sum([self[r].dist(other[r]) for r in self.keys()])

	def isMixed(self):
		return any([p.isMixed() for p in self.values()])

	def isFullyMixed(self):
		return all([p.isFullyMixed() for p in self.values()])

	def __hash__(self):
		try:
			return self._hash
		except AttributeError:
			self._hash = hash(tuple(sorted(self.items())))
			return self._hash

	def repetitions(self):
		"""Combinations of orderings of role profiles that could occur."""
		return reduce(mul, [p.repetitions() for p in self.values()])


class mixture(np.ndarray):
	"""
	Symmetric mixed strategy.

	mixture(data) sets negative values to 0, then normalizes the sum to 1. ex:
	mixture( range(-2,5) ) --> mixture([ 0. , 0. , 0. , 0.1 , 0.2 , 0.3 , 0.4 ])

	Probabilities are rounded to the nearest 10^-10 so they can be compared.
	"""
	def __new__(cls, strategies, probabilities):
		a = np.array(probabilities, dtype=float).clip(0)
		if a.max() == 0:
			a.fill(1)
		a = np.ndarray.__new__(cls, shape=a.shape, buffer=a/a.sum())
		a.strategies = dict(zip(strategies, a))
		return a

	def __getitem__(self, item):
		try:
			return np.ndarray.__getitem__(self, item)
		except ValueError:
			return self.strategies[item]

	def getStrategies(self):
		return list(self.strategies.keys())

	def __repr__(self):
		try:
			return "{" + list_repr((str(s) + ":" + str(int(round(100*p))) + \
					"%" for s,p in sorted(self.strategies.items()))) + "}"
		except AttributeError:
			return np.ndarray.__repr__(self)[8:-1]

	def __str__(self):
		return repr(self)

	def __hash__(self):
		try:
			return self._hash
		except AttributeError:
			self._hash = hash(tuple(self))
			return self._hash

	def dist(self, other):
		return np.linalg.norm(self - other)

	def __eq__(self, other):
		return self.dist(other) == 0

	def __lt__(self, other):
		try:
			return self.strategies < other.strategies or list(self) < \
					list(other)
		except AttributeError as ae:
			if isinstance(other, str):
				return True
			else:
				raise ae


class Game(dict):
	def __init__(self, roles=[], counts={}, strategies={}, payoffs={}):
		"""
		Role-symmetric game representation.

		__init__ parameters:
		roles: list of role-name strings
		counts: mapping from roles to number of players per role
		strategies: mapping from roles to per-role strategy sets
		payoffs: mapping from Profile objects to payoff dicts, which map
				roles to strategies to numbers
		"""
		self.roles = tuple(sorted(set(map(str, roles))))
		self.counts = {r : int(counts[r]) for r in self.roles}
		self.strategies = {r : tuple(sorted(set(map(str, strategies[r])))) \
				for r in self.roles}
		self.size = reduce(mul, map(lambda r: game_size(self.counts[r], \
				len(self.strategies[r])), self.roles))
		dict.update(self, payoffs)

	def __setitem__(self, profile, payoffs):
		assert self.isValidProfile(profile)
		dict.__setitem__(self, profile, payoffs)

	def update(self, *args, **kwargs):
		d = {}
		d.update(*args, **kwargs)
		for profile in d.keys():
			assert self.isValidProfile(profile)
		dict.update(self, d)

	def __hash__(self):
		"""Payoff data doesn't contribute to the hash."""
		try:
			return self._hash
		except AttributeError:
			self._hash = hash(str(self.roles) + str(self.counts) + \
					str(self.strategies))
			return self._hash

	def isValidProfile(self, profile):
		if profile in self:
			return True
		if not isinstance(profile, Profile):
			return False
		for r, sp in profile.items():
			if r not in self.roles:
				return False
			if not isinstance(sp, SymmetricProfile):
				return False
			if len(sp) != self.counts[r]:
				return False
			if not all((s in self.strategies[r] for s in sp.getStrategies())):
				return False
		return True

	def allProfiles(self):
		return [Profile(zip(self.roles, p)) for p in product(*[[ \
				SymmetricProfile(s) for s in CwR(self.strategies[r], \
				self.counts[r])] for r in self.roles])]

	def knownProfiles(self):
		return self.keys()

	def subgame(self, roles=[], strategies={}):
		"""
		Creates a game with a subset each role's strategies.
		Raises a KeyError if required profiles are missing.

		default settings result in a subgame with all roles and no strategies
		"""
		if not roles:
			roles = self.roles
		if not strategies:
			strategies = {r:[] for r in self.roles}
		g = Game(roles, self.counts, strategies)
		g.update({p:self[p] for p in g.allProfiles()})
		return g

	def isSubgame(self, big_game):
		if any([r not in big_game.roles for r in self.roles]):
			return False
		if any([self.counts[r] != big_game.counts[r] for r in self.roles]):
			return False
		for r in self.roles:
			if any([s not in big_game.strategies[r] for s in \
					self.strategies[r]]):
				return False
		return True

	def __eq__(self, other):
		return self.roles==other.roles and self.counts==other.counts and \
				self.strategies==other.strategies and dict.__eq__(self,other)

	def __repr__(self):
		return "RoleSymmetricGame:\nroles: " + list_repr(self.roles) + \
				"\ncounts: " + str(self.counts) + "\nstrategies: " + \
				str(self.strategies) + "\npayoff data for " + \
				str(len(self)) + " out of " + str(self.size) + " profiles"

	def getPayoff(self, profile, role, strategy):
		#try to look up payoff for pure strategy & profile
		try:
			return self[profile][role][strategy]
		except KeyError as ke:
			if not (isinstance(strategy, mixture) or profile.isMixed()):
				raise ke
		#try to compute expected payoff for mixed strategy & profile
		if not isinstance(strategy, mixture):
			strategy = mixture(self.strategies[role], [1 if s==strategy \
					else 0 for s in self.strategies[role]])
		if not profile.isFullyMixed():
			rsp = {}
			for r in self.roles:
				if profile[r].isFullyMixed():
					rsp[r] = profile[r]
					continue
				sp = []
				for s in profile[r]:
					if isinstance(s, mixture):
						sp.append(s)
					else:
						sp.append(mixture(self.strategies[r], [1 if strat==s \
								else 0 for strat in self.strategies[r]]))
				rsp[r] = SymmetricProfile(sp)
			profile = Profile(rsp)
		payoff = 0
		opponent_profile = profile.remove(role, strategy)
		for op in self.roleSymmetricProfiles(opponent_profile):
			for s in self.strategies[role]:
				p = op.add(role, s)
				payoff += self[p][role][s]*opponent_profile.probability(op) \
						* strategy[s]
		return payoff

	def regret(self, profile):
		regret = 0
		for role, symProf in profile.items():
			for strategy in symProf.getStrategies():
				payoff = self.getPayoff(profile, role, strategy)
				deviations = self.deviations(profile, role, strategy)
				for dev_strat, dev_prof in deviations:
					r = self.getPayoff(dev_prof, role, dev_strat) - payoff
					if r > regret:
						regret = r
		return regret

	def deviations(self, profile, role, strategy):
		"""
		Returns a list of pairs (s,p), where s is a strategy for %role and p is
		the profile that results from deviating from %strategy to s in %profile.
		"""
		neighbors = []
		for s in set(self.strategies[role]) - {strategy}:
			strategy_list = list(profile[role])
			strategy_list.remove(strategy)
			strategy_list.append(s)
			p = Profile(profile)
			dict.__setitem__(p, role, SymmetricProfile(strategy_list))
			neighbors.append((s,p))
		return neighbors

	def BR(self, role, opponent_profile):
		best_responses = []
		best_payoff = float("-inf")
		for strategy in self.strategies[role]:
			try:
				payoff = self.getPayoff(opponent_profile.add(role, strategy), \
						role, strategy)
			except KeyError:
				continue
			if payoff > best_payoff:
				best_responses = [strategy]
				best_payoff = payoff
			elif payoff == best_payoff:
				best_responses.append(strategy)
		return best_responses

	def uniformMixedProfile(self):
		return Profile({r : SymmetricProfile([mixture(self.strategies[r], \
				[1]*len(self.strategies[r]))]*self.counts[r]) for \
				r in self.roles})

	def biasedMixedProfile(self, role, strategy):
		return Profile({r : SymmetricProfile([mixture(self.strategies[r], \
				[100 if r==role and s==strategy else 1 for s in \
				self.strategies[r]])] * self.counts[r]) for r in self.roles})

	def payoffList(self, role):
		"""
		Returns all payoff floats associated with a role.

		Useful for determining the minimum achievable payoff.
		"""
		return reduce(add, map(lambda d: list(d[role].values()), self.values()))

	def expectedValues(self, rsmsp):
		"""
		Gives the EV of each pure strategy when played against %rsmsp.

		expectedValues() is preferable to getPayoff() for replicator dynamics,
		because it requires only one loop over the game's payoffs.

		input:
		rsmsp: a role-symmetric (one-per-role) mixed strategy profile

		output:
		EVs: mapping from roles to pure strategies to payoffs
		"""
		values = {r:{s:0.0 for s in self.strategies[r]} for r in self.roles}
		total_prob = {r:{s:0.0 for s in self.strategies[r]} for r in self.roles}
		for profile in self:
			prob = rsmsp.probability(profile)
			if prob == 0:
				continue
			for role in self.roles:
				for strategy in profile[role].getStrategies():
					values[role][strategy] += self[profile][role][strategy] \
							* prob / rsmsp[role][0][strategy] * \
							profile[role].count(strategy) / self.counts[role]
		return values

	def symmetricProfiles(self, role, smsp):
		"""
		Return pure strategy symmetric profiles for role that have positive
		probability under symmetric mixed strategy profile smsp.
		"""
		return map(SymmetricProfile, set(map(lambda p: tuple(sorted(p)), \
				product(*[filter(lambda s: mixture[s], self.strategies[role]) \
				for mixture in smsp]))))

	def roleSymmetricProfiles(self, rsmsp):
		"""
		Return pure strategy role-symmetric profiles that have positive
		probability under role-symmetric mixed strategy profile rsmsp
		"""
		return [Profile(zip(self.roles, p)) for p in product(*[ \
				self.symmetricProfiles(r, rsmsp[r]) for r \
				in self.roles])]

