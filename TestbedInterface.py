#! /usr/bin/env python2.7

from json import loads
from string import join

import requests

from GameIO import read, read_v3_profile, read_v3_samples_profile, read_v3_players_profile
from RoleSymmetricGame import Profile
from Reductions import DPR_profiles
from BasicFunctions import one_line


modifiable_attributes = ["process_memory", "name", "parameter_hash", "active", \
						"time_per_sample", "nodes", "samples_per_simulation"]

class TestbedObject:
	def __init__(self, name, obj_type, name_field="", options={}, \
			url_options={}, skip_name=False):
		self.auth = {'auth_token':"g2LHz1mEtbysFngwLMCz"}
		self.url = "http://d-108-249.eecs.umich.edu/api/v3/" + obj_type

		data = dict(options)
		data.update(self.auth)

		ID = name #try treating name as an id
		r = requests.get(self.url +"/"+ ID + ".json?" + "&".join([str(k) +"="+ \
				str(v) for k,v in url_options.items()]), data=data)

		if r.status_code != 200 and not skip_name: #treat name as a name_field
			r = requests.get(self.url + ".json", data=data)
			try:
				ID = filter(lambda s: s[name_field] == name, \
						loads(r.text)[obj_type])[0]['_id']
				print ID
			except IndexError:
				raise Exception("no such "+obj_type[:-1]+": "+name)

		self.url += "/" + ID
		self.processResponse(r)

	def processResponse(self, r):
		self.json = r.text
		j = loads(r.text)
		if isinstance(j, dict):
			self.__dict__.update()
		elif isinstance(j, list):
			map(lambda d: self.__dict__.update(d), j)
		else:
			raise ValueError(one_line(r.text))

	def update(self):
		r = requests.get(self.url + ".json", data=self.auth)
		self.processResponse(r)
		return r.status_code

	def __repr__(self):
		return join(map(lambda p: one_line(repr(p[0]) + ": " + repr(p[1])), \
				self.__dict__.items()), "\n")


class TestbedScheduler(TestbedObject):
	def __init__(self, name, scheduler_type="generic_"):
		TestbedObject.__init__(self, name, scheduler_type+"schedulers", "name")

	def addProfile(self, profile, samples):
		data = {'sample_count':samples, 'assignment':str(profile)}
		data.update(self.auth)
		r = requests.post(self.url + "/add_profile.json", data=data)
		return r.status_code

	def update(self, **kwds):
		assert all((k in modifiable_attributes for k in kwds)), \
				"invalid parameter for update"
		self.__dict__.update(kwds)
		data = {"scheduler":kwds}
		data.update(self.auth)
		r = requests.put(self.url + ".json", data=data)
		return r.status_code


class TestbedSimulator(TestbedObject):
	def __init__(self, name):
		TestbedObject.__init__(self, name, "simulators", "simulator_fullname")

	def updateRoles(self):
		r = requests.get(self.url + ".json", data=self.auth)
		self.roles = loads(r.text)["roles"]
		return r.status_code

	def hasRole(self, role):
		return any([r["name"] == role for r in self.roles])

	def hasStrategy(self, role, strategy):
		return self.hasRole(role) and strategy in filter(lambda r: \
				r["name"] == role, self.roles)[0]["strategies"]

	def addRole(self, role):
		data = {"role":role}
		data.update(self.auth)
		r = requests.post(self.url +"/"+ "add_role.json", data=data)
		self.updateRoles()
		return r.status_code

	def addStrategy(self, role, strategy):
		assert self.hasRole(role), "no such role: " + role
		data = {"role":role, "strategy":strategy}
		data.update(self.auth)
		r = requests.post(self.url +"/"+ "add_strategy.json", data=data)
		self.updateRoles()
		return r.status_code

	def removeRole(self, role):
		data = {"role":role}
		data.update(self.auth)
		r = requests.post(self.url +"/"+ "remove_role.json", data=data)
		self.updateRoles()
		return r.status_code

	def removeStrategy(self, role, strategy):
		assert self.hasRole(role), "no such role: " + role
		data = {"role":role, "strategy":strategy}
		data.update(self.auth)
		r = requests.post(self.url +"/"+ "remove_strategy.json", data=data)
		self.updateRoles()
		return r.status_code


class TestbedGame(TestbedObject):
	def __init__(self, name, granularity="structure"):
		TestbedObject.__init__(self, name, "games", "name", {"granularity": \
				granularity})
		self.game = read(self.json)

	def update(self):
		TestbedObject.update(self)
		self.game = read(self.json)


class TestbedProfile(TestbedObject):
	def __init__(self, ID, granularity="summary"):
		TestbedObject.__init__(self, ID, "profiles", url_options= \
				{"granularity":granularity}, skip_name=True)
		self.granularity = granularity
		if self.granularity == "summary":
			self.payoffs = read_v3_profile(loads(self.json)[0])
		elif self.granularity == "observations":
			self.payoffs = read_v3_samples_profile(loads(self.json)[0])
		elif self.granularity == "full":
			self.payoffs = read_v3_players_profile(loads(self.json)[0])

	def update(self):
		TestbedObject.update(self)
		if self.granularity == "summary":
			self.payoffs = read_v3_profile(loads(self.json)[0])
		elif self.granularity == "observations":
			self.payoffs = read_v3_samples_profile(loads(self.json)[0])
		elif self.granularity == "full":
			self.payoffs = read_v3_players_profile(loads(self.json)[0])



class DPR_scheduler:
	def __init__(self, scheduler_name, game_name, players, samples):
		self.TB_scheduler = TestbedScheduler(scheduler_name)
		self.TB_game = TestbedGame(game_name)
		self.players = {r['name']:p for r,p in zip(self.TB_game.roles, players)}
		self.samples = samples
		for profile in DPR_profiles(self.TB_game.game, self.players):
			self.TB_scheduler.addProfile(profile, self.samples)

	def addSamples(self, additional_samples):
		self.samples += additional_samples
		for profile in DPR_profiles(self.TB_game.game, self.players):
			self.TB_scheduler.addProfile(profile, self.samples)


from argparse import ArgumentParser

def main():
	parser = ArgumentParser()
	parser.add_argument("scheduler", type=str, help="Name or ID of testbed " +\
			"scheduler.")
	parser.add_argument("-game", type=str, default="", help="Name or ID of " +\
			"testbed scheduler. May be omitted if it's the same as the " +\
			"specified scheduler name.")
	parser.add_argument("samples", type=int, help="Number of samples to " +\
			"schedule for each profile.")
	parser.add_argument("players", type=int, nargs="*", help="Number of " +\
			"players each reduced-game role should have.")
	args = parser.parse_args()
	if args.game == "":
		args.game = args.scheduler
	scheduler = DPR_scheduler(args.scheduler, args.game, args.players, \
			args.samples)

if __name__ == "__main__":
	main()
