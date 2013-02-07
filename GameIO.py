#! /usr/bin/env python2.7

from urllib import urlopen
from json import loads, dumps
from xml.dom.minidom import parseString, Document
from collections import Iterable, Mapping
from functools import partial

from HashableClasses import h_array
from BasicFunctions import flatten, one_line
from RoleSymmetricGame import *


def read(source):
	if isinstance(source, basestring):
		try: #assume source is a filename or url
			u = urlopen(source)
			data = u.read()
			u.close()
		except: #assume source is already a data string
			data = source
	elif hasattr(source, 'read'): #source is file-like
		data = source.read()
		source.close()
	else:
		raise IOError(one_line("could not read: " + str(source), 71))
	return detect_and_load(data)


def detect_and_load(data_str):
	json_data = load_JSON(data_str)
	if json_data != None:
		return read_JSON(json_data)
	xml_data = load_XML(data_str)
	if xml_data != None:
		return read_XML(xml_data)
	if is_NFG(data_str):
		return read_NFG(data_str)
	if is_NE(data_str):
		return read_NE(data_str)
	raise IOError(one_line("could not detect format: " + data_str, 71))


def load_JSON(data):
	if isinstance(data, list) or isinstance(data, dict):
		return data
	try:
		return loads(data)
	except:
		return None


def load_XML(data):
	if isinstance(data, Document):
		return data
	try:
		return parseString(data)
	except:
		return None


def is_NFG(data):
	return data.startswith("NFG")


def is_NE(data):
	return data.startswith("NE")


def read_JSON(data):
	"""
	Convert loaded json data (list or dict) into GameAnalysis classes.
	"""
	if isinstance(data, list):
		return map(read_JSON, data)
	if isinstance(data, dict):
		if "object" in data: # game downloaded from EGATS
			return read_JSON(loads(data['object']))
		if "profiles" in data:
			return read_game_JSON(data)
		if "symmetry_groups" in data:
			return read_v3_profile(data)
		if "observations" in data:
			if "players" in data["observations"]["symmetry_groups"]:
				return read_v3_players_profile(data)
			else:
				return read_v3_samples_profile(data)
		if "sample_count" in data:
			return read_v2_profile(data)
		if "type" in data and data["type"] == "GA_Profile":
			return read_GA_profile(data)
		return {k:read_JSON(v) for k,v in data.items()}
	return data


def read_game_JSON(gameJSON):
	if "players" in gameJSON and "strategies" in gameJSON:
		return read_GA_game(gameJSON)
	elif len(gameJSON["profiles"]) == 0:
		return Game(*parse_roles(gameJSON["roles"]))
	elif "symmetry_groups" in gameJSON["profiles"][0]:
		return read_game_JSON_v3(gameJSON)
	elif "observations" in gameJSON["profiles"][0]:
		if "players" in gameJSON["profiles"][0]["observations"][0][ \
				"symmetry_groups"][0]:
			return read_game_JSON_v3_players(gameJSON)
		else:
			return read_game_JSON_v3_samples(gameJSON)
	elif "strategy_array" in gameJSON["roles"][0]:
		return read_game_JSON_old(gameJSON)
	elif "strategies" in gameJSON["roles"][0]:
		return read_game_JSON_v2(gameJSON)
	else:
		raise IOError(one_line("invalid game JSON: " + str(gameJSON), 71))


def read_GA_game(gameJSON):
	if len(gameJSON["profiles"]) > 0 and isinstance(gameJSON[ \
			"profiles"][0].values()[0][0][2], list):
		game_type = SampleGame
	else:
		game_type = Game
	return game_type(gameJSON["players"].keys(), gameJSON["players"], \
			gameJSON["strategies"], map(lambda p: {r:[PayoffData(*scv) for \
			scv in p[r]]for r in p}, gameJSON["profiles"]))


def read_game_JSON_new(gameJSON, game_type, profile_reader):
	roles, players, strategies = parse_roles(gameJSON["roles"])
	profiles = []
	if "profiles" in gameJSON:
		for profileJSON in gameJSON["profiles"]:
			profiles.append(profile_reader(profileJSON))
	return game_type(roles, players, strategies, profiles)


def parse_roles(rolesJSON):
	players = {r["name"] : int(r["count"]) for r in rolesJSON}
	strategies = {r["name"] : r["strategies"] for r in rolesJSON}
	roles = list(players.keys())
	return roles, players, strategies


def read_game_JSON_old(json_data):
	players = {r["name"] : int(r["count"]) for r in json_data["roles"]}
	strategies = {r["name"] : r["strategy_array"] for r in json_data["roles"]}
	roles = list(players.keys())
	profiles = []
	for profileDict in json_data["profiles"]:
		profile = {r:[] for r in roles}
		prof_strat = {}
		for role_str in profileDict["proto_string"].split("; "):
			role, strategy_str = role_str.split(": ")
			prof_strat[role] = strategy_str.split(", ")
		for roleDict in profileDict["roles"]:
			role = roleDict["name"]
			for strategyDict in roleDict["strategies"]:
				s = strategyDict["name"]
				profile[role].append(PayoffData(str(s), int(prof_strat[role] \
						.count(s)), float(strategyDict["payoff"])))
		profiles.append(profile)
	return Game(roles, players, strategies, profiles)


def read_v2_profile(profileJSON):
	profile = {r["name"]:[] for r in profileJSON["roles"]}
	for roleDict in profileJSON["roles"]:
		role = roleDict["name"]
		for strategyDict in roleDict["strategies"]:
			profile[role].append(PayoffData(str(strategyDict[ \
					"name"]), int(strategyDict["count"]), \
					float(strategyDict["payoff"])))
	return profile


def read_v3_profile(profileJSON):
	prof = {}
	for sym_grp in profileJSON["symmetry_groups"]:
		if sym_grp["role"] not in prof:
			prof[sym_grp["role"]] = []
		prof[sym_grp["role"]].append(PayoffData(sym_grp["strategy"], \
				sym_grp["count"], sym_grp["payoff"]))
	return prof


def read_v3_samples_profile(profileJSON):
	prof = {}
	for obs in profileJSON["observations"]:
		for sym_grp in obs["symmetry_groups"]:
			role = sym_grp["role"]
			if role not in prof:
				prof[role] = {}
			strat = sym_grp["strategy"]
			count = sym_grp["count"]
			value = sym_grp["payoff"]
			if (strat, count) not in prof[role]:
				prof[role][(strat, count)] = []
			prof[role][(strat, count)].append(value)
	return {r:[PayoffData(sc[0], sc[1], v) for sc,v in prof[r].items()] for \
			r in prof}


def read_v3_players_profile(profileJSON):
	raise NotImplementedError


def read_GA_profile(profileJSON):
	try:
		return Profile(profileJSON["data"])
	except KeyError:
		return Profile(profileJSON)


read_game_JSON_v3 = partial(read_game_JSON_new, game_type=Game, \
		profile_reader=read_v3_profile)
read_game_JSON_v3_samples = partial(read_game_JSON_new, game_type=SampleGame, \
		profile_reader=read_v3_samples_profile)
read_game_JSON_v3_players = partial(read_game_JSON_new, game_type=SampleGame, \
		profile_reader=read_v3_players_profile)
read_game_JSON_v2 = partial(read_game_JSON_new, game_type=Game, \
		profile_reader=read_v2_profile)


def read_XML(data):
	"""
	Convert loaded xml data (Document) into GameAnalysis classes.
	"""
	gameNode = data.getElementsByTagName("nfg")[0]
	if len(gameNode.getElementsByTagName("player")[0]. \
			getElementsByTagName("action")) > 0:
		return parse_strategic_XML(gameNode)
	return parse_symmetric_XML(gameNode)


def parse_strategic_XML(gameNode):
	strategies = {p.getAttribute('id') : map(lambda s: s.getAttribute('id'), \
			p.getElementsByTagName('action')) for p in \
			gameNode.getElementsByTagName('player')}
	roles = list(strategies.keys())
	counts = {r:1 for r in roles}
	payoffs = []
	for payoffNode in gameNode.getElementsByTagName('payoff'):
		data = {r:[] for r in roles}
		for outcomeNode in payoffNode.getElementsByTagName('outcome'):
			role = outcomeNode.getAttribute('player')
			strategy = outcomeNode.getAttribute('action')
			value = float(outcomeNode.getAttribute('value'))
			data[role].append(PayoffData(strategy, 1, value))
		payoffs.append(data)
	return Game(roles, counts, strategies, payoffs)


def parse_symmetric_XML(gameNode):
	roles = ["All"]
	counts= {"All" : len(gameNode.getElementsByTagName("player"))}
	strategies = {"All" : [e.getAttribute("id") for e in \
			gameNode.getElementsByTagName("action")]}
	payoffs = []
	for payoffNode in gameNode.getElementsByTagName("payoff"):
		data = []
		for outcomeNode in payoffNode.getElementsByTagName("outcome"):
			strategy = outcomeNode.getAttribute("action")
			count = int(outcomeNode.getAttribute("count"))
			value = float(outcomeNode.getAttribute("value"))
			data.append(PayoffData(strategy, count, value))
		payoffs.append({"All":data})
	return Game(roles, counts, strategies, payoffs)


def read_NFG(data_str):
	if data_str.split('"')[1].lower().startswith("symmetric"):
		return read_NFG_sym(data_str)
	if data_str.split('"')[1].lower().startswith("role symmetric"):
		return read_NFG_rsym(data_str)
	return read_NFG_asym(data_str)


def read_NFG_sym(data):
	raise NotImplementedError("TODO")


def read_NFG_rsym(data):
	raise NotImplementedError("TODO")


def read_NFG_asym(data):
	raise NotImplementedError("TODO")


def read_NE(data):
	prob_strs = data.split(",")[1:]
	probs = []
	for s in prob_strs:
		try:
			num, denom = s.split("/")
			probs.append(float(num) / float(denom))
		except ValueError:
			probs.append(float(s))
	return h_array(probs)


def to_JSON_str(obj, indent=2):
	return dumps(to_JSON_obj(obj), sort_keys=True, indent=indent)


def to_JSON_obj(obj):
	if hasattr(obj, "toJSON"):
		return obj.toJSON()
	if hasattr(obj, "items"):
		if all([hasattr(k, "toJSON") for k in obj.keys()]):
			return {to_JSON_obj(k):to_JSON_obj(v) for k,v in obj.items()}
		return {k:to_JSON_obj(v) for k,v in obj.items()}
	if hasattr(obj, "__iter__"):
		return map(to_JSON_obj, obj)
	return loads(dumps(obj))


def to_XML(game):
	if len(game.roles) == 1:
		to_symmetric_XML(game)
	elif all(map(lambda c: c==1, game.counts.values())):
		to_strategic_XML(game)
	else:
		raise NotImplementedError("no EGAT XML spec for role-symmetric games")


def to_symmetric_XML(game):
	"""
	Writes game to XML according to the EGAT symmetric game spec.
	Assumes (but doesn't check) that game is symmetric.
	"""
	raise NotImplementedError("TODO")


def to_strategic_XML(game):
	"""
	Writes game to XML according to the EGAT strategic game spec.
	Assumes (but doesn't check) that game is not role-symmetric.
	"""
	raise NotImplementedError("TODO")


def to_NFG(game):
	if isinstance(game, list):
		return map(to_NFG, game)
	if is_symmetric(game):
		return to_NFG_sym(game)
	if is_asymmetric(game):
		return to_NFG_asym(game)
	return to_NFG_rsym(game)


def to_NFG_sym(game):
	output = 'NFG 1 R "symmetric"\n'
	raise NotImplementedError("TODO")


def to_NFG_asym(game):
	output = 'NFG 1 R "asymmetric"\n{ '
	output += " ".join(('"' + str(r) + '"' for r in game.roles)) + " } { "
	output += " ".join(map(str, game.numStrategies)) + " }\n\n"
	prof = Profile({r:{game.strategies[r][0]:1} for r in game.roles})
	last_prof = Profile({r:{game.strategies[r][-1]:1} for r in game.roles})
	while prof != last_prof:
		prof_strat = {r:prof[r].keys()[0] for r in game.roles}
		output += NFG_payoffs(game, prof) + " "
		prof = increment_profile(game, prof)
	output += NFG_payoffs(game, last_prof) + "\n"
	return output

def NFG_payoffs(game, prof):
	return  " ".join((str(game.getPayoff(prof, r, prof[r].keys()[0])) \
			for r in game.roles))

def increment_profile(game, prof):
	for role in game.roles:
		strat = prof[role].keys()[0]
		i = game.index(role, strat)
		if i < game.numStrategies[game.index(role)] - 1:
			return prof.deviate(role, strat, game.strategies[role][i+1])
		prof = prof.deviate(role, strat, game.strategies[role][0])
	return prof


def to_NFG_rsym(game):
	output = 'NFG 1 R "role-symmetric: ' + ", ".join([str(game.players[r]) + \
			" " + r for r in game.roles]) + '"\n'
	raise NotImplementedError("TODO")


class io_parser(ArgumentParser):
	def __init__(self, *args, **kwargs):
		ArgumentParser.__init__(self, *args, **kwargs)
		self.add_argument("-input", type=str, default="", help= \
				"Input file. Defaults to stdin.")
		self.add_argument("-output", type=str, default="", help= \
				"Output file. Defaults to stdout.")

	def parse_args(self, *args, **kwargs):
		a = ArgumentParser.parse_args(self, *args, **kwargs)
		if a.input == "":
			a.input = read(sys.stdin.read())
		elif a.input != "None":#programs not requiring input can set it to None
			i = open(a.input)
			a.input = read(i.read())
			i.close()
		if a.output != "":
			sys.stdout = open(a.output, "w")
		return a


def parse_args():
	parser = io_parser()
	parser.add_argument("-format", choices=["json", "xml", "nfg"], \
			default="json", help="Output format.")
	return parser.parse_args()


def main():
	args = parse_args()
	if args.format == "json":
		print to_JSON_str(args.input)
	elif args.format == "xml":
		print to_XML(args.input)
	elif args.format == "nfg":
		print to_NFG(args.input)


if __name__ == "__main__":
	main()
