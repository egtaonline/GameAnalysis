from RoleSymmetricGame import Game, PayoffData
from Sequential import ObservationMatrix
from itertools import product, combinations_with_replacement as CwR
import random

def create_symmetric_game(number_of_players=2, strategy_set=['C', 'D'], payoff_data=None):
    payoff_data = payoff_data or create_payoff_data(['All'], {'All': number_of_players},
                                                    {'All': strategy_set})    
    return Game(['All'], {'All': number_of_players}, {'All': strategy_set}, payoff_data)

def create_observation_matrix(number_of_players=2, strategy_set=['C', 'D'], payoff_data=None):
    payoff_data = payoff_data or create_payoff_data(['All'], {'All': number_of_players},
                                                    {'All': strategy_set})    
    return ObservationMatrix(payoff_data)

def create_payoff_data(roles, players, strategies):
    return [{r: [PayoffData(s, p[index].count(s), [random.random()]) for s in set(p[index])] \
            for index, r in enumerate(roles)} \
            for p in product(*[CwR(strategies[r], players[r]) for r in roles])]