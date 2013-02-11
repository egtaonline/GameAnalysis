from dingus import Dingus
from Sequential import SequentialSamplingGame, EquilibriumCompareEvaluator
from factories import GameFactory
from RoleSymmetricGame import Profile, PayoffData

class describe_sequential_sampling_game:
    def it_should_accept_default_payoff_data(self):
        game = SequentialSamplingGame(['All'], {'All': 2}, {'All': ['A', 'B']}, 
                                      [{'All': [PayoffData('A', 2, 10.0)]},
                                       {'All': [PayoffData('A', 1, 5.0), PayoffData('B', 1, 5.0)]},
                                       {'All': [PayoffData('B', 2, 12.0)]}])
        assert game.getPayoff(Profile({'All': {'B': 2}}), 'All', 'B') == 12.0

class describe_equilibrium_compare_evaluator:
    def it_should_return_true_when_game_has_no_samples(self):
        evaluator = EquilibriumCompareEvaluator(0.01)
        game = GameFactory.create_symmetric_sequential_sampling_game()
        assert evaluator.continue_sampling(game) == True