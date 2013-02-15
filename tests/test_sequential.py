from Sequential import ObservationMatrix, EquilibriumCompareEvaluator
from RoleSymmetricGame import PayoffData, Profile
from factories.GameFactory import create_symmetric_game
from dingus import patch
from numpy import array

class describe_observation_matrix:
    def it_can_return_requested_profile_observations(self):
        matrix = ObservationMatrix([{'All': [PayoffData('A', 2, [12, 11])]},
                                    {'All': [PayoffData('A', 1, [23]), PayoffData('B', 1, [11])]},
                                    {'All': [PayoffData('B', 2, [13.3])]}])
        print matrix.getPayoffData(Profile({'All': {'A': 2}}), 'All', 'A')
        assert matrix.getPayoffData(Profile({'All': {'A': 2}}), 'All', 'A') == [12, 11]
        assert matrix.getPayoffData(Profile({'All': {'A': 1, 'B': 1}}), 'All', 'B') == [11]
        
    def it_can_add_observations_to_existing_profiles(self):
        matrix = ObservationMatrix([{'All': [PayoffData('A', 2, [12, 11])]},
                                    {'All': [PayoffData('A', 1, [23]), PayoffData('B', 1, [11])]},
                                    {'All': [PayoffData('B', 2, [13.3])]}])
        matrix.addObservations(Profile({'All': {'A': 1, 'B': 1}}),
                               {'All': [PayoffData('A', 1, [12, 11]), PayoffData('B', 1, [21, 17])]})
        assert matrix.getPayoffData(Profile({'All': {'A': 1, 'B': 1}}), 'All', 'B') == [11, 21, 17]
    
    def it_can_add_observations_for_previously_unobserved_profiles(self):
        matrix = ObservationMatrix()
        matrix.addObservations(Profile({'All': {'A': 1, 'B': 1}}),
                               {'All': [PayoffData('A', 1, [12, 11]), PayoffData('B', 1, [21, 17])]})
        assert matrix.getPayoffData(Profile({'All': {'A': 1, 'B': 1}}), 'All', 'B') == [21, 17]
        
    def it_can_be_transformed_into_a_game(self):
        matrix = ObservationMatrix([{'R1': [PayoffData('A', 1, [12, 11])],
                                     'R2': [PayoffData('C', 2, [15, 16])]},
                                    {'R1': [PayoffData('B', 1, [11])],
                                     'R2': [PayoffData('C', 1, [23]), PayoffData('D', 1, [17])]}])
        game = matrix.toGame()
        assert game.getPayoff(Profile({'R1': {'B': 1}, 'R2': {'C': 1, 'D': 1}}), 'R2', 'D') == 17
        assert game.getPayoff(Profile({'R1': {'A': 1}, 'R2': {'C': 2}}), 'R2', 'C') == 15.5
        assert len(game.knownProfiles()) == 2
        assert game.roles == ['R1', 'R2']
        assert game.players == {'R1': 1, 'R2': 2}
        print game.strategies
        assert game.strategies == {'R1': ('A', 'B'), 'R2': ('C', 'D')}
        
class describe_equilibrium_compare_evaluator:
    def it_requests_further_sampling_when_there_are_no_old_equilibria(self):
        game = create_symmetric_game()
        evaluator = EquilibriumCompareEvaluator(0.01)
        with patch('Sequential.replicator_dynamics'):
            return array([0.1, 0.9])
        assert evaluator.continue_sampling(game) == True
        assert evaluator.old_equilibria == [array([0.1, 0.9])]
    
    def it_requests_further_sampling_when_the_new_equilibrium_is_distant(self):
        game = create_symmetric_game()
        evaluator = EquilibriumCompareEvaluator(0.01)
        evaluator.old_equilibria = [array([0.5, 0.5])]
        with patch('Sequential.replicator_dynamics'):
            return array([0.1, 0.9])
        assert evaluator.continue_sampling(game) == True
        
    def it_stops_sampling_when_the_new_equilibria_are_all_similar(self):
        game = create_symmetric_game()
        evaluator = EquilibriumCompareEvaluator(0.05)
        evaluator.old_equilibria = [array([0.11, 0.89]), array([1.0, 0.0])]
        import Sequential
        Sequential.replicator_dynamics = lambda g, mix, iters, converge_threshold: \
                array([0.1, 0.9]) if mix is array([0.11, 0.89]) else array([0.999, 0.001])
        with patch('Regret.regret'):
            return 0
        assert evaluator.continue_sampling(game) == False

    def it_requests_further_sampling_when_new_equilibria_are_found(self):
        game = create_symmetric_game()
        evaluator = EquilibriumCompareEvaluator(0.05)
        evaluator.old_equilibria = [array([0.11, 0.89])]
        import Sequential
        Sequential.replicator_dynamics = lambda g, mix, iters, converge_threshold: \
                array([0.1, 0.9]) if mix is array([0.11, 0.89]) else array([0.999, 0.001])
        with patch('Regret.regret'):
            return 0
        assert evaluator.continue_sampling(game) == True
        assert evaluator.old_equilibria == [array([0.1, 0.9]), array([0.999, 0.001])]
        