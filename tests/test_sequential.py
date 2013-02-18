from Sequential import ObservationMatrix, EquilibriumCompareEvaluator, StandardErrorEvaluator
from RoleSymmetricGame import PayoffData, Profile
from factories.GameFactory import create_observation_matrix
from numpy import array
from numpy.testing import assert_array_almost_equal
import Nash
import Regret

class describe_observation_matrix:
    def it_can_return_requested_profile_observations(self):
        matrix = ObservationMatrix([{'All': [PayoffData('A', 2, [12, 11])]},
                                    {'All': [PayoffData('A', 1, [23]), PayoffData('B', 1, [11])]},
                                    {'All': [PayoffData('B', 2, [13.3])]}])
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
        assert game.strategies == {'R1': ('A', 'B'), 'R2': ('C', 'D')}
        
class describe_equilibrium_compare_evaluator:
    Regret.regret = lambda game, profile: 0
    
    def it_requests_further_sampling_when_there_are_no_old_equilibria(self):
        Nash.replicator_dynamics = lambda g, mix, iters, converge_threshold: array([0.1, 0.9])
        matrix = create_observation_matrix()
        evaluator = EquilibriumCompareEvaluator(0.01)
        assert evaluator.continue_sampling(matrix) == True
        assert_array_almost_equal(evaluator.old_equilibria[0], array([0.1, 0.9]))
    
    def it_requests_further_sampling_when_the_new_equilibrium_is_distant(self):
        Nash.replicator_dynamics = lambda g, mix, iters, converge_threshold: array([0.1, 0.9])
        matrix = create_observation_matrix()
        evaluator = EquilibriumCompareEvaluator(0.01)
        evaluator.old_equilibria = [array([0.5, 0.5])]
        assert evaluator.continue_sampling(matrix) == True
        
    def it_stops_sampling_when_the_new_equilibria_are_all_similar(self):
        Nash.replicator_dynamics = lambda g, mix, iters, converge_threshold: \
                array([0.110001, 0.899999]) if list(mix) == [0.11, 0.89] else array([0.9999, 0.0001])
        matrix = create_observation_matrix()
        evaluator = EquilibriumCompareEvaluator(0.05)
        evaluator.old_equilibria = [array([0.11, 0.89]), array([1.0, 0.0])]
        assert evaluator.continue_sampling(matrix) == False

    def it_requests_further_sampling_when_new_equilibria_are_found(self):
        Nash.replicator_dynamics = lambda g, mix, iters, converge_threshold: \
                array([0.1, 0.9]) if mix is array([0.11, 0.89]) else array([0.999, 0.001])
        matrix = create_observation_matrix()
        evaluator = EquilibriumCompareEvaluator(0.05)
        evaluator.old_equilibria = [array([0.11, 0.89])]
        assert evaluator.continue_sampling(matrix) == True
        for eq in evaluator.old_equilibria:
            assert [[0.1, 0.9], [0.999, 0.001]].count(list(eq)) == 1
        
class describe_standard_error_evaluator:
    def it_requests_sampling_when_profiles_in_target_set_have_too_much_variation_in_payoffs(self):
        matrix = ObservationMatrix()
        target_profile = Profile({'All': {'A': 2}})
        matrix.addObservations(target_profile, {'All': [PayoffData('A', 2, [10, 20, 45])]})
        evaluator = StandardErrorEvaluator(1, [target_profile])
        assert evaluator.continue_sampling(matrix) == True

    def it_does_not_request_sampling_when_profiles_in_target_set_do_not_have_too_much_variation_in_payoff(self):
        matrix = ObservationMatrix()
        target_profile = Profile({'All': {'A': 2}})
        matrix.addObservations(target_profile, {'All': [PayoffData('A', 2, [10, 10.1, 9.9])]})
        evaluator = StandardErrorEvaluator(1, [target_profile])
        assert evaluator.continue_sampling(matrix) == False        

    def it_does_not_request_sampling_when_only_profiles_outside_target_set_have_too_much_variation_in_payoffs(self):
        matrix = ObservationMatrix()
        target_profile = Profile({'All': {'A': 2}})
        non_target_profile = Profile({'All': {'B': 2}})
        matrix.addObservations(target_profile, {'All': [PayoffData('A', 2, [10, 10.1, 10.2])]})
        matrix.addObservations(non_target_profile, {'All': [PayoffData('B', 2, [10, 20, 45])]})
        evaluator = StandardErrorEvaluator(1, [target_profile])
        assert evaluator.continue_sampling(matrix) == False
        