from RoleSymmetricGame import SampleGame, PayoffData
from RandomGames import generate_normal_noise
from Nash import mixed_nash
from numpy.linalg import norm

class SequentialSamplingGame(SampleGame):
    def __init__(self, roles=[], players={}, strategies={}, default_payoff_values=[]):
        SampleGame.__init__(roles, players, strategies)

def sequential_normal_noise(game, stdev, evaluator, sample_increment):
    """
    Creates a game with normal noise sequentially
    
    game - provides the necessary structure, i.e. the roles, strategies, and base payoffs
    stdev - the standard deviation for use with normal noise generation
    evaluator - an object that can evaluate whether or not to continue sampling by inspecting game
    sample_increment - the number of samples to take in each step    
    """
    g = SequentialSamplingGame(game.roles, game.players, game.strategies)
    while evaluator.continue_sampling(g):
        new_g = SampleGame(game.roles, game.players, game.strategies)
        for profile in game.knownProfiles():
            new_data = generate_normal_noise(game, profile, stdev, sample_increment)
            payoff_data = {}
            for r in game.roles:
                role_array = []
                for entry in new_data[r]:
                    s = entry.strategy
                    
                    role_array.append(PayoffData(s, profile[r][s], 
                                g.getPayoffData(profile, r, s) + entry.value))
                payoff_data[r] = role_array
            new_g.addProfile(payoff_data)
        g = new_g
    return g

class EquilibriumCompareEvaluator:
    def __init__(self, compare_threshold, regret_threshold=1e-4, dist_threshold=None, 
                 random_restarts=0, iters=10000, converge_threshold=1e-8):
        self.compare_threshold = compare_threshold
        self.regret_threshold = regret_threshold
        self.dist_threshold = dist_threshold or compare_threshold/10.0
        self.random_restarts = random_restarts
        self.iters = iters
        self.converge_threshold = converge_threshold
        self.old_equilibrium = None
        
    def continue_sampling(self, game):
        if game.max_samples is 0:
            return True
        else:
            new_equilibrium = mixed_nash(game, self.regret_threshold, self.dist_threshold,
                    self.random_restarts, True, self.iters, self.converge_threshold)
            decision = self.old_equilibrium is None or \
                    norm(self.old_equilibrium-new_equilibrium) > self.compare_threshold
            self.old_equilibrium = new_equilibrium
            return decision
        