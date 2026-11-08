import numpy as np
from .pymoo_compat import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.operators.crossover.binx import BinomialCrossover
from pymoo.algorithms.moo.nsga2 import NSGA2
from .problem import AuxiliarySingleLevelProblem, SubsetProblem
from .ga import MySampling, BinaryCrossover, MyMutation, IntMutation
from .rbf import RBF

def get_predictor(model, inputs, targets, device='cpu', **kwargs):
    assert model == 'rbf'
    p = RBF(**kwargs)
    p.fit(inputs, targets)
    return p

class Search:
    def _fit_predictor(self, archive, device='cpu'):
        inputs = np.array([self.search_space.encode_predictor(x[0]) for x in archive])
        targets = np.array([x[1] for x in archive])

        kwargs = {}
        if self.predictor == 'rbf':
            n_block = self.config['n_block']
            n_linear = self.config['n_linear']
            lb = np.zeros((n_linear, n_block))
            ub = np.ones((n_linear, n_block))
            
            for linear_idx, linear in enumerate(self.config['linear']):
                ub[linear_idx] = len(self.search_space.bits_range) - 1
            
            lb = np.delete(lb.flatten(), self.search_space.pass_linear_idx_list, axis=-1)
            ub = np.delete(ub.flatten(), self.search_space.pass_linear_idx_list, axis=-1)

            kwargs = {'lb': lb, 'ub': ub}

        quality_predictor = get_predictor(self.predictor, inputs, targets, device=device, **kwargs)

        return quality_predictor, quality_predictor.predict(inputs)

    def _next(self, archive, predictor, K):
        """ searching for next K candidate for high-fidelity evaluation (lower level) """

        # get non-dominated architectures from archive
        F = np.column_stack(([x[1] for x in archive], [x[2] for x in archive]))
        front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        # non-dominated arch bit-strings
        nd_X = np.array([self.search_space.encode(x[0]) for x in archive])[front]

        # initiate a multi-objective solver to optimize the problem
        method = NSGA2(pop_size=self.ga_pop_size, sampling=nd_X,  # initialize with current nd archs
            crossover=BinomialCrossover(prob=self.crossover_prob, n_offsprings=1),
            mutation=IntMutation(prob=self.mut_prob),
            eliminate_duplicates=True)
        
        # initialize the candidate finding optimization problem
        problem = AuxiliarySingleLevelProblem(self.search_space, predictor, self.config, self.group_size)
        
        # kick-off the search
        res = minimize(problem, method, termination=('n_gen', 20), save_history=True, verbose=True)
        
        # check for duplicates
        not_duplicate = np.logical_not(
            [x in [x[0] for x in archive] for x in [self.search_space.decode(x) for x in res.pop.get("X")]])
        print(f'not_duplicate : {sum(not_duplicate)}')

        pop = res.pop[not_duplicate]
        if sum(not_duplicate) >= K:
            indices = self._subset_selection(pop, F[front, 1], K, self.subset_pop_size)
            pop = pop[indices]

        candidates = []
        for x in pop.get("X"):
            candidates.append(self.search_space.decode(x))

        # decode integer bit-string to config and also return predicted top1_err
        return candidates, predictor.predict(self.search_space.decode_encode_predictor(pop.get("X")))

    def _subset_selection(self, pop, nd_F, K, pop_size):
        problem = SubsetProblem(pop.get("F")[:, 1], nd_F, K)
        algorithm = GA(
            pop_size=pop_size, sampling=MySampling(), crossover=BinaryCrossover(),
            mutation=MyMutation(), eliminate_duplicates=True)

        res = minimize(
            problem, algorithm, ('n_gen', 60), verbose=False)

        return res.X
