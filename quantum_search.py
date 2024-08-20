#!/usr/bin/python3
# -*- coding: utf-8 -*-

#=========================================================================
#Copyright (c) 2023

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#=========================================================================

#This project has been supported by ERC-ADG-ALGSTRONGCRYPTO (project 740972).
#It has been partially supported by the French National Research Agency through 
#the DeCrypt project under Contract ANR-18-CE39-0007, and through the France 
#2030 program under grant agreement No. ANR-22-PETQ-0008 PQ-TLS.

#=========================================================================

# REQUIREMENT LIST
#- Python 3.x with x >= 2
#- Scipy

#=========================================================================

# Author: André Schrottenloher & Marc Stevens
# Date: Jan 2024
# Version: 3

#=========================================================================
"""
Defines some useful functions to compute the complexity and success probability
of the quantum algorithms studied in our paper, and to optimize them.
The "minimize" function from scipy.optimize is used for numerical optimizations.

The subsequent steps Ai / Di of our generic framework are represented as dictionaries
with the following keys:

"choice": size of the choice space
"Aspace": size of the work register of A
"Agates": gate count of A
"probl": lower bound on the filtering probability of A (alpha_i'^2 in the paper)
"probu": upper bound on the filtering probability of A
"Dspace": size of the work register of D (if additional space required with respect to A)
"Dgates": gate count of D

"""

from math import *
from scipy.optimize import minimize


def gzero(n):
    """
    Number of gates to implement both the O_{0,n} operator and the O phase flip.
    The definition of this function depends on what you actually want to count
    as "gates". Here we are only counting Toffoli gates.
    """
    return 6 * n


def hadamard(n):
    """
    Number of gates to implement the Hadamard transform on n bits.
    The definition of this function depends on what you actually want to count
    as "gates". Here we are only counting Toffoli gates.
    """
    return 0


def amplify_overcook(d, with_additional_gates=True):
    """
    Given algorithm A represented by dictionary d:
    {"space" : w, "gates" : g, "prob" : pmin}
    where the success probability is only given as a lower bound, returns the
    dictionary corresponding to the "overcooked QAA" on A that reaches a success
    probability at least 1/2.
    Here w includes the flag qubit that A writes indicating if its result is good.
    """
    m = ceil(1.21 / sqrt(d["prob"]))
    return {
        "space":
        d["space"] + 1,
        "gates":
        ((2 * m + 2) * d["gates"] +
         ((m - 1) * (gzero(d["space"]) if with_additional_gates else 0))),
        "prob":
        1 / 2
    }


def print_algo(d):
    """
    Prints a dictionary of key : value representing an algorithm A, by just giving
    the log in base 2 of the values.
    """
    for i in d:
        print(i, ":", log(d[i], 2) if d[i] > 0 else "nan")


def check_list_steps(l):
    _keys = [
        "choice", "Aspace", "Agates", "probl", "probu", "Dspace", "Dgates"
    ]
    assert type(l) == list
    for d in l:
        assert type(d) == dict
        for k in _keys:
            assert k in d
        for k in d:
            assert k in _keys
        assert d["probl"] <= d["probu"]


# Steps for the quantum DS-MITM attack on AES-256 of Bonnetain et al.
aes256_dsmitm_steps = [
    # first step: A1 (D1 is empty)
    {
        "choice": 2**80,
        "Aspace": 0,
        "Agates": 2**88,
        "probl": 1,
        "probu": 1,
        "Dspace": 0,
        "Dgates": 0
    },
    {
        "choice": 2**64,
        "Aspace": 0,
        "Agates": 32,
        "probu": 2**(-8),
        "probl": 2**(-8),
        "Dspace": 0,
        "Dgates": 0
    },
    {
        "choice": 2**32,
        "Aspace": 0,
        "Agates": 16,
        "probu": 2**(-8) * (1 + 2**(-8)),
        "probl": 2**(-8) * (1 - 2**(-8)),
        "Dspace": 0,
        "Dgates": 0
    },
    {
        "choice": 2**32,
        "Aspace": 0,
        "Agates": 16,
        "probu": 2**(-8) * (1 + 2**(-8)),
        "probl": 2**(-8) * (1 - 2**(-8)),
        "Dspace": 0,
        "Dgates": 0
    },
    {
        "choice": 2**32,
        "Aspace": 0,
        "Agates": 16,
        "probu": 2**(-8) * (1 + 2**(-8)),
        "probl": 2**(-8) * (1 - 2**(-8)),
        "Dspace": 0,
        "Dgates": 0
    },
    {
        "choice": 2**32,
        "Aspace": 0,
        "Agates": 16,
        "probu": 2**(-8) * (1 + 2**(-8)),
        "probl": 2**(-8) * (1 - 2**(-8)),
        "Dspace": 0,
        "Dgates": 2**10,
    },
    {
        "choice": 2**9,
        "Aspace": 0,
        "Agates": 2**5 * 40,
        "probu": 1,
        "probl": 1,
        "Dspace": 0,
        "Dgates": 0
    }
]

# Steps for the quantum Square attack on 6-round AES of Bonnetain et al.
aes_square_steps = [{
    "choice": 2**16,
    "Aspace": 2**32,
    "Agates": 2**36,
    "probu": 1,
    "probl": 1,
    "Dspace": 0,
    "Dgates": 0
}, {
    "choice": 2**8,
    "Aspace": 2**24,
    "Agates": 2**27,
    "probu": 1,
    "probl": 1,
    "Dspace": 0,
    "Dgates": 0
}, {
    "choice": 2**8,
    "Aspace": 2**16,
    "Agates": 2**19,
    "probu": 1,
    "probl": 1,
    "Dspace": 0,
    "Dgates": 0
}, {
    "choice": 2**8,
    "Aspace": 2**8,
    "Agates": 2**11,
    "probu": 1,
    "probl": 1,
    "Dspace": 0,
    "Dgates": 0
}]

# Steps fror the ID attack of David et al.
aes_id_steps = [{
    "choice": 2**32,
    "Aspace": 0,
    "Agates": 8 * 2**78.5,
    "probu": 1,
    "probl": 1,
    "Dspace": 0,
    "Dgates": 0
}, {
    "choice": 2**32,
    "Aspace": 0,
    "Agates": 8 * 2**63.5,
    "probu": 1,
    "probl": 1,
    "Dspace": 0,
    "Dgates": 0
}, {
    "choice": 2**32,
    "Aspace": 0,
    "Agates": 8 * 2**47.5,
    "probu": 1,
    "probl": 1,
    "Dspace": 0,
    "Dgates": 0
}, {
    "choice": 2**32,
    "Aspace": 0,
    "Agates": 8 * 2**23.5,
    "probu": 1,
    "probl": 1,
    "Dspace": 0,
    "Dgates": 0
}]


def algorithm_parameters(steps, relative_cost_gates=0.1):
    """
    Given a backtracking algorithm given as a list of steps, where each
    step is a dictionary with the keys:
    "choice", "Aspace", "Agates", "probl", "probu", "Dspace", "Dgates"
    
    Returns the average complexity of an algorithm ensuring a success probability 1/2
    with our framework *using the formulas*.
    
    The parameter "relative_cost_gates" allows to specify if
    we want to count relatively less the framework gates. This happens in the AES
    examples because their "gates" actually count S-Boxes, which are more
    costly than gates. If None, then we will not count the other gates at all.
    """
    check_list_steps(steps)
    if relative_cost_gates is not None:
        gzero_actual = (lambda x: relative_cost_gates * gzero(x))
        hadamard_actual = (lambda x: relative_cost_gates * hadamard(x))
    else:
        gzero_actual = lambda x: 0
        hadamard_actual = lambda x: 0

    # read the steps and compute all the parameters, with the complexities and final success prob.
    ell = len(steps)
    size = [s["choice"] for s in steps]  # size of "choice" sets C_i

    # parameters l_i' for the bound on alpha_i'
    lp = [sqrt(s["probl"]) for s in steps]
    # parameters u_i'
    up = [sqrt(s["probu"]) for s in steps]
    l = [1 / up[i] / sqrt(size[i]) for i in range(ell)]
    u = [1 / lp[i] / sqrt(size[i]) for i in range(ell)]

    # choose with the formulas of Lemma 11 (Sec 5.4)
    kp = [floor(pi / (4 * asin(up[i])) - 0.5) for i in range(ell)]

    # choose values of k
    k = [floor(0.5 / u[i] - 0.5) for i in range(ell - 1)]
    # choose kell
    k.append(floor(0.5 / sqrt(2 * ell) / u[ell - 1] - 0.5))

    # compute success probability by the formula in the paper
    success_prob = 0.5
    for j in range(ell):
        # there is a Grover search at this step
        error_term = (sin((2 * kp[j] + 1) * asin(lp[j])))**2
        success_prob *= error_term
        print("Early-abort search error at step ", j + 1, error_term)
        search_error_term = (2 * k[j] + 1)**2 * l[j]**2
        print("Search error at step ", j + 1, search_error_term)
        success_prob *= search_error_term
    print("Total success prob. (with log):")
    print(success_prob)
    print(log(success_prob, 2))

    print("Number of final calls (to reach 1/2):")
    final_iterates = 2 * ceil(1.21 / sqrt(success_prob)) + 2
    print(final_iterates)

    b_gates = [None for i in range(ell + 1)]
    # gate complexity of B_i
    b_space = [None for i in range(ell + 1)]
    # space complexity of B_i (number of qubits spanned)

    # recall that B_{ell+1} does nothing by definition
    b_gates[-1] = 0
    b_space[-1] = 0
    current_choice_space = 0

    for i in range(ell - 1, -1, -1):
        current_choice_space += log(steps[i]["choice"], 2)
        # space complexity: all the qubits on which bi act (incl. new flag)
        b_space[i] = (b_space[i + 1] + steps[i]["Aspace"] +
                      steps[i]["Dspace"] + log(steps[i]["choice"], 2) + 1)
        b_gates[i] = (
            (2 * k[i] + 1) *
            (b_gates[i + 1] + steps[i]["Dgates"] + (2 * kp[i] + 1) *
             (steps[i]["Agates"] + hadamard_actual(steps[i]["Aspace"])) +
             kp[i] * gzero_actual(steps[i]["Aspace"])) +
            k[i] * gzero_actual(current_choice_space))

    results_gates = [log(t, 2) for t in b_gates[:-1]]
    print(k)
    print(kp)
    print(results_gates)

    final_algo = {
        "space": b_space[0],
        "gates": b_gates[0],
        "prob": success_prob
    }
    overcook = amplify_overcook(final_algo)
    print_algo(overcook)


def algorithm_sqrt(steps):
    """
    Computes the complexity if quantum search was an exact procedure, using the 
    square-root of the expected numbers of iterates. We take the upper bounds
    of success probabilities.
    """
    # read the steps and compute all the parameters, with the complexities
    # and final success prob.
    ell = len(steps)
    size = [s["choice"] for s in steps]  # size of "choice" sets C_i
    alphap = [sqrt(s["probu"]) for s in steps]
    alpha = [1 / alphap[i] / sqrt(size[i]) for i in range(ell)]

    # choose values of kprime
    kp = [1 / alphap[j] for j in range(ell)]
    k = [1 / alpha[j] for j in range(ell)]
    b_gates = [None for i in range(ell + 1)]

    b_gates[-1] = 0
    for i in range(ell - 1, -1, -1):
        b_gates[i] = k[i] * (b_gates[i + 1] + steps[i]["Dgates"] + kp[i] *
                             (steps[i]["Agates"]))

    results_gates = [log(t, 2) for t in b_gates[:-1]]
    print(k)
    print(kp)
    print(results_gates)

    final_algo = {"space": 0, "gates": b_gates[0], "prob": 1}
    print_algo(final_algo)


def algorithm_parameters_optimized(steps,
                                   relative_cost_gates=0.1,
                                   fix_prob=0.95,
                                   prob_exponent=4):
    """
    Given a backtracking algorithm given as a list of steps (as above),
    
    Returns the average complexity of an algorithm to obtain a success. This
    complexity is usually better than the result of 'algorithm_parameters', because
    it is obtained by a numerical optimization over the iteration numbers.
    
    The parameter relative_cost_gates has the same role as in 'algorithm_parameters'.
    
    If 'fix_prob' is not None, then we will search for a minimization under
    the constraint that the probability of success must be at least 'fix_prob'.
    Unfortunately, this does not always work.
    
    Otherwise, we will search for a minimization of the ratio time / prob. of success.
    This will usually give much less, e.g. 80% success.
    In order to increase it (albeit a bit artificially), we optimize time / (prob**r)
    instead, where r is the parameter prob_exponent (default : 4).
    """
    check_list_steps(steps)
    if relative_cost_gates is not None:
        gzero_actual = (lambda x: relative_cost_gates * gzero(x))
        hadamard_actual = (lambda x: relative_cost_gates * hadamard(x))
    else:
        gzero_actual = lambda x: 0
        hadamard_actual = lambda x: 0

    ell = len(steps)
    size = [s["choice"] for s in steps]  # size of "choice" sets C_i
    # parameters l_i' for the bound on alpha_i'
    lp = [sqrt(s["probl"]) for s in steps]
    # parameters u_i'
    up = [sqrt(s["probu"]) for s in steps]
    l = [1 / up[i] / sqrt(size[i]) for i in range(ell)]
    u = [1 / lp[i] / sqrt(size[i]) for i in range(ell)]

    # choose with the formulas of Lemma 11 (Sec 5.4)
    kp = [floor(pi / (4 * asin(up[i])) - 0.5) for i in range(ell)]

    # amplitude of success of the early-abort layers
    early_ab_ampl = [sin((2 * kp[j] + 1) * asin(lp[j])) for j in range(ell)]

    # now the values of k are the unknowns
    def prob(x):
        # probability of success, depending on a certain choice of k_i
        p = sin((2 * x[-1] + 1) * asin(l[-1]))
        for i in range(ell - 1):
            ind = -2 - i
            # includes case epsilon = 0
            p = sin((2 * x[ind] + 1) * asin(p * l[ind] * early_ab_ampl[ind]))
        return p**2

    def complexity(x):
        # complexity, depending on a certain choice of k
        k = x

        b_gates = [None for i in range(ell + 1)]
        # gate complexity of B_i
        b_space = [None for i in range(ell + 1)]
        # space complexity of B_i (number of qubits spanned)

        # recall that B_{ell+1} does nothing by definition
        b_gates[-1] = 0
        b_space[-1] = 0
        current_choice_space = 0

        for i in range(ell - 1, -1, -1):
            current_choice_space += log(steps[i]["choice"], 2)
            # space complexity: all the qubits on which bi act (incl. new flag)
            b_space[i] = (b_space[i + 1] + steps[i]["Aspace"] +
                          steps[i]["Dspace"] + log(steps[i]["choice"], 2) + 1)
            b_gates[i] = (
                (2 * k[i] + 1) *
                (b_gates[i + 1] + steps[i]["Dgates"] + (2 * kp[i] + 1) *
                 (steps[i]["Agates"] + hadamard_actual(steps[i]["Aspace"])) +
                 kp[i] * gzero_actual(steps[i]["Aspace"])) +
                k[i] * gzero_actual(current_choice_space))
        return b_gates[0]

    # bounds on the possible values of k
    k_bounds = [(0, pi / 4 / asin(u[j]) - 0.5) for j in range(ell)]
    start = [t[1] for t in k_bounds]
    if fix_prob is None:
        res = minimize(lambda x: complexity(x) / prob(x)**prob_exponent,
                       start,
                       bounds=k_bounds,
                       options={
                           'maxiter': 50000,
                           'disp': True
                       })
    else:
        # use another optimization method here, and a nonlinar constraint
        # to set prob > fix_prob
        constraints = [{'type': 'ineq', 'fun': lambda x: prob(x) - fix_prob}]
        for j in range(ell):
            # bounds on kj
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: k_bounds[j][1] - x[j]
            })
            constraints.append({'type': 'ineq', 'fun': lambda x: x[j]})
        res = minimize(lambda x: complexity(x) / prob(x),
                       start,
                       constraints=constraints,
                       method="cobyla",
                       options={
                           'maxiter': 10000,
                           'disp': True
                       })

    if not res.success:
        raise Exception("Optimization failed (sorry)")
    resx = [floor(t) for t in res.x]
    print("K values resulting from optimization:")
    print(resx)
    for i in range(ell):
        if resx[i] <= 0 or resx[i] > k_bounds[i][1]:
            raise Exception("Result does not satisfy the bounds on ki")

    # recompute the complexity, and print the result
    print("Success probability: ", prob(resx))
    print("Complexity (log2): ", log(complexity(resx), 2))
    print("Comp / succ. prob (log2): ", log(complexity(resx) / prob(resx), 2))


def multi_test_opti(m1, m2, n):
    """
    Optimizes the number of iterates for a 'search with independent tests', where
    we have already parameterized the number of tests for each individual search.
    Here we are taking three steps, and m1, m2 and m3 = m - m1 - m2 tests for each.
    We assume that n - m1 - m2 >= 12
    """
    if n - m1 - m2 < 12:
        raise ValueError("Constraint not satisfied")
    m = n + 3  # nb of tests
    epsilon1 = 2**(-4)
    epsilon2 = 2**(-4)

    # lower and upper bounds on the success probabilities of the substeps
    alpha_down = [
        2**(-m1) * (1 - epsilon1), 2**(-m2) * (1 - epsilon2) / (1 + epsilon1),
        2**(-n + m1 + m2) / (1 + epsilon2)
    ]
    alpha_up = [
        2**(-m1) * (1 + epsilon1), 2**(-m2) * (1 + epsilon2) / (1 - epsilon1),
        2**(-n + m1 + m2) / (1 - epsilon2)
    ]
    alpha_down = [sqrt(t) for t in alpha_down]
    alpha_up = [sqrt(t) for t in alpha_up]

    # success probability of the whole algorithm for a given choice of k
    def prob(x):
        k1, k2, k3 = x[0], x[1], x[2]
        p1 = sin((2 * k1 + 1) * asin(alpha_down[0]))
        p2 = sin((2 * k2 + 1) * asin(alpha_down[1] * p1))
        p3 = sin((2 * k3 + 1) * asin(alpha_down[2] * p2))
        return p3**2

    # complexity for a given choice of k (counted in number of tests)
    def complexity(x):
        k1, k2, k3 = x[0], x[1], x[2]
        tmp = m1 * (2 * k1 + 1) * (2 * k2 + 1) * (2 * k3 + 1) + m2 * (
            2 * k2 + 1) * (2 * k3 + 1) + (m - m1 - m2) * (2 * k3 + 1)
        return tmp / prob(x)

    # bounds on k for optimization
    k_bounds = [(0, pi / 4 / asin(alpha_up[i]) - 0.5) for i in range(3)]
    bounds = k_bounds
    start = [t[1] for t in k_bounds]

    res = minimize(complexity,
                   start,
                   bounds=bounds,
                   options={
                       'maxiter': 30000,
                       'disp': False
                   })
    if not res.success:
        raise Exception("Optimization failed (sorry)")
    resx = [floor(t) for t in res.x]
    return complexity(resx), resx


def multi_test_find(n):
    """
    Optimizes a 'search with independent tests' for a given n, so a search
    space of size 2^n, and n+3 independent tests to run (which each succeed
    with probability 1/2). We cut the tests in three groups and perform
    a nested QAA. For a given choice, we use 'multi_test_opti' to optimize
    the time complexity. We then look for the best choice.
    """
    best = 10**n
    val = 0, 0
    for m1 in range(1, n // 4):
        for m2 in range(m1, n // 2):
            if n - m1 - m2 >= 12:
                try:
                    tmp = multi_test_opti(m1, m2, n)[0]
                    if tmp < best:
                        val = m1, m2
                        best = tmp
                except Exception as e:
                    pass  # possibly a failure of the optimization. This may
                    # happen in (uninteresting) corner cases.
    return val, log(best, 2)


if __name__ == "__main__":

    #==================================
    # computing the complexity of the AES-256 DS-MITM attack, for success probability
    # 1/2, using our formula (note that the compelxity is counted in S-Boxes,
    # but even setting relative_cost_gates = 1 does not change significantly the result anyway)

    algorithm_parameters(aes256_dsmitm_steps, relative_cost_gates=0.1)
    print("\n----------------------------------\n")
    algorithm_parameters(aes_square_steps, relative_cost_gates=0.1)
    print("\n================================\n")
    algorithm_parameters(aes_id_steps, relative_cost_gates=0.1)
    print("\n================================\n")

    algorithm_sqrt(aes256_dsmitm_steps)
    print("\n----------------------------------\n")
    algorithm_sqrt(aes_square_steps)
    print("\n================================\n")
    algorithm_sqrt(aes_id_steps)
    print("\n================================\n")

    #=================================
    # computing the complexities (average) of the AES-256 DS-MITM and the AES
    # square attacks, using a numerical optimization of the parameters.

    # We optimize (complexity / proba of success*r) for some power r (prob_exponent).
    # (note that giving too much weight will result in a prob. of success close
    # to 1 at all levels, and paying the pi/2 factor of each level).
    algorithm_parameters_optimized(aes256_dsmitm_steps,
                                   relative_cost_gates=0.1,
                                   fix_prob=None,
                                   prob_exponent=3.5)
    print("\n----------------------------------\n")
    algorithm_parameters_optimized(aes_square_steps,
                                   relative_cost_gates=0.1,
                                   fix_prob=None,
                                   prob_exponent=3)

    print("\n----------------------------------\n")
    algorithm_parameters_optimized(aes_id_steps,
                                   relative_cost_gates=0.1,
                                   fix_prob=None,
                                   prob_exponent=3)

    #=================================
    # Looking for the optimal configuration for a "search with independent tests"
    # on a search space of size 2^256
    #print(multi_test_find(256))
