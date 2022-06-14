#!/usr/bin/python3
# -*- coding: utf-8 -*-

#=========================================================================
#Copyright (c) 2022

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

#=========================================================================

# REQUIREMENT LIST
#- Python 3.x with x >= 2

#=========================================================================

# Author: André Schrottenloher & Marc Stevens
# Date: June 2022
# Version: 1

#=========================================================================
"""
Defines some useful functions to compute the complexity and success probability
of the quantum algorithms studied in our paper, and to optimize them.
The "minimize" function from scipy.optimize is used for numerical optimizations.

In these scripts, algorithms / quantum circuits are simply represented as dictionaries containing
various information such as:
"space": space complexity (the number of qubits this circuit acts on)
"ancilla": number of ancilla qubits
"gates": number of gates
"prob": probability of success
"epsilon": relative interval on the probability of success
        i.e., we know that it belongs to an interval ((1-epsilon)prob ; (1+epsilon)prob)
"choice": size of the "choice" space, for the first substep of each step in our backtracking framework.

"""

from math import *
from scipy.optimize import minimize


def amplify_overcook(d, with_additional_gates=True):
    """
    Given algorithm A represented by dictionary d:
    {"space" : w, "ancilla" : a, "gates" : g, "prob" : pmin}
    where the success probability is only given as a lower bound, returns the
    dictionary corresponding to the "overcooked QAA" on A that reaches a success
    probability at least 1/2.
    Here w includes the flag qubit that A writes indicating if its result is good.
    """
    m = ceil(1.21 / sqrt(d["prob"]))
    return {
        "space":
        d["space"],  # no more space
        "ancillas":
        max(d["ancillas"], d["space"] - 1),
        # computing the inversion around zero requires some additional space
        "gates":
        ((2 * m + 2) * d["gates"] +
         ((m - 1) * (44 * d["space"] - 33) if with_additional_gates else 0)),
        # counting, in Clifford+T, the inversion around zero + phase flip
        "prob":
        1 / 2
    }


def print_algo(d):
    """
    Prints a dictionary of key : value representing an algorithm A, by just giving
    the log in base 2 of the values.
    """
    for i in d:
        print(i, ":", log(d[i], 2))


# Steps for the quantum DS-MITM attack on AES-256 of Bonnetain et al.
aes256_dsmitm_steps = [
    # first step: A1 (D1 is empty)
    {
        "A": {
            "choice": 2**80,
            "space": 0,
            "gates": 2**88,
            "prob": 1,
            "epsilon": 0
        },
        "D": {
            "space": 0,
            "gates": 0
        }
    },
    {
        "A": {
            "choice": 2**64,
            "space": 0,
            "gates": 32,
            "prob": 2**(-8),
            "epsilon": 0
        },
        "D": {
            "space": 0,
            "gates": 0
        }
    },
    {
        "A": {
            "choice": 2**32,
            "space": 0,
            "gates": 16,
            "prob": 2**(-8),
            "epsilon": 2**(-8)
        },
        "D": {
            "space": 0,
            "gates": 0
        }
    },
    {
        "A": {
            "choice": 2**32,
            "space": 0,
            "gates": 16,
            "prob": 2**(-8),
            "epsilon": 2**(-8)
        },
        "D": {
            "space": 0,
            "gates": 0
        }
    },
    {
        "A": {
            "choice": 2**32,
            "space": 0,
            "gates": 16,
            "prob": 2**(-8),
            "epsilon": 2**(-8)
        },
        "D": {
            "space": 0,
            "gates": 0
        }
    },
    {
        "A": {
            "choice": 2**32,
            "space": 0,
            "gates": 16,
            "prob": 2**(-8),
            "epsilon": 2**(-8)
        },
        "D": {
            "space": 0,
            "gates": 2**10,
        }
    },
    {
        "A": {
            "choice": 2**9,
            "space": 0,
            "gates": 2**5 * 40,
            "prob": 1,
            "epsilon": 0
        },
        "D": {
            "space": 0,
            "gates": 0
        }
    },
]

# Steps for the quantum Square attack on 6-round AES of Bonnetain et al.
aes_square_steps = [{
    "A": {
        "choice": 2**16,
        "space": 2**32,
        "gates": 2**36,
        "prob": 1,
        "epsilon": 0
    },
    "D": {
        "space": 0,
        "gates": 0
    }
}, {
    "A": {
        "choice": 2**8,
        "space": 2**24,
        "gates": 2**27,
        "prob": 1,
        "epsilon": 0
    },
    "D": {
        "space": 0,
        "gates": 0
    }
}, {
    "A": {
        "choice": 2**8,
        "space": 2**16,
        "gates": 2**19,
        "prob": 1,
        "epsilon": 0
    },
    "D": {
        "space": 0,
        "gates": 0
    }
}, {
    "A": {
        "choice": 2**8,
        "space": 2**8,
        "gates": 2**11,
        "prob": 1,
        "epsilon": 0
    },
    "D": {
        "space": 0,
        "gates": 0
    }
}]


def inversion_around_zero_gate_count(w):
    """
    Gate count of the "inversion around zero" operator on w qubits.
    """
    return 44 * w - 39


def algorithm_parameters(steps, relative_cost_gates=0.1):
    """
    Given a backtracking algorithm given as a list of steps: {"A" : A , "D" : D}
    where "A" is a dictionary representing an algorithm with parameters:
    "choice", "space", "gates", "prob" and "epsilon"
    and "D" is a dictionary representing an algorithm with parameters:
    "space", "gates"
    
    Returns the average complexity of an algorithm ensuring a success probability 1/2
    with our framework *using the formulas*.
    
    The parameter "relative_cost_gates" allows to specify if
    we want to count relatively less the framework gates. This happens in the AES
    examples because their "gates" actually count S-Boxes, which are more
    costly than gates. If None, then we will not count the other gates at all.
    """

    if relative_cost_gates is not None:
        inversion_around_zero_actual_count = (
            lambda x: relative_cost_gates * inversion_around_zero_gate_count(x)
        )
    else:
        inversion_around_zero_actual_count = lambda x: 0

    # read the steps and compute all the parameters, with the complexities and final success prob.
    ell = len(steps)
    size = [s["A"]["choice"] for s in steps]  # size of "choice" sets C_i
    gammap = [sqrt(s["A"]["prob"]) for s in steps]
    epsilon = [s["A"]["epsilon"] for s in steps]
    gamma = [1 / gammap[i] / sqrt(size[i])
             for i in range(ell)]  # values of gammai for each step

    # choose values of kprime
    kp = [
        0 if gammap[j] == 1 else floor(
            pi / (4 * asin(gammap[j] * sqrt(1 + epsilon[j]))))
        for j in range(ell)
    ]

    # choose values of k (depending on whether gammap = 1)
    k = [
        floor(0.5 * sqrt(size[j]) - 0.5) if gammap[j] == 1 else
        floor(0.5 * gammap[j] * sqrt((1 - epsilon[j]) * size[j]) - 1)
        for j in range(ell - 1)
    ]
    # choose the constant c
    optimized_c = False
    if not optimized_c:
        constant_c = 4 / (pi**2)
    else:
        constant_c = 1 / (1 + 1 / 9 * (pi**2 / 4 - 1))  # with

    k.append(floor(0.5 * sqrt(constant_c / ell) * sqrt(size[-1]) -
                   0.5))  # k_ell
    if optimized_c:
        for i in range(ell):
            if k[i] <= 0:
                raise ValueError(
                    "Incorrect parameters (number of iterations = 0)")
                # the better value of c given above works only if there is at least
                # one iteration at each step

    # compute success probability by the formula in the paper
    success_prob = exp(-1)
    for j in range(ell):
        if gammap[j] != 1:
            # there is a Grover search at this step
            error_term = (sin(
                (2 * kp[j] + 1) * asin(gammap[j] * sqrt(1 - epsilon[j]))))**2
            success_prob *= error_term
            print("Early-abort search error at step ", j + 1, error_term)
        if epsilon[j] != 0:
            error_term *= 1 / (1 + epsilon[j])
            print("Error at step ", j + 1, error_term)
            success_prob *= error_term
        print("Search error at step ", j + 1,
              (2 * k[j] + 1)**2 / gammap[j]**2 / size[j])
        success_prob *= (2 * k[j] + 1)**2 / gammap[j]**2 / size[j]
    print("Total success prob. (with log):")
    print(success_prob)
    print(log(success_prob, 2))

    print("Number of final calls (to reach 1/2):")
    final_iterates = 2 * ceil(1.21 / sqrt(success_prob)) + 2
    print(final_iterates)

    # then compute the complexities of B_ell, until B_1
    b_gates = [None for i in range(ell)]
    b_space = [None for i in range(ell)
               ]  # space complexity of bi (number of qubits spanned)

    # bell: (2kell+1) (G(Aell) + nell) + kell * I(nell)
    b_gates[-1] = (
        (2 * k[-1] + 1) * (steps[-1]["A"]["gates"] +
                           steps[-1]["A"]["space"] * relative_cost_gates) +
        k[-1] * inversion_around_zero_actual_count(steps[-1]["A"]["space"]))
    b_space[-1] = steps[-1]["A"]["space"]

    for i in range(ell - 2, -1, -1):
        # space complexity: all the qubits on which bi act (incl. new flag)
        b_space[i] = b_space[
            i + 1] + steps[i]["A"]["space"] + steps[i]["D"]["space"] + log(
                steps[i]["A"]["choice"], 2) + 1
        b_gates[i] = (
            (2 * k[i] + 1) *
            (b_gates[i + 1] + (2 * kp[i] + 1) *
             (steps[i]["A"]["gates"] +
              steps[i]["A"]["space"] * relative_cost_gates) +
             kp[i] * inversion_around_zero_actual_count(steps[i]["A"]["space"])
             + steps[i]["D"]["gates"]) +
            k[i] * inversion_around_zero_actual_count(b_space[i]))

    results_gates = [log(t, 2) for t in b_gates]
    print(k)
    print(kp)
    print(results_gates)

    final_algo = {
        "space": b_space[0],
        "ancillas": 0,
        "gates": b_gates[0],
        "prob": success_prob
    }
    overcook = amplify_overcook(final_algo)
    print_algo(overcook)


def algorithm_parameters_optimized(steps,
                                   relative_cost_gates=0.1,
                                   fix_prob=0.95,
                                   prob_factor=4):
    """
    Given a backtracking algorithm given as a list of steps: {"A" : A , "D" : D}
    where "A" is a dictionary representing an algorithm with parameters:
    "choice", "space", "gates", "prob" and "epsilon"
    and "D" is a dictionary representing an algorithm with parameters:
    "space", "gates"
    
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
    instead, where r is the parameter prob_factor (default : 4).
    """

    if relative_cost_gates is not None:
        inversion_around_zero_actual_count = (
            lambda x: relative_cost_gates * inversion_around_zero_gate_count(x)
        )
    else:
        inversion_around_zero_actual_count = lambda x: 0

    ell = len(steps)
    size = [s["A"]["choice"] for s in steps]  # size of "choice" sets C_i
    gammap = [sqrt(s["A"]["prob"]) for s in steps]
    epsilon = [s["A"]["epsilon"] for s in steps]
    gamma = [1 / gammap[i] / sqrt(size[i])
             for i in range(ell)]  # values of gammai for each step

    # choose values of kprime
    kp = [
        0 if gammap[j] == 1 else
        floor(pi / (4 * asin(gammap[j] * sqrt(1 + epsilon[j]))) - 0.5)
        for j in range(ell)
    ]
    # amplitude of success of the early-abort Grover searches
    early_ab_ampl = [
        1 if gammap[j] == 1 else sin(
            (2 * kp[j] + 1) * asin(gammap[j] * sqrt(1 - epsilon[j])))
        for j in range(ell)
    ]

    # now the values of k are the unknowns
    def prob(x):
        # probability of success, depending on a certain choice of k_i
        p = sin((2 * x[-1] + 1) * asin(gamma[-1] / (1 + epsilon[-1])))
        for i in range(ell - 1):
            ind = -2 - i
            # includes case epsilon = 0
            p = sin((2 * x[ind] + 1) * asin(
                p * gamma[ind] / sqrt(1 + epsilon[ind]) * early_ab_ampl[ind]))
        return p**2

    def complexity(x):
        # complexity, depending on a certain choice of k
        k = x
        b_gates = [None for i in range(ell)]
        b_space = [None for i in range(ell)
                   ]  # space complexity of bi (number of qubits spanned)

        # bell: (2kell+1) (G(Aell) + nell) + kell * I(nell)
        b_gates[-1] = (
            (2 * k[-1] + 1) * (steps[-1]["A"]["gates"] +
                               steps[-1]["A"]["space"] * relative_cost_gates) +
            k[-1] *
            inversion_around_zero_actual_count(steps[-1]["A"]["space"]))
        b_space[-1] = steps[-1]["A"]["space"]

        for i in range(ell - 2, -1, -1):
            # space complexity: all the qubits on which bi act (incl. new flag)
            b_space[i] = b_space[
                i + 1] + steps[i]["A"]["space"] + steps[i]["D"]["space"] + log(
                    steps[i]["A"]["choice"], 2) + 1
            b_gates[i] = (
                (2 * k[i] + 1) *
                (b_gates[i + 1] + (2 * kp[i] + 1) *
                 (steps[i]["A"]["gates"] +
                  steps[i]["A"]["space"] * relative_cost_gates) + kp[i] *
                 inversion_around_zero_actual_count(steps[i]["A"]["space"]) +
                 steps[i]["D"]["gates"]) +
                k[i] * inversion_around_zero_actual_count(b_space[i]))
        return b_gates[0]

    # bounds on the possible values of k
    k_bounds = [(0, pi / 4 / asin(gamma[j] / sqrt(1 - epsilon[j])) - 0.5)
                for j in range(ell)]
    start = [t[1] for t in k_bounds]
    if fix_prob is None:
        res = minimize(lambda x: complexity(x) / prob(x)**prob_factor,
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

    pass
    # EXAMPLES

    #==================================
    # computing the complexity of the AES-256 DS-MITM attack, for success probability
    # 1/2, using our formula (note that the compelxity is counted in S-Boxes,
    # but even setting relative_cost_gates = 1 does not change significantly the result anyway)

#    algorithm_parameters(aes256_dsmitm_steps, relative_cost_gates=0.1)
#    algorithm_parameters(aes_square_steps, relative_cost_gates=0.1)

#=================================
# computing the complexities (average) of the AES-256 DS-MITM and the AES
# square attacks, using a numerical optimization of the parameters.

# with fixed probability of success >= 95%: does not work for DS-MITM
#    algorithm_parameters_optimized(aes256_dsmitm_steps,
#                                   relative_cost_gates=0.1, fix_prob=0.95)

# without a fixed prob, only by optimizing a single function, but giving
# more weight to the prob of success (prob_factor = 4)
# (note that giving too much weight will result in a prob. of success close
# to 1 at all levels, and paying the pi/2 factor of each level).
#    algorithm_parameters_optimized(aes256_dsmitm_steps,
#                                   relative_cost_gates=0.1,
#                                   fix_prob=None,
#                                   prob_factor=4)

#    # here the fixed probability to 0.95 does work.
#    algorithm_parameters_optimized(aes_square_steps,
#                                   relative_cost_gates=0.1,
#                                   fix_prob=0.95)

#=================================
# Looking for the optimal configuration for a "search with independent tests"
# on a search space of size 2^256
#print(multi_test_find(256))
