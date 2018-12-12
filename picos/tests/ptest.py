# coding: utf-8

#-------------------------------------------------------------------------------
# Copyright (C) 2018 Maximilian Stahlberg
#
# This file is part of PICOS.
#
# PICOS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PICOS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# This file implements a production (optimization) test framework.
#
# Note the following naming convention: A "test" is a single test method (e.g. a
# test of optimality of a primal solution), a "test case" is a class containing
# tests (e.g. an optimization problem formulation), and a "test set" is a module
# containing related test cases (e.g. linear optimization problems). Further, a
# "test suite" is any collection of tests.
#-------------------------------------------------------------------------------

import unittest
import os
import sys
import re
import importlib
import inspect
import cvxopt
import picos

PRODUCTION_TEST_PREFIX = "ptest_"

class ProductionTestError(Exception):
    """
    Base class for production testing specific exceptions.
    """
    pass

class ProductionTestCase(unittest.TestCase):
    """
    A test case base class for production (optimization) tests.

    Implementations would usually define `setUp`, `testPrimalSolution`, and
    `testDualSolution`.
    """
    class Options:
        """
        A class that contains options shared by all tests in a production test
        suite, such as their verbosity level or numerical precision.
        """
        def __init__(self, verbosity = 0, objPlaces = 6, varPlaces = 3,
                minSupport = picos.solvers.SUPPORT_LEVEL_SECONDARY,
                solveBoth = False):
            """
            :param int verbosity: Verbosity level, can be used inside the tests
                and gets passed to PICOS.
            :param int objPlaces: Number of places after the Point to consider
                when comparing objective values.
            :param int varPlaces: Number of places after the Point to consider
                when comparing variable values.
            :param int minSupport: Minimum support level. If the solver reports
                a smaller support level for a problem, the test is skipped.
            """
            self.verbosity  = verbosity
            self.objPlaces  = objPlaces
            self.varPlaces  = varPlaces
            self.minSupport = minSupport
            self.solveBoth  = solveBoth

    def __init__(self, test, solver, solverOptions = {}, testOptions = None):
        """
        Constructs a single test, parameterized by a solver and solver options.

        The parameterization can be automated with :func:`loadTests`.
        """
        super(ProductionTestCase, self).__init__(test)
        self.solver = solver
        self.options = solverOptions

        # Extract test options.
        if not testOptions:
            testOptions = self.Options()
        self.verbosity  = testOptions.verbosity
        self.objPlaces  = testOptions.objPlaces
        self.varPlaces  = testOptions.varPlaces
        self.minSupport = testOptions.minSupport
        self.solveBoth  = testOptions.solveBoth

        # Sanity check solver options.
        for forbiddenOption in ("solver", "noprimals", "noduals", "verbose"):
            if forbiddenOption in solverOptions:
                raise ProductionTestError("Forbidden testing option '{}'."
                    .format(forbiddenOption))

    @classmethod
    def loadTests(cls, tests = None,
        solvers = picos.solvers.available_solvers(), solverOptionSets = [{}],
        testOptions = None, listSelection = False):
        """
        Helper used by `makeProductionTestSuite` to generate a parameterized set
        of tests from the given test case.

        This can be seen as a factory method of :class:`ProductionTestCase`,
        except that it creates potentially multiple instances and merges them
        in a :class:`unittest.TestSuite`. If `listSelection` is True, then only
        the names of the test methods matching the `tests` filter are returned.

        :returns: A :class:`unittest.TestSuite` containing one copy of every
        test matching `tests` for every solver in `solvers` and for every set of
        solver options in `solverOptionSets`.
        """
        # Preprocess the tests filter.
        if tests:
            # Ignore case.
            tests = [test.lower() for test in tests]

            # Temporarily strip prefix, if given.
            tests = [test[4:] if test.startswith("test") else test for test in
                tests]

            # Allow some convenient short names for standard test names.
            if "primal" in tests:
                tests.append("primalsolution")
            if "dual" in tests:
                tests.append("dualsolution")

            # Add back the prefix.
            tests = ["test" + test for test in tests]

        # Select tests (in the form of method names).
        selectedTests = [test for test in
            unittest.TestLoader().getTestCaseNames(cls) if not tests or
            test.lower() in tests]

        if listSelection:
            return selectedTests

        # Assemble the test suite.
        testSuite = unittest.TestSuite()
        for solver in solvers:
            solverSuite = unittest.TestSuite()
            for solverOptions in solverOptionSets:
                for test in selectedTests:
                    solverSuite.addTest(
                        cls(test, solver, solverOptions.copy(), testOptions))
            testSuite.addTest(solverSuite)
        return testSuite

    def optionsToString(self, options):
        """
        Helper to transform a set of solver options into a string that is used
        in the test case description.
        """
        pairs = []
        for key, val in options.items():
            pairs.append(key + "=" + str(val))
        pairs.sort()
        return ", ".join(pairs)

    def getTestName(self):
        """
        A helper to transform test method names into test names.

        For example, getTestName("testPrimalSolution") == "Primal Solution".
        """
        name = self._testMethodName.split("test", 1)[1]
        return " ".join(re.sub('(?!^)([A-Z][a-z]+)', r' \1', name).split())

    def __str__(self):
        """
        This method is used by :package:`unittest` to describe the test case.

        Note that you can use a docstring with your test case to assign a more
        descriptive name to it.
        """
        if self.__doc__:
            # Select first non-empty line of docstring.
            problemName = self.__doc__.split("\n")
            problemName.remove("")
            problemName = problemName[0].strip()
        else:
            problemName = self.__class__.__name__
        solverName = self.solver.upper()
        testName = self.getTestName()
        if self.options:
            optionString = " with {}".format(self.optionsToString(self.options))
        else:
            optionString = ""
        description = "{} ({}): {} [{}{}]".format(problemName,
            self.__class__.__name__, testName, solverName, optionString)
        return description

    def solve(self, problem, primals = True, duals = True):
        """
        Produces a primal/dual solution pair for the given problem, using the
        selected solver, verbosity, and set of options.

        If `primals` or `duals` is set to `None`, then the solver is not told to
        not produce the respective solution, but its presence is not checked.
        """
        options = self.options.copy()

        options["solver"]    = self.solver
        options["verbose"]   = self.verbosity
        options["noprimals"] = primals is False
        options["noduals"]   = duals   is False
        options["allow_license_warnings"] = False

        problem.update_options(**options)

        solver = picos.solvers.get_solver(self.solver)
        supportLevel = solver.support_level(problem)

        if supportLevel < self.minSupport:
            self.skipTest("{} support."
            .format(picos.solvers.supportLevelString(supportLevel).title()))

        solution = problem.solve()

        if primals and not solution["primals"]:
            self.fail("No primal solution returned.")

        if duals and not solution["duals"]:
            self.fail("No dual solution returned")

        return solution

    def primalSolve(self, problem):
        """
        Produces a primal solution for the given problem, using the selected
        solver, verbosity, and set of options.
        """
        self.solve(problem, duals = None if self.solveBoth else False)

    def dualSolve(self, problem):
        """
        Produces a dual solution for the given problem, using the selected
        solver, verbosity, and set of options.
        """
        self.solve(problem, primals = None if self.solveBoth else False)

    def infeasibleSolve(self, problem):
        solution = self.solve(problem, primals = None, duals = None)

        # if solution["primals"]:
        #     self.fail("The problem is supposed to be infeasible but a primal "
        #         "solution was returned.")

        return solution

    def unboundedSolve(self, problem):
        solution = self.solve(problem, primals = None, duals = None)

        # if solution["duals"]:
        #     self.fail("The problem is supposed to be unbounded but a dual "
        #         "solution was returned.")

        return solution

    def assertAlmostEqual(
        self, first, second, places, msg = None, delta = None):
        """
        A wrapper around :func:`unittest.TestCase.assertAlmostEqual` that allows
        comparison of :class:`cvxopt.matrix` matrices.
        """
        if type(first) is int:
            first = float(first)
        if type(second) is int:
            second = float(second)

        if type(first) is float and type(second) is float:
            super(ProductionTestCase, self).assertAlmostEqual(
                first, second, places, msg, delta)
        else:
            firstMatrix = isinstance(first, cvxopt.matrix)
            secondMatrix = isinstance(second, cvxopt.matrix)

            if firstMatrix and secondMatrix:
                pass
            elif firstMatrix and not secondMatrix:
                second = cvxopt.matrix(second)
            elif not firstMatrix and secondMatrix:
                first = cvxopt.matrix(first)
            else:
                raise TypeError(
                    "Expecting one argument to be a CVXOPT matrix.")

            self.assertEqual(first.size, second.size)

            for i in range(len(first)):
                super(ProductionTestCase, self).assertAlmostEqual(
                    first[i], second[i], places, msg, delta)

    def readDual(self, constraint, variable):
        """
        Reads a dual value from a constraint into a variable.
        """
        dual = constraint.dual
        dualLen = 1 if isinstance(dual, float) else len(dual)
        self.assertEqual(dualLen, len(variable),
            msg="Dual of incompatible size.")
        variable.set_value(dual)

    def readDuals(self, constraint, *variables):
        """
        Reads dual values from a constraint into a number of variables.
        """
        dual = constraint.dual
        dualLen = 1 if isinstance(dual, float) else len(dual)
        self.assertEqual(dualLen, sum(len(var) for var in variables),
            msg="Dual of incompatible size.")
        varIndex = 0
        for variable in variables:
            nextVarIndex = varIndex + len(variable)
            variable.set_value(dual[varIndex : nextVarIndex])
            varIndex = nextVarIndex

    def expectObjective(self, problem, should):
        """
        Asserts that the objective value of a problem is as expected.
        """
        infeasibility = problem.check_current_value_feasibility()[1]
        infeasibility = infeasibility if infeasibility else 0
        self.assertAlmostEqual(infeasibility, 0, self.objPlaces,
            msg = "Infeasible.")
        self.assertAlmostEqual(problem.obj_value(), should, self.objPlaces,
            msg = "Objective value.")

    def expectVariable(self, variable, should):
        """
        Asserts that a variable has a certain value.

        Note that solvers might terminate as soon as the objective value gap is
        small while the distances of (dual) variables from their exact and
        unique solution can be much larger (but cancel out with respect to the
        objective function value). This is circumvented with some probability by
        using a lower numeric precision for variable checks by default.
        """
        if hasattr(should, "len"):
            self.assertEqual(len(variable), len(should),
                msg = "Variable of incompatible size.")
        self.assertAlmostEqual(variable.value, should, self.varPlaces,
            msg = "Variable {}.".format(variable.name))

def availableTestSets():
    """
    :returns: The names of all available production test sets.
    """
    testSets = []
    for testFileName in os.listdir(os.path.dirname(__file__)):
        if not testFileName.startswith(PRODUCTION_TEST_PREFIX) \
        or not testFileName.endswith(".py"):
            continue
        testFileBaseName = testFileName.rsplit(".py", 1)[0]
        testSetName = testFileBaseName.split(PRODUCTION_TEST_PREFIX, 1)[1]
        testSets.append(testSetName)
    return testSets

def makeTestSuite(
    testSetFilter = None, testCaseFilter = None, testFilter = None,
    solvers = picos.solvers.available_solvers(), solverOptionSets = [{}],
    testOptions = None, listSelection = False):
    """
    Helper to create a parameterized test suite containing tests from
    :class:`ProductionTestCase` test cases defined over multiple files.

    Defaults to collect all tests for all solvers and with default options.
    With the `listSelection` switch, all selected test sets, test cases, and
    tests are returned, which can be used to generate a list of all available
    filter options (by leaving all filters blank).

    :returns: A :class:`unittest.TestSuite` containing one copy of every
    production test matching `tests` in test cases matching `testCases` in test
    sets matching `testSets` for every solver in `solvers` and for every set of
    solver options in `solverOptionSets`.
    """
    testSuite = unittest.TestSuite()

    # Ignore case for all filters handled in this method.
    if testSetFilter:
        testSetFilter = [testSet.lower() for testSet in testSetFilter]
    if testCaseFilter:
        testCaseFilter = [testCase.lower() for testCase in testCaseFilter]

    # Select test sets (in the form of string names).
    selectedSets = [testSet for testSet in availableTestSets()
        if not testSetFilter or testSet.lower() in testSetFilter]

    if listSelection:
        setList  = set(selectedSets)
        caseList = set()
        testList = set()

    for testSet in selectedSets:
        # Load the test set (as a module).
        testSetModuleName = PRODUCTION_TEST_PREFIX + testSet
        testSetModule = importlib.import_module("." + testSetModuleName,
            package = __package__)

        # Select test cases (in the form of classes).
        selectedCases = [
            testCase for testCaseName, testCase
            in inspect.getmembers(testSetModule, inspect.isclass)
            if issubclass(testCase, ProductionTestCase)
            and testCase is not ProductionTestCase
            and (not testCaseFilter or testCaseName.lower() in testCaseFilter)]

        if listSelection:
            caseList.update([testCase.__name__ for testCase in selectedCases])

        for testCase in selectedCases:
            if listSelection:
                testList.update(testCase.loadTests(listSelection = True))
            else:
                # Add all tests matching `tests` from the test case.
                testSuite.addTest(testCase.loadTests(
                    testFilter, solvers, solverOptionSets, testOptions))

    if listSelection:
        return setList, caseList, testList

    return testSuite
