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
# This file implements a production test set featuring QPs and QCQPs.
#-------------------------------------------------------------------------------

from .ptest import ProductionTestCase
import picos
import math

class USQP(ProductionTestCase):
    """
    Unconstrained Scalar QP

    (P) min. x² + x + 1
    """
    def setUp(self):
        # Primal problem.
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x")
        P.set_objective("min", x**2 + x + 1)

    def testPrimalSolution(self):
        self.primalSolve(self.P)
        self.expectObjective(self.P, 3.0/4.0)
        self.expectVariable(self.x, -1.0/2.0)

class ISQP(ProductionTestCase):
    """
    Inequality Scalar QP

    (P) min. x² + x + 1
        s.t. x ≥ 1
    """
    def setUp(self):
        # Primal problem.
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x")
        P.set_objective("min", x**2 + x + 1)
        P.add_constraint(x >= 1)

    def testPrimalSolution(self):
        self.primalSolve(self.P)
        self.expectObjective(self.P, 3.0)
        self.expectVariable(self.x, 1.0)

class UVQP(ProductionTestCase):
    """
    Unconstrained Vector QP

    (P) min. xᵀIx + 𝟙ᵀx + 1
    """
    def setUp(self):
        # Set the dimensionality.
        self.n = n = 4

        # Primal problem.
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x", n)
        P.set_objective("min", abs(x)**2 + (1|x) + 1)

    def testPrimalSolution(self):
        self.primalSolve(self.P)
        self.expectObjective(self.P, -self.n/4.0 + 1.0)
        self.expectVariable(self.x, [-1.0/2.0]*self.n)

class IVQP(ProductionTestCase):
    """
    Inequality Vector QP

    (P) min. xᵀIx + 𝟙ᵀx + 1
        s.t. x ≥ 1
    """
    def setUp(self):
        # Set the dimensionality.
        self.n = n = 4

        # Primal problem.
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x", n)
        P.set_objective("min", abs(x)**2 + (1|x) + 1)
        P.add_constraint(x >= 1)

    def testPrimalSolution(self):
        self.primalSolve(self.P)
        self.expectObjective(self.P, 2.0*self.n + 1.0)
        self.expectVariable(self.x, [1.0]*self.n)

class QCQP(ProductionTestCase):
    """
    Standard form QCQP

    The objective function's nonempty sublevel sets are hyperspheres centered
    at 𝟙 and the constraint region is the unit hypersphere centered at 𝟘, so the
    optimum solution in n dimensions is 𝟙/sqrt(n) (the point in the constraint
    region closest to the objective function's unconstrained minimum).

    (P) min. 0.5xᵀIx - 𝟙ᵀx - 0.5
        s.t. 0.5xᵀIx + 𝟘ᵀx - 0.5 ≤ 0
    """
    def setUp(self):
        # Set the dimensionality.
        self.n = n = 4

        # Define parameters.
        ones = picos.new_param("ones", [1.0]*n)
        I    = picos.diag(ones)

        # Primal problem.
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x", n)
        P.set_objective("min", 0.5*x.T*I*x - (1|x) - 0.5)
        P.add_constraint(0.5*x.T*I*x + (0|x) - 0.5 <= 0)

    def testPrimalSolution(self):
        self.primalSolve(self.P)
        self.expectObjective(self.P, -math.sqrt(self.n))
        self.expectVariable(self.x, [1.0/math.sqrt(self.n)]*self.n)

class NCQCQP(ProductionTestCase):
    """
    Nonconvex QCQP

    The objective function's nonempty sublevel sets are hyperspheres centered
    at 𝟙 and the constraint region is the unit hypersphere centered at 𝟘, so the
    optimum solution in n dimensions is -𝟙/sqrt(n) (the point in the constraint
    region furthest away from the objective function's unconstrained minimum).

    (P) max. 0.5xᵀIx - 𝟙ᵀx - 0.5
        s.t. 0.5xᵀIx + 𝟘ᵀx - 0.5 ≤ 0
    """
    def setUp(self):
        # Set the dimensionality.
        self.n = n = 4

        # Define parameters.
        ones = picos.new_param("ones", [1.0]*n)
        I    = picos.diag(ones)

        # Primal problem.
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x", n)
        P.set_objective("max", 0.5*x.T*I*x - (1|x) - 0.5)
        P.add_constraint(0.5*x.T*I*x + (0|x) - 0.5 <= 0)

    def testPrimalSolution(self):
        from ..tools import NonConvexError

        try:
            self.primalSolve(self.P)
        except NonConvexError:
            self.skipTest("Correctly detected as nonconvex.")

        self.expectObjective(self.P, math.sqrt(self.n))
        self.expectVariable(self.x, [-1.0/math.sqrt(self.n)]*self.n)
