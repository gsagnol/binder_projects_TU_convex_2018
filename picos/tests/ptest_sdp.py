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
# This file implements a production test set featuring SDPs.
#-------------------------------------------------------------------------------

from .ptest import ProductionTestCase
import picos
import cvxopt

class SDP(ProductionTestCase):
    """
    SDP with PSD Constraint on Variable

    (P) max. <X, J>
        s.t. diag(X) = 𝟙 (CT)
             X ≽ 0       (CX)

    (D) min. <𝟙, μ>
        s.t. J - Diag(μ) + Z = 0
             μ free
             Z ≽ 0
    """
    def setUp(self):
        # Set the dimensionality.
        n = self.n = 4

        # Primal problem.
        self.P = P = picos.Problem()
        self.X = X = P.add_variable("X", (n, n), "symmetric")
        P.set_objective("max", X | 1)
        self.CT = P.add_constraint(picos.diag_vect(X) == 1)
        self.CX = P.add_constraint(X >> 0)

        # Dual problem.
        self.D = D = picos.Problem()
        self.mu = mu = D.add_variable("mu", n)
        self.Z = Z = D.add_variable("Z", (n, n), "symmetric")
        D.set_objective("min", mu | 1)
        D.add_constraint(1 - picos.diag(mu) + Z == 0)
        D.add_constraint(Z >> 0)

    def testPrimalSolution(self):
        self.primalSolve(self.P)
        self.expectObjective(self.P, self.n**2)
        self.expectVariable(self.X, cvxopt.matrix(1, (self.n, self.n)))

    def testDualSolution(self):
        self.dualSolve(self.P)
        self.readDual(self.CT, self.mu)
        self.readDual(self.CX, self.Z)
        self.expectObjective(self.D, self.n**2)
