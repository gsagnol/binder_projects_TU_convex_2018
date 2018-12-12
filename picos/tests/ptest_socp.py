# coding: utf-8

#-------------------------------------------------------------------------------
# Copyright (C) 2018 Guillaume Sagnol
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
# This file implements a production test set featuring SOCPs.
#-------------------------------------------------------------------------------

from .ptest import ProductionTestCase
import picos
import math

class SOCPLP(ProductionTestCase):
    """
    SOCP with Affine Constraint

    (P) max  x + y + z
        s.t. ‖[x; y; z]‖ ≤ 1   (CS)
             3x + 2y + z ≤ 3.3 (CL)

    (D) min  3.3μ + λ
        s.t. zₛ + [3μ; 2μ; μ]ᵀ = [1; 1; 1]ᵀ
             ‖zₛ‖ ≤ λ
             μ ≥ 0
    """
    def setUp(self):
        # Primal problem.
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x")
        self.y = y = P.add_variable("y")
        self.z = z = P.add_variable("z")
        P.set_objective("max", x + y + z)
        self.CS = P.add_constraint(abs(x // y // z) <= 1.0)
        self.CL = self.P.add_constraint(3.0*x + 2.0*y + z <= 3.3)

        # Dual problem.
        self.D = D = picos.Problem()
        self.lb = lb = D.add_variable("lambda")
        self.zs = zs = D.add_variable("zs", 3)
        self.mu = mu = D.add_variable("mu", lower = 0.0)
        D.set_objective("min", 3.3*mu + lb)
        D.add_constraint(zs + (3.0*mu) // (2.0*mu) // mu == 1.0)
        D.add_constraint(abs(zs) <= lb)

        self.expX = 99.0/140.0 - math.sqrt(1866)/210.0
        self.expY = 33.0/70.0 + math.sqrt(1866)/420.0
        self.expZ = 33.0/140.0 + math.sqrt(1866)/105.0
        self.expMu = 3.0/7.0 - (33.0*math.sqrt(3.0/622.0))/7.0
        self.expLb = 10.0*math.sqrt(6.0/311.0)
        self.expZs = [1.0-3.0*self.expMu, 1.0-2.0*self.expMu, 1.0-self.expMu]
        self.optimum = 3.3*self.expMu + self.expLb

    def testPrimalSolution(self):
        self.primalSolve(self.P)
        self.expectObjective(self.P, self.optimum)
        self.expectVariable(self.x, self.expX)
        self.expectVariable(self.y, self.expY)
        self.expectVariable(self.z, self.expZ)

    def testDualSolution(self):
        self.dualSolve(self.P)
        self.readDuals(self.CS, self.lb, self.zs)
        self.readDual(self.CL, self.mu)
        self.expectObjective(self.D, self.optimum)
        self.expectVariable(self.lb, self.expLb)
        self.expectVariable(self.zs, self.expZs)
        self.expectVariable(self.mu, self.expMu)

class RSOCP(ProductionTestCase):
    """
    Rotated SOCP

    (P) min  3x + 2y
        s.t. 1 ≤ xy (C)

    (D) max  z
        s.t. α = 3
             β = 2
             z² ≤ 4αβ
    """
    def setUp(self):
        # Primal problem.
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x")
        self.y = y = P.add_variable("y")
        P.set_objective("min", 3*x + 2*y)
        self.C = P.add_constraint(1 < x*y)

        # Dual problem.
        self.D = D = picos.Problem()
        self.a = a = D.add_variable("alpha")
        self.b = b = D.add_variable("beta")
        self.z = z = D.add_variable("z")
        D.set_objective("max", z)
        D.add_constraint(a == 3.0)
        D.add_constraint(b == 2.0)
        D.add_constraint(z**2 <= 4.0*a*b)

    def testPrimalSolution(self):
        self.primalSolve(self.P)
        self.expectObjective(self.P, 2*6**0.5)
        self.expectVariable(self.x, (2.0/3.0)**0.5)
        self.expectVariable(self.y, (3.0/2.0)**0.5)

    def testDualSolution(self):
        self.dualSolve(self.P)
        self.readDuals(self.C, self.a, self.b, self.z)
        self.expectObjective(self.D, 2*6**0.5)
        self.expectVariable(self.a, 3.0)
        self.expectVariable(self.b, 2.0)
        self.expectVariable(self.z, 2*6**0.5)
