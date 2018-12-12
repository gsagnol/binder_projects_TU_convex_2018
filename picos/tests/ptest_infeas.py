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
# This file implements a test set featuring infeasingle and unbounded problems.
#-------------------------------------------------------------------------------

from .ptest import ProductionTestCase
import picos

class INFCLP(ProductionTestCase):
    """
    A simple LP with infeasible constraints.
    """
    def setUp(self):
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x")
        self.y = y = P.add_variable("y")
        P.set_objective("min", x)
        self.C  = P.add_constraint(x < y)
        self.Cx = P.add_constraint(x > 2)
        self.Cy = P.add_constraint(y < -2)

    def testSolution(self):
        self.infeasibleSolve(self.P)

class INFBLP(ProductionTestCase):
    """
    A simple LP with infeasible variable bounds.
    """
    def setUp(self):
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x", lower = 2)
        self.y = y = P.add_variable("y", upper = -2)
        P.set_objective("min", x)
        self.C = P.add_constraint(x < y)

    def testSolution(self):
        self.infeasibleSolve(self.P)

class UNBLP(ProductionTestCase):
    """
    A simple LP that is unbounded.
    """
    def setUp(self):
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x")
        self.C = P.add_constraint(x < 0)
        P.set_objective("min", x)

    def testSolution(self):
        self.unboundedSolve(self.P)
