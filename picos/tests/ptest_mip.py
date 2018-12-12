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
# This file implements a production test set featuring Mixed Integer Programs.
#-------------------------------------------------------------------------------

from .ptest import ProductionTestCase
import picos

class ILP(ProductionTestCase):
    """
    Integer LP

    (P) min. x + y + z
        s.t. x ≥ 1.5
             |y| ≤ 1
             |z| ≤ 2
             y + z ≥ 3
             x, y, z integer
    """
    def setUp(self):
        # Primal problem.
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x", vtype = "integer")
        self.y = y = P.add_variable("y", vtype = "integer")
        self.z = z = P.add_variable("z", vtype = "integer")
        P.set_objective("min", x + y + z)
        P.add_constraint(x >= 1.5)
        P.add_constraint(abs(y) <= 1)
        P.add_constraint(abs(z) <= 2)
        P.add_constraint(y + z >= 3)

    def testPrimalSolution(self):
        self.primalSolve(self.P)
        self.expectObjective(self.P, 5)
        self.expectVariable(self.x, 2)
        self.expectVariable(self.y, 1)
        self.expectVariable(self.z, 2)

class MILP(ProductionTestCase):
    """
    Mixed Integer LP

    (P) min. x + y + z
        s.t. x ≥ 1.5
             |y| ≤ 1
             |z| ≤ 2
             y + z ≥ 3
             y, z integer
    """
    def setUp(self):
        # Primal problem.
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x")
        self.y = y = P.add_variable("y", vtype = "integer")
        self.z = z = P.add_variable("z", vtype = "integer")
        P.set_objective("min", x + y + z)
        P.add_constraint(x >= 1.5)
        P.add_constraint(abs(y) <= 1)
        P.add_constraint(abs(z) <= 2)
        P.add_constraint(y + z >= 3)

    def testPrimalSolution(self):
        self.primalSolve(self.P)
        self.expectObjective(self.P, 4.5)
        self.expectVariable(self.x, 1.5)
        self.expectVariable(self.y, 1)
        self.expectVariable(self.z, 2)

class IQP(ProductionTestCase):
    """
    Integer QP

    (P) min. x² + y² + z²
        s.t. x ≥ 1.5
             |y| ≤ 1
             |z| ≤ 2
             y + z ≥ 3
             x, y, z integer
    """
    def setUp(self):
        # Primal problem.
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x", vtype = "integer")
        self.y = y = P.add_variable("y", vtype = "integer")
        self.z = z = P.add_variable("z", vtype = "integer")
        P.set_objective("min", x**2 + y**2 + z**2)
        P.add_constraint(x >= 1.5)
        P.add_constraint(abs(y) <= 1)
        P.add_constraint(abs(z) <= 2)
        P.add_constraint(y + z >= 3)

    def testPrimalSolution(self):
        self.primalSolve(self.P)
        self.expectObjective(self.P, 9)
        self.expectVariable(self.x, 2)
        self.expectVariable(self.y, 1)
        self.expectVariable(self.z, 2)

class MIQP(ProductionTestCase):
    """
    Mixed Integer QP

    (P) min. x² + y² + z²
        s.t. x ≥ 1.5
             |y| ≤ 1
             |z| ≤ 2
             y + z ≥ 3
             y, z integer
    """
    def setUp(self):
        # Primal problem.
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x")
        self.y = y = P.add_variable("y", vtype = "integer")
        self.z = z = P.add_variable("z", vtype = "integer")
        P.set_objective("min", x**2 + y**2 + z**2)
        P.add_constraint(x >= 1.5)
        P.add_constraint(abs(y) <= 1)
        P.add_constraint(abs(z) <= 2)
        P.add_constraint(y + z >= 3)

    def testPrimalSolution(self):
        self.primalSolve(self.P)
        self.expectObjective(self.P, 7.25)
        self.expectVariable(self.x, 1.5)
        self.expectVariable(self.y, 1)
        self.expectVariable(self.z, 2)

class ISOCP(ProductionTestCase):
    """
    Integer SOCP

    Also an Integer QCP.

    (P) max. ∑ᵢ i·xᵢ
        s.t. xᵢ² ≤ 1 ∀ i ∈ {1, …, n}
             ∑ᵢ xᵢ = k
             x nonnegative integer vector
    """
    def setUp(self):
        # Set the dimensionality and the parameter.
        self.n = n = 8
        self.k = k = 5

        # Primal problem.
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x", n, vtype = "integer", lower = 0)
        P.set_objective("max", sum([(i+1)*x[i] for i in range(n)]))
        P.add_list_of_constraints([x[i]**2 <= 1 for i in range(n)])
        P.add_constraint(1|x == k)

    @staticmethod
    def f(n):
        """The sum of the natural numbers up to ``n``."""
        return (n*(n+1)) // 2

    def testPrimalSolution(self):
        n = self.n
        k = self.k
        self.primalSolve(self.P)
        self.expectObjective(self.P, self.f(n) - self.f(n-k))
        self.expectVariable(self.x, [0]*(n-k) + [1]*k)

class IQCP(ProductionTestCase):
    """
    Integer QCP

    (P) max. ∑ᵢ i·xᵢ
        s.t. xᵢ² + xᵢ ≤ 2 ∀ i ∈ {1, …, n}
             ∑ᵢ xᵢ = k
             x nonnegative integer vector
    """
    def setUp(self):
        # Set the dimensionality and the parameter.
        self.n = n = 8
        self.k = k = 5

        # Primal problem.
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x", n, vtype = "integer", lower = 0)
        P.set_objective("max", sum([(i+1)*x[i] for i in range(n)]))
        P.add_list_of_constraints([x[i]**2 + x[i] <= 2 for i in range(n)])
        P.add_constraint(1|x == k)

    @staticmethod
    def f(n):
        """The sum of the natural numbers up to ``n``."""
        return (n*(n+1)) // 2

    def testPrimalSolution(self):
        n = self.n
        k = self.k
        self.primalSolve(self.P)
        self.expectObjective(self.P, self.f(n) - self.f(n-k))
        self.expectVariable(self.x, [0]*(n-k) + [1]*k)

class ISOCPQP(ProductionTestCase):
    """
    Integer SOCP with Quadratic Objective

    Also an Integer QCQP.

    (P) min. ∑ᵢ (i·xᵢ)²
        s.t. xᵢ² ≤ 1 ∀ i ∈ {1, …, n}
             ∑ᵢ xᵢ = k
             x nonnegative integer vector
    """
    def setUp(self):
        # Set the dimensionality and the parameter.
        self.n = n = 8
        self.k = k = 5

        # Primal problem.
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x", n, vtype = "integer", lower = 0)
        P.set_objective("min", sum([((i+1)*x[i])**2 for i in range(n)]))
        P.add_list_of_constraints([x[i]**2 <= 1 for i in range(n)])
        P.add_constraint(1|x == k)

    @staticmethod
    def f(n):
        """The sum of squares of the natural numbers up to ``n``."""
        return (n*(n+1)*(2*n+1)) // 6

    def testPrimalSolution(self):
        n = self.n
        k = self.k
        self.primalSolve(self.P)
        self.expectObjective(self.P, self.f(k))
        self.expectVariable(self.x, [1]*k + [0]*(n-k))

class IQCQP(ProductionTestCase):
    """
    Integer QCQP

    (P) min. ∑ᵢ (i·xᵢ)²
        s.t. xᵢ² + xᵢ ≤ 2 ∀ i ∈ {1, …, n}
             ∑ᵢ xᵢ = k
             x nonnegative integer vector
    """
    def setUp(self):
        # Set the dimensionality and the parameter.
        self.n = n = 8
        self.k = k = 5

        # Primal problem.
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x", n, vtype = "integer", lower = 0)
        P.set_objective("min", sum([((i+1)*x[i])**2 for i in range(n)]))
        P.add_list_of_constraints([x[i]**2 + x[i] <= 2 for i in range(n)])
        P.add_constraint(1|x == k)

    @staticmethod
    def f(n):
        """The sum of squares of the natural numbers up to ``n``."""
        return (n*(n+1)*(2*n+1)) // 6

    def testPrimalSolution(self):
        n = self.n
        k = self.k
        self.primalSolve(self.P)
        self.expectObjective(self.P, self.f(k))
        self.expectVariable(self.x, [1]*k + [0]*(n-k))

class NCISOCPQP(ProductionTestCase):
    """
    Integer SOCP with Nonconvex Quadratic Objective

    Also a Nonconvex Integer QCQP.

    (P) max. ∑ᵢ (i·xᵢ)²
        s.t. xᵢ² ≤ 1 ∀ i ∈ {1, …, n}
             ∑ᵢ xᵢ = k
             x nonnegative integer vector
    """
    def setUp(self):
        # Set the dimensionality and the parameter.
        self.n = n = 8
        self.k = k = 5

        # Primal problem.
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x", n, vtype = "integer", lower = 0)
        P.set_objective("max", sum([((i+1)*x[i])**2 for i in range(n)]))
        P.add_list_of_constraints([x[i]**2 <= 1 for i in range(n)])
        P.add_constraint(1|x == k)

    @staticmethod
    def f(n):
        """The sum of squares of the natural numbers up to ``n``."""
        return (n*(n+1)*(2*n+1)) // 6

    def testPrimalSolution(self):
        from ..tools import NonConvexError

        n = self.n
        k = self.k

        try:
            self.primalSolve(self.P)
        except NonConvexError:
            self.skipTest("Correctly detected as nonconvex.")

        self.expectObjective(self.P, self.f(n) - self.f(n-k))
        self.expectVariable(self.x, [0]*(n-k) + [1]*k)

class NCIQCQP(ProductionTestCase):
    """
    Nonconvex Integer QCQP

    (P) max. ∑ᵢ (i·xᵢ)²
        s.t. xᵢ² + xᵢ ≤ 2 ∀ i ∈ {1, …, n}
             ∑ᵢ xᵢ = k
             x nonnegative integer vector
    """
    def setUp(self):
        # Set the dimensionality and the parameter.
        self.n = n = 8
        self.k = k = 5

        # Primal problem.
        self.P = P = picos.Problem()
        self.x = x = P.add_variable("x", n, vtype = "integer", lower = 0)
        P.set_objective("max", sum([((i+1)*x[i])**2 for i in range(n)]))
        P.add_list_of_constraints([x[i]**2 + x[i] <= 2 for i in range(n)])
        P.add_constraint(1|x == k)

    @staticmethod
    def f(n):
        """The sum of squares of the natural numbers up to ``n``."""
        return (n*(n+1)*(2*n+1)) // 6

    def testPrimalSolution(self):
        from ..tools import NonConvexError

        n = self.n
        k = self.k

        try:
            self.primalSolve(self.P)
        except NonConvexError:
            self.skipTest("Correctly detected as nonconvex.")

        self.expectObjective(self.P, self.f(n) - self.f(n-k))
        self.expectVariable(self.x, [0]*(n-k) + [1]*k)

class ISDP(ProductionTestCase):
    """
    Integer SDP

    (P) max. <X, J>
        s.t. diag(X) = 𝟙
             X ≽ 0
             X integer
    """
    # NOTE: At the time of writing this test case, no solver supported by PICOS
    #       supports integer SDPs.
    def setUp(self):
        # Set the dimensionality.
        n = self.n = 4

        # Primal problem.
        self.P = P = picos.Problem()
        self.X = X = P.add_variable("X", (n, n), "integer")
        P.set_objective("max", X | 1)
        P.add_constraint(picos.diag_vect(X) == 1)
        # The following constraint is necessary because PICOS does not support
        # integer symmetric matrices via a variable type.
        P.add_constraint(X == X.T)
        P.add_constraint(X >> 0)

    def testPrimalSolution(self):
        self.primalSolve(self.P)
        self.expectObjective(self.P, self.n**2)
        self.expectVariable(self.X, cvxopt.matrix(1, (self.n, self.n)))
