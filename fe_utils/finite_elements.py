# Cause division to always mean floating point division.
from __future__ import division
import numpy as np
from .reference_elements import ReferenceInterval, ReferenceTriangle
np.seterr(invalid='ignore', divide='ignore')


def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """
    # 1d situation
    if cell is ReferenceInterval:
        # simply return the equispaced points between 0 and 1
        return np.linspace(0,1,num = degree + 1, endpoint = True ).reshape(-1,1)
    # 2d situation
    elif cell is ReferenceTriangle:
        points = []
        # iterate through j
        for j in range(degree+1):
            # initialisation
            i = 0
            # while loop through i. constraint applied
            while i+j <= degree:
                # add this point to the collection
                points.append([i/degree,j/degree])
                # update i
                i += 1 
        return np.array(points)
    
    else:
        raise ValueError("Unknown reference cell")

def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`

    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """

    # 1d situation
    if cell is ReferenceInterval:
        # initialisation
        V = np.zeros((points.shape[0],degree+1))

        if grad == False:
            # for loop
            for n in range(degree+1):
                V[:,n] = np.squeeze(points) ** n
            return V
        else:
            V = np.zeros((points.shape[0],degree+1,1))

            for n in range(degree+1):
                V[:,n,0] = np.squeeze(np.array([n*points ** max(n-1,0)]))
            return V
            

    # 2d situation
    elif cell is ReferenceTriangle:
        # iterate for each degree of expressions
        V = []

        if grad == False:
            for n in range(degree+1):
                # iterate every column
                for i in range(n+1):
                    V.append(points[:,0] ** (n-i) * points[:,1] ** i)
            
            return np.array(V).T

        else:
            # the first column
            V = [[[0,0] for uselessvalue in range(points.shape[0])]]

            # transform to list for comprehension
            points.tolist()
            for n in range(1,degree+1):
                
                # iterate every column
                
                for i in range(n+1):
                    V.append([[(n-i)* x ** max(n-i-1,0) * y ** i , 
                        i*x ** (n-i) * y ** max(i-1,0)] for [x,y] in points])
            
            # this code block could use the similar numpy method as
            #  1d but im too lazy. will come back if higher speed needed.

            # transpose a multidimensional array requires numpy
            return np.transpose(np.array(V),(1,0,2))
    
    else:
        raise ValueError("Unknown reference cell")


class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            the nodes of the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with entity `(d, i)`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0])
                                                for d in range(cell.dim+1)])

        # Replace this exception with some code which sets
        # self.basis_coefs
        # to an array of polynomial coefficients defining the basis functions.
        self.basis_coefs = np.linalg.inv(vandermonde_matrix(self.cell,
                                            self.degree,self.nodes))


        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """
        if grad == False:
            return vandermonde_matrix(self.cell,
                                    self.degree,points) @ self.basis_coefs
        else:
            
            return np.einsum('ijk,jl->ilk', vandermonde_matrix(self.cell,
                                    self.degree,points,True), self.basis_coefs)

    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """

        return fn(self.nodes)

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """

        nodes = lagrange_points(cell,degree)
        
        entity_nodes = {}

        # 1d case, using simple intuition
        if cell is ReferenceInterval:
            entity_nodes[0] = {0:[0],1:[degree]}
            entity_nodes[1] = {0:list(range(1,degree))}
        
        # 2d case, minimise calculation by assigning each dictionary
        elif cell is ReferenceTriangle:

            # for the top vertex use k + k + (k-1)... + 1 = k + k(k+1)/2
            top = degree*(degree+3)//2
            entity_nodes[0] = {0:[0],1:[degree],2:[top]}

            # vertical edge, again use k(k+1)/2
            edge = [top+1-k*(k+1)//2 for k in reversed(range(2,degree+1))]
            # slant
            slant = [top-k*(k+1)//2 for k in reversed(range(1,degree))]
            # all edges
            entity_nodes[1] = {0:slant,1:edge,2:list(range(1,degree))}

            # all remaining ones. generate set without lower edge and
            # top vertex. then remove the vertical edge and slant by 
            # comparison between sets.
            face = list(set(range(degree+1,top))-set(edge)-set(slant))
            # sets are unordered
            face.sort()
            entity_nodes[2] = {0:face}

        else:
            raise ValueError("Unknown reference cell")

        # Use lagrange_points to obtain the set of nodes.  Once you
        # have obtained nodes, the following line will call the
        # __init__ method on the FiniteElement class to set up the
        # basis coefficients.
        super(LagrangeElement, self).__init__(cell, degree, nodes, entity_nodes)
