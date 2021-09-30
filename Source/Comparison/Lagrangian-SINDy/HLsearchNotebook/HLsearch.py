import itertools
import operator
import numpy as np
from collections import OrderedDict

from sympy import symbols, var, diff,simplify, collect,sympify,solve
from sympy.utilities.lambdify import lambdify, implemented_function

from operator import add,sub,mul

def Reverse(tuples): 
    """
    get reversed tuple (1,2,3) -> (3,2,1)
    """
    reversed_tup = () 
    for k in reversed(tuples): 
        reversed_tup = reversed_tup + (k,) 
    return reversed_tup

def buildFunctionExpressions(P, d, data_description = None, use_sine=False):
    """
    generate a base of functions which are polynomials and trigonometric functions (sin and cos)

    params:
    P: max power in polynomial
    d: number of variables
    data_description: variables' name
    use_sine: True for using trigonometric functions

    return:
    a list of functions
    """
    if use_sine:
        sin_description=[]
        cos_description=[]
        for name in data_description[::2]:
            sin_description = sin_description + ['sin({})'.format(name)]
            cos_description = cos_description + ['cos({})'.format(name)]
        data_description = data_description + sin_description + cos_description
        d = len(data_description)

    # Create a list of all polynomials in d variables up to degree P
    rhs_functions = OrderedDict()
    f = lambda x, y : np.prod(np.power(list(x), list(y)))
    powers = []            
    for p in range(1,P+1):
            size = d + p - 1
            for indices in itertools.combinations(range(size), d-1):
                starts = [0] + [index+1 for index in indices]
                stops = indices + (size,)
                powers.append(tuple(map(operator.sub, stops, starts)))
    for power in powers:
        power=Reverse(power)
        rhs_functions[power] = [lambda x, y = power: f(x,y), power]
    
    descr = []
    for k in rhs_functions.keys():
        if data_description is None: descr.append(str(rhs_functions[k][1]))
        else:
            function_description = ''
            written=False
            for j in range(d):
                if rhs_functions[k][1][j] != 0:
                    if written:
                            function_description = function_description + '*'
                    if rhs_functions[k][1][j] == 1:
                        function_description = function_description + data_description[j]
                        written=True
                    else:
                        function_description = function_description + data_description[j] + '**' + str(rhs_functions[k][1][j])
                        written=True
            descr.append(function_description)
    return descr

def buildTimeSerieFromFunction(data,function_description, data_description):
    column = []
    f=lambdify(data_description,function_description,'numpy')
    for i in range(data.shape[0]):
        column.append(f(*data[i,:]))
    return(np.array(column))

def buildTimeSerieMatrixFromFunctions(data,function_description, data_description):
    """
    params:
        data: values of variables. In [x,x_dot] format. Each row presents state at one time
        function_description: symbollic expression of functions
        data_description: variables' names
    return:
        time-serie of calculated functions
    """
    Matrix = []
    for func in function_description:
        column = []
        f=lambdify(data_description,func,'numpy')
        for i in range(data.shape[0]):
            column.append(f(*data[i,:]))
        Matrix.append(column)
    return np.array(Matrix).T

def gradient(func_description, data_description):
    """Symbolic grad""" 
    grad = []
    for x in data_description:
        dfdx_expr = diff(func_description,x)
        grad.append(dfdx_expr)
    return grad

def buildTimeDerivativeSerieFromFunction(data, data_t, function_description, data_description):
    grad = gradient(function_description,data_description)
    grad_funcs= [lambdify(data_description,grad_func,'numpy') for grad_func in grad]
    # compute Gamma column corresponding to each function
    column = []
    for j in range(data.shape[0]):
        result = 0
        for i in range(data.shape[1]):
            result = result + grad_funcs[i](*data[j,:])*data_t[j,i]
        column.append(result)
    return(np.array(column))

def buildTimeDerivativeMatrixFromFunctions(data, data_t, function_description, data_description):
    """
    compute df/dt by taking partial derivative over all variables and multiplying their derivative and taking sum

    params:
        data: values of variables. In [x,x_dot] format. Each row presents state at one time
        data_t: values of time derivatives. In [x_dot,x_2dot] format. Each row presents state at one time
        function_description: symbollic expression of functions
        data_description: variables' names
    return:
        time-serie of calculated derivative functions
    """
    Gamma = []
    for func_descr in function_description:
        # find grad and lambdify into functions
        grad = gradient(func_descr,data_description)
        grad_funcs= [lambdify(data_description,grad_func,'numpy') for grad_func in grad]
        # compute Gamma column corresponding to each function
        column = []
        for j in range(data.shape[0]):
            result = 0
            for i in range(data.shape[1]):
                result = result + grad_funcs[i](*data[j,:])*data_t[j,i]
            column.append(result)
        Gamma.append(column)
    return np.array(Gamma).T

def generateExpression(coefficient_vector,function_description,threshold = 1e-8):
    ret = ''
    for coef,func in zip(coefficient_vector,function_description):
        if abs(coef)>threshold:
            if ret!='' and coef>=0:
                ret = ret + '+'
            ret = ret + str(coef) + '*' + str(func)
    if ret != '' : ret = sympify(ret) 
    return ret


def generateSimplifiedExpression(coefficient_vector, function_description, threshold = 1e-8):
    strToSimlify = generateExpression(coefficient_vector,function_description,threshold)
    temp = simplify(strToSimlify)
    c_tup = ()
    remaining_functions = ()
    for f in function_description:
        collected = collect(temp,f)
        if abs(collected.coeff(f)) > threshold:
            c_tup = c_tup + (collected.coeff(f),)
            remaining_functions = remaining_functions + (sympify(f),)
        else:
            c_tup = c_tup + (0.0,)
    simplifiedStr=generateExpression(c_tup,function_description,threshold)
    return simplifiedStr,remaining_functions

def findLagrangianFromHamiltonian(Hamiltonian,terms,data_description_sym,threshold=1e-8):
    qdotderiv = []
    for qdot in data_description_sym[1::2]:
        derivs = ()
        for term in terms:
            derivs = derivs + (diff(term,qdot),)
        qdotderiv_row = []
        for deriv in derivs:
            qdotderiv_row.append( qdot*deriv )
        qdotderiv.append(qdotderiv_row)

    # print('qdotderiv', qdotderiv)
    sumQtd=None
    for qtd in qdotderiv:
        if sumQtd is None:
            sumQtd=qtd
        else:
            sumQtd = list( map(add, sumQtd, qtd) )

    alpha = list( map(sub, sumQtd, terms) )
    xi_L = var('xi:{}'.format(len(alpha)))
    beta = list( map(mul, xi_L, alpha) )
    L_with_coef = sum(beta)
    equations = []
    for f in terms:
        collected = collect(L_with_coef,f)
        collectedH= collect(Hamiltonian,f)
        equations.append(collected.coeff(f)-collectedH.coeff(f))
    solution = solve(equations,xi_L,dict=True)
    if solution == []: return None
    elif len(solution[0])!= len(xi_L): return None
    else:
        reordered_solution = OrderedDict([ (k,solution[0][k]) for k in xi_L ])
        xi_L = list(reordered_solution.values())
        # print(solution[0].keys())
        # print(xi_L)
        Lagrangian = generateExpression(xi_L,terms,threshold)
        return Lagrangian

def STRidge(X0, y, lam, maxit, tol, normalize = 0, print_results = False):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse 
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

    This assumes y is only one column
    """

    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.float64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0
    
    # Get the standard ridge esitmate
    if lam != 0: w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d),X.T.dot(y),rcond=-1)[0]
    else: w = np.linalg.lstsq(X,y,rcond=-1)[0]
    # print(w)
    num_relevant = d
    biginds = np.where( abs(w) > tol)[0]
    
    # Threshold and continue
    for j in range(maxit):

        # Figure out which items to cut out
        smallinds = np.where( abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]
            
        # If nothing changes then stop
        if num_relevant == len(new_biginds): break
        else: num_relevant = len(new_biginds)
            
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0: 
                #if print_results: print "Tolerance too high - all coefficients set below tolerance"
                return w
            else: break
        biginds = new_biginds
        
        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0: w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y),rcond=-1)[0]
        else: w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=-1)[0]
        
        # print(w)

    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=-1)[0]
    
    if normalize != 0: return np.multiply(Mreg,w)
    else: return w