# Ivy Weber, 2019-09-03

# Module defining polynomial class, and useful functions and methods for it

# Aims: Class defining real polynomials in r different variables
#
#       Methods: Initialise the polynomial from a numpy array input
#                Print polynomial as a string describing the function
#                Return the degrees in given variables of the polynomial
#                Evaluate the polynomial at a given value
#                Contracting (replacing one variable with another already in the polynomial)
#                Differentiate
#                Integrate from 0 to x
#
#       Functions: Addition
#                  Multiplication/Tensor product
#                  Powers of polynomials
#                  Generation from a set of roots (1 variable)
#
# Represent each polynomial internally as a numpy array in r dimensions.

#Import required modules

import math as m
import numpy as np

#Define functions and classes

class polynomial(object):
    """Class defining a polynomial"""

    def __init__(self, cf = np.zeros(1), var = []):

        sz = np.size(cf)

        if sz == 0:

            self.coeffs = np.zeros(1)

        elif (cf == np.zeros(np.shape(cf))).all():

            self.coeffs = np.zeros(1)
            
        else:

            self.coeffs = np.array(cf)
            
        self.degrees = np.shape(self.coeffs)
        
        try:

            self.dim = len(self.degrees)

        except:

            self.dim = 1
            self.degrees = np.array(self.degrees)
            
        self.degrees -= np.ones(self.dim)

        if self.dim < len(var):

            self.var = var[:self.dim]

        elif self.dim > len(var):

            self.var = np.zeros(self.dim)

            if len(var) != 0:

                self.var[:len(var)] = var[:]
                self.var[len(var):] = range(var[-1] + 1, self.dim)

            else:

                self.var[:] = list(range(0, self.dim))

        else:

            self.var = var[:]

        var_prime = self.var[:]
        self.var.sort()
        
        #Transpose the coefficient tensor in the same way 
        trans = np.zeros(len(var_prime))
        count = 0
        
        for v in var_prime:

            ind = 0
            
            for w in self.var:
                
                if v == w:

                    trans[count] = ind
                    break

                else:
                    
                    ind += 1

            count += 1

        trans = tuple(trans.astype(int))

        self.coeffs = np.transpose(self.coeffs, trans)
        self.degrees = np.array(np.shape(self.coeffs)) - np.ones(self.dim)
        

    def __str__(self):
        """Write a string showing the expanded polynomial"""
        #Proceed in order of the degree of each variable, favouring terms with
        #more earlier variables.
        #eg: 1 + x_1 + x_2 + x_1 x_2 + x_3 + x_1 x_3 + x_2 x_3 + x_1 x_2 x_3
        #    = (1 + x_1 + x_2 (1 + x_1)) + x_3 (1 + x_1 + x_2 (1 + x_1))
        output = ""

        if np.size(self.coeffs) == 1:
            if self.coeffs == 0:
                return "0"

    
        #deg_curr is describes the current term, with the entries given the degrees
        #of each variable in order. tot_deg_curr gives the total degree of the current
        #term
        deg_curr = np.zeros(self.dim)
        
        index_max = 1
        
        for p in self.degrees:
            
            index_max *= (p + 1)
        
        start = 1

        index_max = int(index_max)
        
        for index in range(0,index_max):

            #index gives the current position in the array
            temp_index = index
            index_count = 0
            
            while index_count < self.dim:

                deg_curr[index_count] = temp_index % (self.degrees[index_count] + 1)

                temp_index -= deg_curr[index_count]
                temp_index = temp_index // (self.degrees[index_count] + 1)

                index_count += 1
                
            
            pos = tuple(deg_curr.astype(int))
            c = self.coeffs[pos]
            
            if c < 0:

                sign = "- "
                start = 0

            elif c > 0 and not start:

                sign = "+ "

            elif c == 0:

                continue

            else:

                sign = ""
                start = 0

            output += sign

            if c != 1 or (start == 0 and index == 0):

                output += str(m.fabs(c)) + " "

            if index == 0:

                output += " "

            else:
                #Find the required combination of variables, add it to string
                var_ind = 0
                
                for power in pos:

                    if power == 1:

                        output += "x_" + str(self.var[var_ind]) + " "

                    elif power > 1:

                        output += "(x_" + str(self.var[var_ind]) + ")^" + str(power) + " "

                    var_ind += 1

        return output

    def __float__(self):
        
        try:

            ans = self.coeffs[0]

        except:

            ans = float(self.coeffs)

        return ans

    
    def evaluate(self, x, evals = [0]):
        """Evaluate the polynomial at real values given by x, for the variables specified by eval_dims"""

        eval_dims = []
        eval_dims_pos = []
        out_dims = []
        curr_pos = 0
        
        for d in self.var:
            if d in evals:

                eval_dims.append(d)
                eval_dims_pos.append(curr_pos)
                
            else:

                out_dims.append(d)

            curr_pos += 1
            
                
        try:
            m = len(x)
        except:
            m = 1
            
        n = len(eval_dims)

        if n == 0:

            #Return the same polynomial
            return self

        else:

            if m > n:

                print("ERROR: Too many input values of x_i. Taking only the first " + str(n) + " values.")
                input_x = x[:n]

            elif m < n:

                print("ERROR: Too few input values of x_i. Taking remaining values to be 0")
                input_x = np.zeros(n)

                try:
                    input_x[:m] = x[:]
                except:
                    input_x[0] = x


            else:

                try:
                    input_x = x[:]
                except:
                    input_x = x

            input_x = np.array(input_x)

            #Evaluate the required result, a polynomial with self.dims - n variables,
            #by taking tensor products between the coefficient tensor and vectors of
            #the form (1, x, x^2, ..., x^m)
            M = int(max(self.degrees))

            vectors = np.ones( (M + 1, n) )

            for power in range(0, M):
            
                #Create the next entries in each vector from the last set of entries
                vectors[power + 1, :] = vectors[power, :] * input_x

            output = self.coeffs[:]

            for vec_ind in range(0,n):

                vec_curr = vectors[:int(self.degrees[vec_ind] + 1), vec_ind]
                output = np.tensordot(output, vec_curr, axes=([eval_dims_pos[vec_ind] - vec_ind],[0]) )

            if np.size(output) > 1:

                return polynomial(output, out_dims)

            else:
    
                return float(output)

    def diff(self, d = 0):
        """Return a derivative of the current polynomial with respect to dimension d"""

        #Check that the polynomial has terms with x_d in
        indicator = 0
        diff_pos = 0
        diff_dims = list(self.var[:])
        
        for dim in self.var:
            if dim == d:

                indicator = 1
                break

            else:

                diff_pos += 1
                

        if indicator == 0:

            return polynomial()

        else:

            dims_new = list(np.shape(self.coeffs))
            remove_ind = 0
            
            if dims_new[diff_pos] == 2:

                dims_new.pop(diff_pos)
                diff_dims.pop(diff_pos)
                remove_ind = 1

            else:
                    
                dims_new[diff_pos] -= 1

            if remove_ind == 1:

                diff_mat = np.array([0,1])
                output = np.tensordot(diff_mat, self.coeffs, axes = ([0],[diff_pos]))

            else:
                #Define a matrix, with which a tensor product can be taken to produce
                #the required result
                    #output[...,k-1,...] = k input[...,k,...]
                diff_mat = np.zeros( (dims_new[diff_pos], dims_new[diff_pos] + 1) )
                
                for k in range(0,dims_new[diff_pos]):

                    diff_mat[k,k + 1] = k + 1

                output = np.tensordot(diff_mat, self.coeffs, axes = ([1],[diff_pos]))
                output = np.moveaxis(output, 0, diff_pos)
                
                
            return polynomial(output, diff_dims)

    def integ(self, d = 0):
        """Integrate the polynomial from 0 to x_d, with respect to x_d"""

        #Check that the polynomial has terms with x_d in
        indicator = 0
        integ_pos = 0
        integ_dims = list(self.var[:])
        integ_shape = np.shape(self.coeffs)
        
        for dim in integ_dims:
            if dim == d:

                indicator = 1
                break

            elif dim > d:

                break

            else:

                integ_pos += 1

        #If d was not found, insert it in the correct position
        if indicator == 0 or np.size(self.coeffs) == 1:
            
            integ_dims.insert(d, integ_pos)

            output = np.tensordot(np.array([0,1]), self.coeffs, axes = 0)
            output = np.moveaxis(output, 0, integ_pos)

        else:

            N = int(integ_shape[integ_pos])

            mat_integ = np.zeros( (N + 1, N) )

            for k in range(0,N):

                mat_integ[k + 1, k] = 1/(k + 1)

            output = np.tensordot(mat_integ, self.coeffs, axes = ([1], [integ_pos]))
            output = np.moveaxis(output, 0, integ_pos)

        return polynomial(output, integ_dims)
    

    def substitute(self, d_1 = 0, d_2 = 0):
        """Replaces all terms in the polynomial of x_(d_1) with x_(d_2), summing as necessary"""

        #Check that x_(d_1), x_(d_2) are featured and different
        ind_1 = 0
        ind_2 = 0
        pos_1 = 0
        pos_2 = 0
        count = 0

        new_var = list(self.var[:])
        
        for dim in self.var:

            if dim == d_1:

                ind_1 = 1
                pos_1 = count

            if dim == d_2:

                ind_2 = 1
                pos_2 = count

            count += 1

        if ind_1 == 0 or (pos_1 == pos_2 and ind_2 == 1):

            return self

        elif ind_2 == 0:

            #Simply reassign the label for x_(d_1), since x_(d_2) does not appear yet            
            new_var[pos_1] = d_2
            
            return polynomial(self.coeffs, new_var)

        else:

            #Both x_(d_1) and x_(d_2) actively appear in the polynomial
            deg_1 = int(self.degrees[pos_1])
            deg_2 = int(self.degrees[pos_2])
            deg_final = deg_1 + deg_2

            tens_cont = np.zeros( (deg_1 + 1, deg_2 + 1, deg_final + 1) )

            for i in range(0, deg_1 + 1):
                for j in range(0, deg_2 + 1):

                    tens_cont[i][j][i + j] = 1

            output = np.tensordot(tens_cont, self.coeffs, axes = ([0,1],[pos_1, pos_2]))
            output = np.moveaxis(output,0,pos_2)
            new_var.pop(pos_1)
            
            return polynomial(output, new_var)


#Define polynomial functions here

def poly_mult(p_1, p_2):

    """Returns the product of two polynomials in multiple variables"""

    #Find common variables for later contraction, with positions in both polynomials
    #Find the set union of the variables
    
    common_var = []
    out_var = []
    all_var = []
    count_1 = 0
    
    for dim_1 in p_1.var:

        out_var.append(dim_1)
        all_var.append(dim_1)
        
        count_2 = 0
        
        for dim_2 in p_2.var:

            if dim_1 == dim_2:

                common_var.append([dim_1, count_1, count_2])
                break

            else:

                count_2 += 1


        count_1 += 1

    mr_big = max(all_var)
    
    for dim_2 in p_2.var:

        all_var.append(dim_2 + mr_big + 1)

        if not dim_2 in p_1.var:

            out_var.append(dim_2)

    #Take a tensor product of the coefficients

    output = np.tensordot(p_1.coeffs, p_2.coeffs, axes = 0)
    p_curr = polynomial(output, all_var)

    #Contract on all duplicate variables, removing the occurence due to p_2 in each case

    #After n contractions, the positions of shared variable [r, rp_1, rp_2] will be rp_1, rp_2 - n + p_1.dims
    n = 0
    
    for var in common_var:

        p_curr = p_curr.substitute(all_var[var[2] - n + p_1.dim], all_var[var[1]])
        n += 1

    for var in all_var:

        if var > mr_big:

            p_curr = p_curr.substitute(var, var - mr_big - 1)

    return p_curr

def poly_add(p_1, p_2):

    #Find degrees of result
    max_1 = max(p_1.var)
    max_2 = max(p_2.var)
    max_out = max(max_1, max_2)

    var_out = []
    var_out_in_1 = []
    var_out_in_2 = []
    
    shape_out = []
    count = 0
    count_1 = 0
    count_2 = 0

    for var in range(0, max_out + 1):

        var_ind = 0

        if var in p_1.var:

            var_ind += 1
            var_out_in_1.append(count)
            
            count += 1
            count_1 += 1

        if var in p_2.var:

            var_ind += 2
            count_2 += 1

            if not var in p_1.var:

                var_out_in_2.append(count)
                count += 1

            else:

                var_out_in_2.append(count - 1)
                

            

        #0 = neither, 1 = p_1 only, 2 = p_2 only, 3 = both

        var_out.append(var)

        if var_ind == 1:

            size = int(1 + p_1.degrees[count_1 - 1])

        elif var_ind == 2:

            size = int(1 + p_2.degrees[count_2 - 1])

        elif var_ind == 3:

            size = int(1 + max(p_1.degrees[count_1 - 1], p_2.degrees[count_2 - 1]))

        shape_out.append(size)

    p_1_new = p_1.coeffs[:]
    p_2_new = p_2.coeffs[:]
    
    #Positioning an array within part of a larger array of zeros is a linear operation, so can be done by tensor arithmetic
    #Expand existing axes to requirements, then add extras
        #To expand, premultiply by matrix [I,0]^T along required axis
        #To add an axis, take a tensor product with [1,0,0,...]^T

    count_1 = 0
    count_2 = 0
    
    for var in range(0,len(var_out)):

        if var_out[var] in p_1.var:

            l_coord = int(1 + p_1.degrees[count_1])

            matrix = np.zeros( (shape_out[var], l_coord) )
            matrix[:l_coord,:] = np.eye(l_coord)

            count_1 += 1

            p_1_new = np.tensordot(matrix, p_1_new, axes = ([1],[var]))
            p_1_new = np.moveaxis(p_1_new, 0, var)

        else:

            matrix = np.zeros(shape_out[var])
            matrix[0] = 1

            p_1_new = np.tensordot(matrix, p_1_new, axes = 0)
            p_1_new = np.moveaxis(p_1_new, 0, var)


        if var_out[var] in p_2.var:

            l_coord = int(1 + p_2.degrees[count_2])
            
            matrix = np.zeros( (shape_out[var], l_coord) )
            matrix[:l_coord,:] = np.eye(l_coord)

            count_2 += 1

            p_2_new = np.tensordot(matrix, p_2_new, axes = ([1],[var]))
            p_2_new = np.moveaxis(p_2_new, 0, var)

        else:

            matrix = np.zeros(shape_out[var])
            matrix[0] = 1

            p_2_new = np.tensordot(matrix, p_2_new, axes = 0)
            p_2_new = np.moveaxis(p_2_new, 0, var)



    poly_out = p_1_new + p_2_new

        
    return polynomial(poly_out, var_out)
        
def poly_pow(poly, exp):

    #Require that exp is an integer

    try:
        
        leng = len(exp)
        
    except:
        leng = 1
        exp_use = m.fabs(m.floor(exp))
        
    
    if leng != 1:

        print("ERROR: exp must be a positive integer")

        if leng == 0:

            return polynomial([1],[0])

        else:

            exp_use = exp[0]

    p_out = polynomial([1],[0])
    
    while exp_use > 0:

        p_out = poly_mult(p_out, poly)
        exp_use -= 1

    return p_out

def poly_from_roots(roots = []):

    """Define a polynomial in one variable from the values of that variable where it is zero"""

    p_out = polynomial([1],[0])

    try:

        leng = len(roots)

    except:

        roots = [roots]
        

    for r in roots:

        p_mult = polynomial([-r,1],[0])
        p_out = poly_mult(p_out, p_mult)

    return p_out


#MAIN:

if __name__ == "__main__":

    #<Add code here>
    p0 = polynomial()
    p1 = polynomial([0,0,0,1],[1])
    p2 = polynomial([[0,1,2],[2,2,2],[3,0,1]], [0,1])
    p3 = polynomial([[1,2,3],[4,5,6]], [0,1])
    p4 = polynomial([[[1,2],[3,4]],[[5,6],[7,8]]], [0,1,2])

    print("TESTING STRING METHOD: \n")
    
    print(p0)
    print(p1)
    print(p2)
    print(p3)
    print(p4)

    print("\n TESTING EVALUATION METHOD: \n")

    print(p1.evaluate(2, [1]))
    print(p1.evaluate(2, [0]))

    print("")

    print(p2.evaluate(1,[0]))
    print(p2.evaluate(1,[0,1]))
    print(p2.evaluate([1,1,1], [1]))

    print("\n TESTING DIFFERENTIATION METHOD: \n")

    print(p1.diff())
    print(p1.diff(1))

    print("")

    print(p2.diff())
    print(p2.diff(1))

    print("")

    print(p3.diff(1))
    print(p4.diff(1))
    print(p4.diff(2))

    print("\n TESTING INTEGRAL METHOD: \n")

    print(p0.integ())

    print("")

    print(p1.integ())
    print(p1.integ(1))

    print("")

    print(p2.integ())
    print(p4.integ(1))

    print("\n TESTING SUBSTITUTION METHOD: \n")

    print(p1.substitute(1,0))
    print(p1.substitute(0,1))

    print("")

    print(p2.substitute(1,0))
    print(p2.substitute(1,2))
    print(p2.substitute(0,2))

    print("\n TESTING MULTIPLICATION FUNCTION: \n")
    
    print(poly_mult(p1, p2))
    print(poly_mult(p0,p4))
    print(poly_mult(p2,p3))

    print("\n TESTING ADDITION FUNCTION: \n")
    
    print(poly_add(p0,p1))
    print(poly_add(p1,p2))
    print(poly_add(p2,p4))

    print("\n TESTING POWER FUNCTION: \n")

    print(poly_pow(polynomial([1,1],[0]), 4))
    print(poly_pow(p1,3))
    print(poly_pow(p2,2))

    print("\n TESTING ROOT GENERATION FUNCTION: \n")

    print(poly_from_roots())
    print(poly_from_roots(5))
    print(poly_from_roots(np.ones(3)))
    print(poly_from_roots([1,2,3]))
    
#END 
