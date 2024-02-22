##Mid_sem
##Name:Dipanwita Mondal
##Roll No: 23124006


import my_library as lib

#linear equations in the form of input matrices

A=[[1,-1,4,0,2,9],
   [0,5,-2,7,8,4],
   [1,0,5,7,3,-2],
   [6,-1,2,3,0,8],
   [-4,2,0,5,-5,3],
   [0,7,-1,5,4,-2]]
b=[19,2,13,-7,-9,2]
x=lib.solve_linear_equations_LU(A,b)
print("Solution vector:",x)

#---------------OUTPUT--------------------
#Solution vector: [-1.761817043997862, 0.8962280338740133, 4.051931404116158, -1.6171308025395421, 2.041913538501913, 0.15183248715593525]
