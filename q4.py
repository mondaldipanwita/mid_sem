##Mid_sem
##Name: Dipanwita Mondal
##Roll No: 23124006


import my_library as lib
def f(x):
    return x**2            # mass density is given as x^2
def g(x):
    return x**3                 # to calculate integration over x dm


# Applying Simpson's rule total mass
x0=0
xn=2                            # the beam is 2m long
tol=1e-4
m=5              #taking m intervals at first
I= lib.simpson13(f, x0, xn, 2)
I_m= lib.simpson13(f, x0,xn, m)
m=m+1
while abs ((I-I_m))>tol:
    I=lib.simpson13(f, x0, xn, m)
    I_m= lib.simpson13(f,x0, xn, m+2)
    #print(I,I_m,m)
    m+=1

## Applying Simpson's rule total mass
x0=0
xn=2                            # the beam is 2m long
tol=1e-4
m=5              #taking m intervals at first
I0= lib.simpson13(g, x0, xn, 2)
I_x= lib.simpson13(g, x0,xn, m)
m=m+1
while abs ((I0-I_x))>tol:
    I0=lib.simpson13(g, x0, xn, m)
    I_x= lib.simpson13(g,x0, xn, m+2)
    #print(I,I_m,m)
    m+=1

# Centre of mass
centre_mass=I_x/I_m

print("The centre of mass of the beam:", centre_mass,"m")




#__________________OUTPUT____________________________
#The centre of mass of the beam: 1.5 m







