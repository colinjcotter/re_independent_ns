from firedrake import *
from matplotlib import *
import numpy as np

#Some settings
n = 16 # number of grid points
h = 1/n # "length" of each side
viscosity = 1 #viscosity
c = 10 # dark magic
f = Constant((1,0))
gamma = Constant((100.0))

# Load mesh
mesh = UnitSquareMesh(n, n)

# Define function spaces
V = FunctionSpace(mesh, "BDM", 1)
Q = FunctionSpace(mesh, "DG", 0)
W = V * Q
#defining
x,y= SpatialCoordinate(mesh)
#defining the normal
n = FacetNormal(mesh)



# Boundary Conditions #

# No-slip boundary condition for velocity bottom, left and right,
noslip = Constant((0.0, 0.0))
bc1 = DirichletBC(W.sub(0), noslip, (1,2,3))

# Constant inflow Top
inflow = Constant((1.0, 0.0))
bc2 = DirichletBC(W.sub(0), inflow, 4)

#boundary conditions
bcs=(bc1,bc2)

#pressure
up = Function(W)

# Removing Pressure constant
nullspace = MixedVectorSpaceBasis(
    W, [W.sub(0), VectorSpaceBasis(constant=True)])

# Define variational problem #

#setting up trial and test functions
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

#Assembling LHS
f = Constant((1,0)) #currently we set force to be 0
L = inner(f, v)*dx

#dealing with viscous term
viscous_byparts1 = inner(grad(u), grad(v))*dx #this is the term over omega from the integration by parts
viscous_byparts2 = 2*inner(avg(outer(v,n)),avg(grad(u)))*dS #this the term over interior surfaces from integration by parts
viscous_symetry = 2*inner(avg(outer(u,n)),avg(grad(v)))*dS #this the term ensures symetry while not changing the continuous equation
viscous_stab = c*1/h*inner(jump(v),jump(u))*dS #stabilizes the equation, somehow
viscous_byparts2_ext = (inner(outer(v,n),grad(u)) + inner(outer(u,n),grad(v)))*ds #This deals with boundaries TOFIX : CONSIDER NON-0 BDARIEs 
viscous_ext = c/h*inner(v,u)*ds #this is a penalty term for the boundaries

viscous_term = (
    viscous_byparts1
    + viscous_byparts2
    + viscous_symetry
    + viscous_stab
    + viscous_byparts2_ext
    + viscous_ext # assembles everything
    )

graddiv_term = gamma*div(v)*div(u)*dx

a = viscosity*viscous_term + graddiv_term + q * div(u) * dx - p * div(v) * dx

#Solving problem #

parameters = {
    "ksp_type": "gmres",
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "lu"
}

pmass = q*p*dx

ap = viscous_term + graddiv_term  + (viscosity + gamma)*pmass

stokesproblem = LinearVariationalProblem(a,L, aP=aP,
                                         bcs=(bc1,bc2),
                                         nullspace=nullspace)

stokessolver = LinearVariationalProblem(stokesproblem, up,
                                        solver_parameters=parameters)

stokessolver.solve()

u, p = up.split()
u.rename("Velocity")
p.rename("Pressure")

 # Plot solution
File("stokes.pvd").write(u, p)
