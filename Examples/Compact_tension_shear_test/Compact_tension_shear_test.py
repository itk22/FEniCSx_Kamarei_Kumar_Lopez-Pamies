from mpi4py import MPI
from dolfinx import mesh, fem, io, plot, nls, log, geometry, la
from dolfinx import cpp as _cpp
from dolfinx import default_real_type
import basix
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import time
import gmsh
import os
from dolfinx.fem.petsc import NonlinearProblem


def adjust_array_shape(input_array):
    if input_array.shape == (2,):  # Check if the shape is (2,)
        adjusted_array = np.append(input_array, 0.0)  # Append 0.0 to the array
        return adjusted_array
    else:
        return input_array


def evaluate_function(domain, u, x):
    """[summary]
        Helps evaluated a function at a point `x` in parallel
    Args:
        domain ([dolfinx.mesh.Mesh]): [mesh]
        u ([dolfin.Function]): [function to be evaluated]
        x ([Union(tuple, list, numpy.ndarray)]): [point at which to evaluate function `u`]

    Returns:
        [numpy.ndarray]: [function evaluated at point `x`]
    """

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)

    if isinstance(x, np.ndarray):
        # If x is already a NumPy array
        points0 = x
    elif isinstance(x, (tuple, list)):
        # If x is a tuple or list, convert it to a NumPy array
        points0 = np.array(x)
    else:
        # Handle the case if x is of an unsupported type
        points0 = None

    points = adjust_array_shape(points0)

    u_value = []

    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    if len(colliding_cells.links(0)) > 0:
        u_value = u.eval(points, colliding_cells.links(0)[0])
        u_value = domain.comm.gather(u_value, root=0)

    if domain.comm.rank == 0 and u_value:
        return u_value[0]
    return None


def main():
    # Material properties
    E, nu = ScalarType(70000), ScalarType(0.22)  # Young's modulus and Poisson's ratio
    mu, lmbda, kappa = (
        E / (2 * (1 + nu)),
        E * nu / ((1 + nu) * (1 - 2 * nu)),
        E / (3 * (1 - 2 * nu)),
    )
    Gc = ScalarType(0.010)  # Critical energy release rate
    sts, scs = (
        ScalarType(40),
        ScalarType(1000),
    )  # Tensile strength and compressive strength
    shs = ScalarType(27.8)
    Wts = sts**2 / (2 * E)
    Whs = shs**2 / (2 * kappa)

    # The regularization length
    eps = 0.16
    h = eps / 5

    delta = (1 + 3 * h / (8 * eps)) ** (-2) * (
        (sts + (1 + 2 * np.sqrt(3)) * shs) / ((8 + 3 * np.sqrt(3)) * shs)
    ) * 3 * Gc / (16 * Wts * eps) + (1 + 3 * h / (8 * eps)) ** (-1) * (2 / 5)
    comm = MPI.COMM_WORLD
    comm_rank = MPI.COMM_WORLD.rank
    log.set_log_level(log.LogLevel.ERROR)

    # Geometry parameters
    L = 10  # Length of the outer rectangle

    ac = 5  # Length of the crack
    y_ac = 5  # Y coordinate of the crack
    theta = np.radians(0.05)  # Angle of the sharp crack

    if comm_rank == 0:
        gmsh.initialize()

        gmsh.model.add("2Dcompactshear")

        # Create outer box
        block = gmsh.model.occ.addRectangle(0, 0, 0, L, L)

        # Create the crack
        half_opening = ac * np.tan(theta / 2)

        # Crack tip
        p_tip = (ac, y_ac)

        # Two outer points of the wedge (on left edge)
        p1 = (0, y_ac + half_opening)
        p2 = (0, y_ac - half_opening)

        # Create the triangular crack shape
        pt1 = gmsh.model.occ.addPoint(*p_tip, 0)
        pt2 = gmsh.model.occ.addPoint(*p1, 0)
        pt3 = gmsh.model.occ.addPoint(*p2, 0)

        l1 = gmsh.model.occ.addLine(pt1, pt2)
        l2 = gmsh.model.occ.addLine(pt2, pt3)
        l3 = gmsh.model.occ.addLine(pt3, pt1)

        crack_loop = gmsh.model.occ.addCurveLoop([l1, l2, l3])
        crack_surface = gmsh.model.occ.addPlaneSurface([crack_loop])

        # --- Cut the wedge crack from the block ---\
        cut = gmsh.model.occ.cut([(2, block)], [(2, crack_surface)])
        cut_tag = cut[0][0][1]  # Resulting surface tag after cut

        # Synchronize to reflect the changes in the model
        gmsh.model.occ.synchronize()

        # Add physical group for the volume (the tube itself)
        dcb_group = gmsh.model.addPhysicalGroup(2, [cut_tag])
        gmsh.model.setPhysicalName(2, dcb_group, "block")

        # Distance field from crack tip point
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "PointsList", [pt1])

        # Threshold field for size variation
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", h)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", 5 * h)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 2)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 3)

        # Set this as the background mesh
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        # Generate and optimize the mesh
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.optimize("Netgen")

    partitioner = dolfinx.cpp.mesh.create_cell_partitioner(
        dolfinx.mesh.GhostMode.shared_facet
    )
    mesh_data = io.gmsh.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=2, partitioner=partitioner
    )

    if comm_rank == 0:
        gmsh.finalize()

    domain = mesh_data[0]

    if comm_rank == 0:
        with dolfinx.io.XDMFFile(domain.comm, "refined_mesh.xdmf", "w") as xdmf:
            xdmf.write_mesh(domain)
        if not os.path.exists("paraview"):
            os.mkdir("paraview")

    with io.XDMFFile(domain.comm, "paraview/2DCTS.xdmf", "w") as file_results:
        file_results.write_mesh(domain)

    # Defining the function spaces
    V = fem.functionspace(
        domain, ("CG", 1, (domain.geometry.dim,))
    )  # Function space for u
    Y = fem.functionspace(domain, ("CG", 1))  # Function space for z

    def bottom(x):
        return np.isclose(x[1], 0)

    def top(x):
        return np.isclose(x[1], L)

    def cracktip(x):
        return np.logical_and.reduce(
            (x[1] < y_ac + 1e-4, x[1] > y_ac - 1e-4, x[0] < ac + h, x[0] > ac - 5 * h)
        )

    fdim = domain.topology.dim - 1

    bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
    top_facets = mesh.locate_entities_boundary(domain, fdim, top)

    cracktip_facets = mesh.locate_entities_boundary(domain, fdim, cracktip)

    dofs_bottom0 = fem.locate_dofs_topological(V.sub(0), fdim, bottom_facets)
    dofs_bottom1 = fem.locate_dofs_topological(V.sub(1), fdim, bottom_facets)

    dofs_top0 = fem.locate_dofs_topological(V.sub(0), fdim, top_facets)
    dofs_top1 = fem.locate_dofs_topological(V.sub(1), fdim, top_facets)

    dofs_cracktip = fem.locate_dofs_topological(Y, fdim, cracktip_facets)

    bcb0 = fem.dirichletbc(ScalarType(0), dofs_bottom0, V.sub(0))
    bcb1 = fem.dirichletbc(ScalarType(0), dofs_bottom1, V.sub(1))

    bct0 = fem.dirichletbc(ScalarType(0), dofs_top0, V.sub(0))
    bct1 = fem.dirichletbc(ScalarType(0), dofs_top1, V.sub(1))

    bcs = [bcb0, bcb1, bct0, bct1]

    bct_z = fem.dirichletbc(
        ScalarType(1), fem.locate_dofs_topological(Y, fdim, top_facets), Y
    )
    bcb_z = fem.dirichletbc(
        ScalarType(1), fem.locate_dofs_topological(Y, fdim, bottom_facets), Y
    )

    bc_z_tip = fem.dirichletbc(ScalarType(0), dofs_cracktip, Y)

    bcs_z = [bct_z, bcb_z, bc_z_tip]

    marked_facets = np.hstack([bottom_facets, top_facets])
    marked_values = np.hstack(
        [np.full_like(bottom_facets, 1), np.full_like(top_facets, 2)]
    )
    sorted_facets = np.argsort(marked_facets)
    facet_tag = mesh.meshtags(
        domain,
        domain.topology.dim - 1,
        marked_facets[sorted_facets],
        marked_values[sorted_facets],
    )

    metadata = {"quadrature_degree": 2}
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag, metadata=metadata)
    dS = ufl.Measure("dS", domain=domain, metadata=metadata)
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)

    # Define functions
    du = ufl.TrialFunction(V)  # Incremental displacement
    v = ufl.TestFunction(V)  # Test function for u
    u = fem.Function(V, name="displacement")  # Displacement from previous iteration
    u_inc = fem.Function(V)
    dz = ufl.TrialFunction(Y)  # Incremental phase field
    y = ufl.TestFunction(Y)  # Test function for z
    z = fem.Function(Y, name="phasefield")  # Phase field from previous iteration
    z_inc = fem.Function(Y)
    d = len(u)

    u.x.array[:] = 0.0

    z.x.array[:] = 1.0

    u_prev = fem.Function(V)
    u_prev.x.array[:] = u.x.array
    z_prev = fem.Function(Y)
    z_prev.x.array[:] = z.x.array

    y_dofs_top = fem.locate_dofs_topological(V.sub(0), fdim, top_facets)

    # Stored energy, strain and stress functions in linear isotropic elasticity (plane strain)

    def energy(v):
        return (
            mu * (ufl.inner(ufl.sym(ufl.grad(v)), ufl.sym(ufl.grad(v))))
            + 0.5 * (lmbda) * (ufl.tr(ufl.sym(ufl.grad(v)))) ** 2
        )

    def epsilon(v):
        return ufl.sym(ufl.grad(v))

    def sigma(v):
        return 2.0 * mu * ufl.sym(ufl.grad(v)) + (lmbda) * ufl.tr(
            ufl.sym(ufl.grad(v))
        ) * ufl.Identity(len(v))

    def sigmavm(sig, v):
        return ufl.sqrt(
            1
            / 2
            * (
                ufl.inner(
                    sig - 1 / 3 * (1 + nu) * ufl.tr(sig) * ufl.Identity(len(v)),
                    sig - 1 / 3 * (1 + nu) * ufl.tr(sig) * ufl.Identity(len(v)),
                )
                + ((2 * nu / 3 - 1 / 3) ** 2) * ufl.tr(sig) ** 2
            )
        )

    eta = 0.0
    # Stored energy density
    psi1 = (z**2 + eta) * (energy(u))
    psi11 = energy(u)
    # Total potential energy
    Pi = psi1 * dx
    # Compute first variation of Pi (directional derivative about u in the direction of v)
    R = ufl.derivative(Pi, u, v)
    # Compute Jacobian of R
    Jac = ufl.derivative(R, u, du)

    I1 = (z**2) * (1 + nu) * ufl.tr(sigma(u))
    SQJ2 = (z**2) * sigmavm(sigma(u), u)

    alpha1 = (delta * Gc) / (shs * 8 * eps) - (2 * Whs) / (3 * shs)
    alpha2 = (
        (3**0.5 * (3 * shs - sts) * delta * Gc) / (shs * sts * 8 * eps)
        + (2 * Whs) / (3**0.5 * shs)
        - (2 * 3**0.5 * Wts) / (sts)
    )

    ce = alpha2 * SQJ2 + alpha1 * I1 - z * (1 - ufl.sqrt(I1**2) / I1) * psi11

    # Balance of configurational forces PDE
    pen = 1000 * (3 * Gc / 8 / eps) * ufl.conditional(ufl.lt(delta, 1), 1, delta)
    Wv = pen / 2 * ((abs(z) - z) ** 2 + (abs(1 - z) - (1 - z)) ** 2) * dx

    R_z = (
        y * 2 * z * (psi11) * dx
        + y * (ce) * dx
        + 3
        * delta
        * Gc
        / 8
        * (-y / eps + 2 * eps * ufl.inner(ufl.grad(z), ufl.grad(y)))
        * dx
        + ufl.derivative(Wv, z, y)
    )

    # Compute Jacobian of R_z
    Jac_z = ufl.derivative(R_z, z, dz)

    # Define maximum displacement
    disp_max = L * 0.0002
    # time-stepping parameters

    T = 1
    Totalsteps = 500
    startstepsize = 1 / Totalsteps
    stepsize = startstepsize
    t = stepsize
    step = 1
    rtol = 1e-9
    rnorm_stag0 = 1
    rnorm_stag = 1
    printsteps = 10

    # PETSc solver options
    petsc_options_u = {
        "snes_type": "newtonls",
        "snes_max_it": 10,
        "snes_atol": 1.0e-8,
        "snes_rtol": 1.0e-7,
        "snes_error_if_not_converged": False,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    petsc_options_z = {
        "snes_type": "newtonls",
        "snes_max_it": 10,
        "snes_atol": 1.0e-8,
        "snes_rtol": 1.0e-7,
        "snes_error_if_not_converged": False,
        "ksp_type": "preonly",
        "pc_type": "lu",
    }

    # Create nonlinear problem
    problem_u = NonlinearProblem(
        R, u, bcs, Jac, petsc_options=petsc_options_u, petsc_options_prefix="problem_u_"
    )

    # Create nonlinear problem
    problem_z = NonlinearProblem(
        R_z,
        z,
        bcs_z,
        Jac_z,
        petsc_options=petsc_options_z,
        petsc_options_prefix="problem_z_",
    )

    t1 = 0.0

    while t - stepsize < T:
        if comm_rank == 0:
            print("Step= %d" % step, "t= %f" % t, "Stepsize= %e" % stepsize)

        bct0.g.value[...] = ScalarType(t / T * disp_max)
        bcb0.g.value[...] = ScalarType(-t / T * disp_max)
        stag_iter = 1
        rnorm_stag = 1
        while stag_iter < 100 and rnorm_stag / rnorm_stag0 > 1e-7:
            start_time = time.time()
            ##############################################################
            # PDE for u
            ##############################################################
            problem_u.solve()
            # u.x.scatter_forward()
            ##############################################################
            # PDE for z
            ##############################################################
            problem_z.solve()
            # z.x.scatter_forward()
            ##############################################################

            zmin = domain.comm.allreduce(np.min(z.x.array), op=MPI.MIN)

            if comm_rank == 0:
                print(zmin)

            if comm_rank == 0:
                print('--- %s seconds ---" % (time.time() - start_time)')

            ###############################################################
            # Residual check for stag loop
            ###############################################################
            b_e = fem.petsc.assemble_vector(fem.form(-R))
            fint = b_e.copy()
            fem.petsc.set_bc(b_e, bcs)

            rnorm_stag = b_e.norm()
            stag_iter += 1

        ########### Post-processing ##############

        u_prev.x.array[:] = u.x.array
        z_prev.x.array[:] = z.x.array

        # Calculate Reaction

        Fx = domain.comm.allreduce(np.sum(b_e[y_dofs_top]), op=MPI.SUM)
        z_x_val = evaluate_function(domain, z, (ac + eps, 0.0))
        z_x = 1.0
        if z_x_val is not None:
            z_x = z_x_val[0]

        if comm_rank == 0:
            print(Fx)
            print(z_x)
            with open("Glass_CTS.txt", "a") as rfile:
                rfile.write("%s %s %s %s\n" % (str(t), str(zmin), str(z_x), str(Fx)))

        if step % printsteps == 0:
            if comm_rank == 0:
                if not os.path.exists("paraview"):
                    os.makedirs("paraview", exist_ok=True)
            # XDMF files opened in 'w' mode overwrite, so ideally we keep it open or use 'a'.
            # However, in script we probably want to keep it open outside the loop if possible or append.
            # The original code opened it once per step which overwrites if filename is same,
            # but filename varied in user code: "paraview/2D_Ratio40checkeps2_" + str(step) + ".xdmf"
            # Here we might want to just append to a time series file.
            # For now replicating the single file write logic if intended, or better, keep file open.
            pass  # We already have file_results open outside? Wait, in Cylindrical it was wrapped.
            # Let's fix file output logic to match best practices.

            # Re-opening 'a' mode to append time steps
            with io.XDMFFile(domain.comm, "paraview/2DCTS.xdmf", "a") as file_res:
                file_res.write_function(u, t)
                file_res.write_function(z, t)

        if z_x < 0.05 or np.isnan(zmin):
            t1 = t
            break

        # time stepping
        step += 1
        t += stepsize


if __name__ == "__main__":
    main()
