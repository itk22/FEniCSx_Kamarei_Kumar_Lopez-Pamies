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
import os
import gmsh
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
    E, nu = ScalarType(15000), ScalarType(0.25)  # Young's modulus and Poisson's ratio
    mu, lmbda, kappa = (
        E / (2 * (1 + nu)),
        E * nu / ((1 + nu) * (1 - 2 * nu)),
        E / (3 * (1 - 2 * nu)),
    )
    Gc = ScalarType(0.010)  # Critical energy release rate
    sts, scs = (
        ScalarType(10),
        ScalarType(100),
    )  # Tensile strength and compressive strength
    shs = (2 / 3) * sts * scs / (scs - sts)
    Wts = sts**2 / (2 * E)
    Whs = shs**2 / (2 * kappa)

    # Irwin characteristic length
    lch = 3 * Gc * E / 8 / (sts**2)
    # The regularization length
    eps = 0.5  # epsilon should not be chosen to be too large compared to lch. Typically eps<4*lch should work

    h = eps / 5

    delta = (1 + 3 * h / (8 * eps)) ** (-2) * (
        (sts + (1 + 2 * np.sqrt(3)) * shs) / ((8 + 3 * np.sqrt(3)) * shs)
    ) * 3 * Gc / (16 * Wts * eps) + (1 + 3 * h / (8 * eps)) ** (-1) * (2 / 5)
    comm = MPI.COMM_WORLD
    comm_rank = MPI.COMM_WORLD.rank
    log.set_log_level(log.LogLevel.ERROR)

    # Geometry of the single edge notch geometry
    Rad = 5  # notch length
    W, L = 15.0, 50.0  # making use of symmetry

    mesh_comm = MPI.COMM_WORLD
    model_rank = 0

    if comm_rank == 0:
        gmsh.initialize()
        gmsh.model.add("Rectangle minus Circle")
        gmsh.model.setCurrent("Rectangle minus Circle")

        rectangle_dim_tags = gmsh.model.occ.addRectangle(0, 0, 0, W, L)
        circle_dim_tags = gmsh.model.occ.addDisk(0, 0, 0, Rad, Rad)
        model_dim_tags = gmsh.model.occ.cut(
            [(2, rectangle_dim_tags)], [(2, circle_dim_tags)]
        )
        gmsh.model.occ.synchronize()

        # Add physical tag 1 for exterior surfaces
        boundary = gmsh.model.getBoundary(model_dim_tags[0], oriented=False)
        boundary_ids = [b[1] for b in boundary]
        gmsh.model.addPhysicalGroup(1, boundary_ids, tag=1)
        gmsh.model.setPhysicalName(1, 1, "boundary of rectangle")

        # Add physical tag 2 for the volume
        volume_entities = [model[1] for model in gmsh.model.getEntities(2)]
        gmsh.model.addPhysicalGroup(2, volume_entities, tag=2)
        gmsh.model.setPhysicalName(2, 2, "rectangle area")

        # Generating Mesh
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(1)

    partitioner = dolfinx.cpp.mesh.create_cell_partitioner(
        dolfinx.mesh.GhostMode.shared_facet
    )
    # Import the mesh on all ranks (rank 0 reads/generates, others receive)
    # Note: io.gmsh.model_to_mesh needs to be called on all ranks, but model only exists on rank 0 if we wrap it in if comm_rank==0 above?
    # Actually io.gmsh.model_to_mesh takes comm and rank. The model should be available on rank 0.

    # Wait, the original code broadcasted the model?
    # "model = MPI.COMM_WORLD.bcast(gmsh.model, root=0)" <- This line in original notebook is suspicious/wrong for gmsh model object which is C++ binding.
    # Usually one does:

    if comm_rank == 0:
        # We don't need to broadcast the model object itself, dolfinx handles the distribution from rank 0
        pass

    mesh_data = io.gmsh.model_to_mesh(
        gmsh.model, MPI.COMM_WORLD, 0, gdim=2, partitioner=partitioner
    )

    if comm_rank == 0:
        gmsh.finalize()

    MPI.COMM_WORLD.barrier()
    domain = mesh_data[0]

    def cell_criterion(x):
        """Given mesh coordinates, return if each point
        satisfies x[1]<Rad

        :param x: Input coordinates, shape (num_points, 3)
        :returns: Boolean array of shape (num_points, )
        """
        return x[1] < Rad

    ir = 0
    while ir < 2:
        domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)
        cells_local = np.arange(
            domain.topology.index_map(domain.topology.dim).size_local, dtype=np.int32
        )
        midpoints = dolfinx.mesh.compute_midpoints(
            domain, domain.topology.dim, cells_local
        ).T
        should_refine = np.flatnonzero(cell_criterion(midpoints)).astype(np.int32)
        domain.topology.create_entities(1)
        local_edges = dolfinx.mesh.compute_incident_entities(
            domain.topology, should_refine, domain.topology.dim, 1
        )
        domain = dolfinx.mesh.refine(domain, local_edges)[0]
        ir += 1

    def cell_criterion2(x):
        """Given mesh coordinates, return if each point
        satisfies (x[1]<2*eps)

        :param x: Input coordinates, shape (num_points, 3)
        :returns: Boolean array of shape (num_points, )
        """
        return x[1] < 2 * eps

    ir = 0
    while ir < 2:
        domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)
        cells_local = np.arange(
            domain.topology.index_map(domain.topology.dim).size_local, dtype=np.int32
        )
        midpoints = dolfinx.mesh.compute_midpoints(
            domain, domain.topology.dim, cells_local
        ).T
        should_refine = np.flatnonzero(cell_criterion2(midpoints)).astype(np.int32)
        domain.topology.create_entities(1)
        local_edges = dolfinx.mesh.compute_incident_entities(
            domain.topology, should_refine, domain.topology.dim, 1
        )
        domain = dolfinx.mesh.refine(domain, local_edges)[0]
        ir += 1

    # Ensure output directory exists
    if comm_rank == 0:
        if not os.path.exists("out_gmsh"):
            os.makedirs("out_gmsh")

    with io.XDMFFile(
        domain.comm, f"out_gmsh/mesh_rank_{MPI.COMM_WORLD.rank}.xdmf", "w"
    ) as file:
        file.write_mesh(domain)
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

    # Defining the function spaces
    V = fem.functionspace(
        domain, ("CG", 1, (domain.geometry.dim,))
    )  # Function space for u
    Y = fem.functionspace(domain, ("CG", 1))  # Function space for z

    maxdisp = 0.03

    def left(x):
        return np.isclose(x[0], 0)

    def front(x):
        return np.isclose(x[0], W)

    def top(x):
        return np.isclose(x[1], L)

    def bottom(x):
        return (x[1] < 1e-4) & (x[0] > Rad - 1e-4)

    def cracktip(x):
        return (x[1] < 1e-4) & (x[0] > Rad - h * 8 * 2) & (x[0] < Rad + 1e-4)

    def outer(x):
        return x[1] > L / 10

    fdim = domain.topology.dim - 1
    left_facets = mesh.locate_entities_boundary(domain, fdim, left)
    front_facets = mesh.locate_entities_boundary(domain, fdim, front)
    top_facets = mesh.locate_entities_boundary(domain, fdim, top)
    bottom_facets = mesh.locate_entities(domain, fdim, bottom)
    cracktip_facets = mesh.locate_entities(domain, fdim, cracktip)
    outer_facets = mesh.locate_entities(domain, fdim, outer)

    dofs_left = fem.locate_dofs_topological(V.sub(0), fdim, left_facets)
    dofs_top = fem.locate_dofs_topological(V.sub(1), fdim, top_facets)
    dofs_bottom = fem.locate_dofs_topological(V.sub(1), fdim, bottom_facets)

    dofs_outer = fem.locate_dofs_topological(Y, fdim, outer_facets)
    dofs_cracktip = fem.locate_dofs_topological(Y, fdim, cracktip_facets)

    bcl = fem.dirichletbc(ScalarType(0), dofs_left, V.sub(0))
    bct = fem.dirichletbc(ScalarType(0), dofs_top, V.sub(1))
    bcb = fem.dirichletbc(ScalarType(0), dofs_bottom, V.sub(1))
    bcs = [bcl, bct, bcb]

    bct_z = fem.dirichletbc(ScalarType(1), dofs_outer, Y)
    bct_z2 = fem.dirichletbc(ScalarType(0), dofs_cracktip, Y)
    bcs_z = [bct_z, bct_z2]

    marked_facets = np.hstack([top_facets, bottom_facets, left_facets])
    marked_values = np.hstack(
        [
            np.full_like(top_facets, 1),
            np.full_like(bottom_facets, 2),
            np.full_like(left_facets, 3),
        ]
    )
    sorted_facets = np.argsort(marked_facets)
    facet_tag = mesh.meshtags(
        domain,
        domain.topology.dim - 1,
        marked_facets[sorted_facets],
        marked_values[sorted_facets],
    )

    metadata = {"quadrature_degree": 4}
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

    # Initializing the functions
    u.x.array[:] = 0.0

    z.x.array[:] = 1.0

    u_prev = fem.Function(V)
    u_prev.x.array[:] = u.x.array
    z_prev = fem.Function(Y)
    z_prev.x.array[:] = z.x.array

    y_dofs_top = fem.locate_dofs_topological(V.sub(1), fdim, top_facets)

    # Stored Energy, strain and stress functions in linear isotropic elasticity (plane stress)

    def energy(v):
        return (
            mu
            * (
                ufl.inner(ufl.sym(ufl.grad(v)), ufl.sym(ufl.grad(v)))
                + ((nu / (1 - nu)) ** 2) * (ufl.tr(ufl.sym(ufl.grad(v)))) ** 2
            )
            + 0.5
            * (lmbda)
            * (ufl.tr(ufl.sym(ufl.grad(v))) * (1 - 2 * nu) / (1 - nu)) ** 2
        )

    def epsilon(v):
        return ufl.sym(ufl.grad(v))

    def sigma(v):
        return 2.0 * mu * ufl.sym(ufl.grad(v)) + (lmbda) * ufl.tr(
            ufl.sym(ufl.grad(v))
        ) * (1 - 2 * nu) / (1 - nu) * ufl.Identity(len(v))

    def sigmavm(sig, v):
        return ufl.sqrt(
            1
            / 2
            * (
                ufl.inner(
                    sig - 1 / 3 * ufl.tr(sig) * ufl.Identity(len(v)),
                    sig - 1 / 3 * ufl.tr(sig) * ufl.Identity(len(v)),
                )
                + (1 / 9) * ufl.tr(sig) ** 2
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

    I1 = (z**2) * ufl.tr(sigma(u))
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

    # time-stepping parameters
    T = 1
    Totalsteps = 500
    startstepsize = 1 / Totalsteps
    stepsize = startstepsize
    t = stepsize
    step = 1
    rtol = 1e-9

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

    # Create nonlinear problems
    problem_u = NonlinearProblem(
        R,
        u,
        bcs=bcs,
        J=Jac,
        petsc_options=petsc_options_u,
        petsc_options_prefix="problem_u_",
    )

    problem_z = NonlinearProblem(
        R_z,
        z,
        bcs=bcs_z,
        J=Jac_z,
        petsc_options=petsc_options_z,
        petsc_options_prefix="problem_z_",
    )

    # Paraview output
    if comm_rank == 0:
        if not os.path.exists("paraview"):
            os.mkdir("paraview")

    file_results = io.XDMFFile(domain.comm, "paraview/2D_Ratio40checkeps2.xdmf", "w")
    file_results.write_mesh(domain)

    t1 = 0.0

    while t - stepsize < T:
        if comm_rank == 0:
            print("Step= %d" % step, "t= %f" % t, "Stepsize= %e" % stepsize)

        bct.g.value[...] = ScalarType(t * maxdisp)
        stag_iter = 1
        rnorm_stag = 1
        while stag_iter < 100 and rnorm_stag > 1e-7:
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

        Fx = domain.comm.allreduce(np.sum(fint[y_dofs_top]), op=MPI.SUM)
        z_x_val = evaluate_function(domain, z, (Rad + 2 * eps, 0.0))
        z_x = 1.0
        if z_x_val is not None:
            z_x = z_x_val

        if comm_rank == 0:
            print(Fx)
            print(z_x)
            with open("output.txt", "a") as rfile:
                rfile.write("%s %s %s %s\n" % (str(t), str(zmin), str(z_x), str(-Fx)))

        if z_x < 0.05 or np.isnan(zmin):
            t1 = t
            break

        # time stepping
        if step % 10 == 0:
            file_results.write_function(u, t)
            file_results.write_function(z, t)

        step += 1
        t += stepsize

    file_results.close()
    stretch_critical = t1 * maxdisp / L + 1
    if comm_rank == 0:
        with open("Critical_stretch.txt", "a") as rfile:
            rfile.write("Critical stretch= %s\n" % (str(stretch_critical)))
        print("Critical stretch= %f" % stretch_critical)


if __name__ == "__main__":
    main()
