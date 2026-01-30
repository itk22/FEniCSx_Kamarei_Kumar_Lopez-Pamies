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
import sys
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
    # ========== QUICK MODE FOR DEBUGGING ==========
    # Set QUICK_MODE=1 to run only first 1% of steps (5 steps instead of 500)
    QUICK_MODE = os.environ.get('QUICK_MODE', '0') == '1'

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

    # Mesh refinement iterations (reduce for faster testing: 1 iteration = ~16x fewer cells per loop)
    num_refinements_1 = 0  # First refinement loop (original: 2) - SET TO 0 FOR TESTING
    num_refinements_2 = 0  # Second refinement loop (original: 2) - SET TO 0 FOR TESTING

    delta = (1 + 3 * h / (8 * eps)) ** (-2) * (
        (sts + (1 + 2 * np.sqrt(3)) * shs) / ((8 + 3 * np.sqrt(3)) * shs)
    ) * 3 * Gc / (16 * Wts * eps) + (1 + 3 * h / (8 * eps)) ** (-1) * (2 / 5)
    comm = MPI.COMM_WORLD
    comm_rank = MPI.COMM_WORLD.rank
    log.set_log_level(log.LogLevel.ERROR)

    def log_rank0(msg, flush=True):
        """Log message only from rank 0"""
        if comm_rank == 0:
            print(msg, flush=flush)

    # Geometry of the single edge notch geometry
    Rad = 5  # notch length
    W, L = 15.0, 50.0  # making use of symmetry

    mesh_comm = MPI.COMM_WORLD
    model_rank = 0

    log_rank0("=" * 60)
    log_rank0("PHASE 1: Mesh Creation with Gmsh")
    log_rank0("=" * 60)

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
    log_rank0("Gmsh mesh generation complete")

    log_rank0("Broadcasting gmsh model to all ranks...")
    model = MPI.COMM_WORLD.bcast(gmsh.model, root=0)
    partitioner = dolfinx.cpp.mesh.create_cell_partitioner(
        dolfinx.mesh.GhostMode.shared_facet
    )
    mesh_data = io.gmsh.model_to_mesh(
        model, MPI.COMM_WORLD, 0, gdim=2, partitioner=partitioner
    )

    gmsh.finalize()
    log_rank0("Gmsh finalized, entering barrier...")
    MPI.COMM_WORLD.barrier()
    log_rank0("All ranks synchronized after gmsh finalization")
    domain = mesh_data[0]
    log_rank0(f"Initial mesh: {domain.topology.index_map(domain.topology.dim).size_global} cells")

    def cell_criterion(x):
        """Given mesh coordinates, return if each point
        satisfies x[1]<Rad

        :param x: Input coordinates, shape (num_points, 3)
        :returns: Boolean array of shape (num_points, )
        """
        return x[1] < Rad

    log_rank0("=" * 60)
    log_rank0(f"PHASE 2: First Refinement Loop (criterion: x[1] < Rad) - {num_refinements_1} iterations")
    log_rank0("=" * 60)
    ir = 0
    while ir < num_refinements_1:
        iter_start = time.time()
        log_rank0(f"Refinement loop 1, iteration {ir + 1}/{num_refinements_1}")

        log_rank0(f"  Creating topology connectivity...")
        domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)

        cells_local = np.arange(
            domain.topology.index_map(domain.topology.dim).size_local, dtype=np.int32
        )

        log_rank0(f"  Computing midpoints...")
        midpoints = dolfinx.mesh.compute_midpoints(
            domain, domain.topology.dim, cells_local
        ).T

        log_rank0(f"  Applying refinement criterion...")
        should_refine = np.flatnonzero(cell_criterion(midpoints)).astype(np.int32)
        total_to_refine = domain.comm.allreduce(len(should_refine), op=MPI.SUM)
        log_rank0(f"  Cells to refine (global): {total_to_refine}")

        log_rank0(f"  Creating edge entities...")
        domain.topology.create_entities(1)

        log_rank0(f"  Computing incident edges...")
        local_edges = dolfinx.mesh.compute_incident_entities(
            domain.topology, should_refine, domain.topology.dim, 1
        )

        log_rank0(f"  Refining mesh...")
        domain = dolfinx.mesh.refine(domain, local_edges)[0]

        num_cells_global = domain.topology.index_map(domain.topology.dim).size_global
        iter_time = time.time() - iter_start
        log_rank0(f"  Iteration {ir + 1} complete: {num_cells_global} cells (took {iter_time:.2f}s)")

        ir += 1

    log_rank0("First refinement loop complete")
    MPI.COMM_WORLD.barrier()
    log_rank0("All ranks synchronized after first refinement")

    def cell_criterion2(x):
        """Given mesh coordinates, return if each point
        satisfies (x[1]<2*eps)

        :param x: Input coordinates, shape (num_points, 3)
        :returns: Boolean array of shape (num_points, )
        """
        return x[1] < 2 * eps

    log_rank0("=" * 60)
    log_rank0(f"PHASE 3: Second Refinement Loop (criterion: x[1] < 2*eps) - {num_refinements_2} iterations")
    log_rank0("=" * 60)
    ir = 0
    while ir < num_refinements_2:
        iter_start = time.time()
        log_rank0(f"Refinement loop 2, iteration {ir + 1}/{num_refinements_2}")

        log_rank0(f"  Creating topology connectivity...")
        domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim)

        cells_local = np.arange(
            domain.topology.index_map(domain.topology.dim).size_local, dtype=np.int32
        )

        log_rank0(f"  Computing midpoints...")
        midpoints = dolfinx.mesh.compute_midpoints(
            domain, domain.topology.dim, cells_local
        ).T

        log_rank0(f"  Applying refinement criterion...")
        should_refine = np.flatnonzero(cell_criterion2(midpoints)).astype(np.int32)
        total_to_refine = domain.comm.allreduce(len(should_refine), op=MPI.SUM)
        log_rank0(f"  Cells to refine (global): {total_to_refine}")

        log_rank0(f"  Creating edge entities...")
        domain.topology.create_entities(1)

        log_rank0(f"  Computing incident edges...")
        local_edges = dolfinx.mesh.compute_incident_entities(
            domain.topology, should_refine, domain.topology.dim, 1
        )

        log_rank0(f"  Refining mesh...")
        domain = dolfinx.mesh.refine(domain, local_edges)[0]

        num_cells_global = domain.topology.index_map(domain.topology.dim).size_global
        iter_time = time.time() - iter_start
        log_rank0(f"  Iteration {ir + 1} complete: {num_cells_global} cells (took {iter_time:.2f}s)")

        ir += 1

    log_rank0("Second refinement loop complete")
    MPI.COMM_WORLD.barrier()
    log_rank0("All ranks synchronized after second refinement")

    # Ensure output directory exists
    log_rank0("=" * 60)
    log_rank0("PHASE 4: Writing Mesh Output")
    log_rank0("=" * 60)

    log_rank0("Creating output directory...")
    if comm_rank == 0:
        if not os.path.exists("out_gmsh"):
            os.makedirs("out_gmsh")

    MPI.COMM_WORLD.barrier()
    log_rank0("All ranks ready to write mesh file (collective)")

    # Use a single collective file for all ranks (modern FEniCSx approach)
    with io.XDMFFile(
        domain.comm, "out_gmsh/mesh.xdmf", "w"
    ) as file:
        file.write_mesh(domain)

    log_rank0("Mesh output written to out_gmsh/mesh.xdmf (collective file with all ranks)")

    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    log_rank0(f"Final mesh topology connectivity created")
    log_rank0(f"Final mesh statistics:")
    log_rank0(f"  - Global cells: {domain.topology.index_map(domain.topology.dim).size_global}")
    log_rank0(f"  - Global vertices: {domain.topology.index_map(0).size_global}")

    # Defining the function spaces
    log_rank0("=" * 60)
    log_rank0("PHASE 5: Function Space Setup")
    log_rank0("=" * 60)

    V = fem.functionspace(
        domain, ("CG", 1, (domain.geometry.dim,))
    )  # Function space for u
    Y = fem.functionspace(domain, ("CG", 1))  # Function space for z

    log_rank0(f"Displacement DOFs (global): {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")
    log_rank0(f"Phase field DOFs (global): {Y.dofmap.index_map.size_global}")

    # Quick mode: reduce displacement to 1% for debugging
    maxdisp = 0.03 * 0.01 if QUICK_MODE else 0.03
    if QUICK_MODE:
        log_rank0("*** QUICK MODE ENABLED: maxdisp reduced to 1% ***")

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
    # Quick mode: only 5 steps for debugging, normal: 500 steps
    Totalsteps = 5 if QUICK_MODE else 500
    startstepsize = 1 / Totalsteps
    stepsize = startstepsize
    if QUICK_MODE:
        log_rank0(f"*** QUICK MODE: Running only {Totalsteps} steps ***")
    t = stepsize
    step = 1
    rtol = 1e-9

    # PETSc solver options
    # Add monitoring to diagnose convergence issues
    petsc_options_u = {
        "snes_type": "newtonls",
        "snes_max_it": 10,
        "snes_atol": 1.0e-8,
        "snes_rtol": 1.0e-7,
        "snes_error_if_not_converged": False,
        "ksp_type": "preonly",
        "pc_type": "lu",
        # "snes_monitor": None,  # Show SNES convergence iterations
    }

    petsc_options_z = {
        "snes_type": "newtonls",
        "snes_max_it": 10,
        "snes_atol": 1.0e-8,
        "snes_rtol": 1.0e-7,
        "snes_error_if_not_converged": False,
        "ksp_type": "preonly",
        "pc_type": "lu",
        # "snes_monitor": None,  # Show SNES convergence iterations
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
    log_rank0("=" * 60)
    log_rank0("PHASE 6: Setting Up Paraview Output")
    log_rank0("=" * 60)

    if comm_rank == 0:
        if not os.path.exists("paraview"):
            os.mkdir("paraview")

    MPI.COMM_WORLD.barrier()

    file_results = io.XDMFFile(domain.comm, "paraview/2D_Ratio40checkeps2.xdmf", "w")
    file_results.write_mesh(domain)
    log_rank0("Paraview output file ready")

    log_rank0("=" * 60)
    log_rank0("PHASE 7: Time-Stepping Simulation")
    log_rank0("=" * 60)
    log_rank0("Starting simulation loop...")

    t1 = 0.0

    while t - stepsize < T:
        step_start = time.time()
        log_rank0("-" * 60)
        log_rank0(f"Step {step}/{Totalsteps} | t={t:.6f} | dt={stepsize:.3e} | Progress: {100*t/T:.1f}%")
        log_rank0("-" * 60)

        bct.g.value[...] = ScalarType(t * maxdisp)

        # Staggered iteration following Algorithm 3 from the paper
        k = 1  # Stagger iteration counter (Algorithm line 6)
        R_stag_u = 1e-5  # Relative tolerance for displacement
        R_stag_v = 1e-5  # Relative tolerance for phase field
        N_stag = 100  # Maximum stagger iterations

        log_rank0("  Starting staggered iterations (Algorithm 3)...")

        while k <= N_stag:
            iter_start = time.time()

            ##############################################################
            # Line 8: Solve for displacement field
            ##############################################################
            t_u_start = time.time()
            problem_u.solve()
            t_u = time.time() - t_u_start
            u.x.scatter_forward()

            ##############################################################
            # Lines 9-21: Solve for phase field
            ##############################################################
            t_z_start = time.time()
            problem_z.solve()
            t_z = time.time() - t_z_start
            z.x.scatter_forward()
            ##############################################################

            zmin = domain.comm.allreduce(np.min(z.x.array), op=MPI.MIN)

            ###############################################################
            # Lines 22-29: Convergence check
            ###############################################################
            b_e = fem.petsc.assemble_vector(fem.form(-R))
            fint = b_e.copy()
            fem.petsc.set_bc(b_e, bcs)

            if k == 1:
                # Line 23: Compute absolute 2-norms on first iteration
                u_norm = la.norm(u.x)
                v_norm = la.norm(z.x)
                # Line 24: Initialize relative norms to 1
                r_u = 1.0
                r_v = 1.0
            else:
                # Lines 26-27: Compute changes and relative norms (use arrays for subtraction)
                du_norm = np.linalg.norm(u.x.array - u_prev.x.array)
                dv_norm = np.linalg.norm(z.x.array - z_prev.x.array)
                r_u = du_norm / u_norm if u_norm > 1e-16 else 0.0
                r_v = dv_norm / v_norm if v_norm > 1e-16 else 0.0

            iter_total = time.time() - iter_start
            log_rank0(f"    Stag iter {k}: r_u={r_u:.3e}, r_v={r_v:.3e}, zmin={zmin:.6f} | u:{t_u:.2f}s, z:{t_z:.2f}s")

            # Line 29: Check convergence
            if r_u <= R_stag_u and r_v <= R_stag_v:
                log_rank0(f"    Converged in {k} iterations: r_u={r_u:.3e}, r_v={r_v:.3e}")
                break

            # Line 34: Update previous solution
            u_prev.x.array[:] = u.x.array
            z_prev.x.array[:] = z.x.array

            k += 1

        ########### Post-processing ##############
        u_prev.x.array[:] = u.x.array
        z_prev.x.array[:] = z.x.array

        # Calculate Reaction
        Fx = domain.comm.allreduce(np.sum(fint[y_dofs_top]), op=MPI.SUM)
        z_x_val = evaluate_function(domain, z, (Rad + 2 * eps, 0.0))
        z_x = 1.0
        if z_x_val is not None:
            # Extract scalar - evaluate_function returns array, take first element
            if isinstance(z_x_val, np.ndarray):
                z_x = float(z_x_val.ravel()[0])
            else:
                z_x = float(z_x_val)

        log_rank0(f"  Results: Fx={-Fx:.6e}, z_x={z_x:.6f}, zmin={zmin:.6f}")

        if comm_rank == 0:
            with open("output.txt", "a") as rfile:
                rfile.write(f"{t} {zmin} {z_x} {-Fx}\n")

        # Check termination criteria
        if z_x < 0.05 or np.isnan(zmin):
            log_rank0(f"  *** Termination criterion met: z_x={z_x:.6f}, zmin={zmin:.6f} ***")
            t1 = t
            break

        # Write output every 10 steps
        if step % 10 == 0:
            log_rank0("  Writing output to paraview file...")
            file_results.write_function(u, t)
            file_results.write_function(z, t)

        step_time = time.time() - step_start
        log_rank0(f"  Step complete in {step_time:.2f}s")

        step += 1
        t += stepsize

    file_results.close()
    log_rank0("Paraview output file closed")
    log_rank0("=" * 60)
    log_rank0("Simulation Complete")
    log_rank0("=" * 60)

    stretch_critical = t1 * maxdisp / L + 1
    log_rank0(f"Critical stretch: {stretch_critical:.6f}")

    if comm_rank == 0:
        with open("Critical_stretch.txt", "a") as rfile:
            rfile.write(f"Critical stretch= {stretch_critical:.6f}\n")


if __name__ == "__main__":
    main()
