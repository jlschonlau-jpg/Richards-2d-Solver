import os
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import argparse
from classes import VanGenuchtenSoil, Grid2D

# ==========================================
# 1. 2D FLUX AND RESIDUAL MODULE
# ==========================================
def get_k_face_2d(K_2d, h_2d, axis, mean_type, dz=1.0):
    """
    axis=0 is Vertical (Top/Bottom), axis=1 is Horizontal (Left/Right)
    """
    if axis == 1: 
        K_1, K_2 = K_2d[:, :-1], K_2d[:, 1:]
        h_1, h_2 = h_2d[:, :-1], h_2d[:, 1:]
        
        if mean_type == 'upwind':
            # Horizontal: No gravity difference, compare h directly
            return np.where(h_1 >= h_2, K_1, K_2)

    else:         
        K_1, K_2 = K_2d[:-1, :], K_2d[1:, :]
        h_1, h_2 = h_2d[:-1, :], h_2d[1:, :]
        
        if mean_type == 'upwind':
            # Vertical: Compare Total Head (H). 
            # Top cell (1) has gravity advantage of +dz over Bottom cell (2)
            return np.where((h_1 + dz) >= h_2, K_1, K_2)

    # Fallback for other averaging methods
    if mean_type == 'arithmetic':
        return 0.5 * (K_1 + K_2)
    elif mean_type == 'geometric':
        return np.sqrt(K_1 * K_2 + 1e-20)
    elif mean_type == 'harmonic':
        return (2 * K_1 * K_2) / (K_1 + K_2 + 1e-20)


def calculate_2d_residuals(h_flat, h_old_flat, theta_old_flat, soil, grid, dt, mean_type):
    # Reshape
    h = h_flat.reshape((grid.Nz, grid.Nx))
    theta = soil.calc_theta(h)
    K = soil.calc_K(h)
    
    # Pass grid.dz into the vertical face calculator
    K_face_x = get_k_face_2d(K, h, axis=1, mean_type=mean_type)
    K_face_z = get_k_face_2d(K, h, axis=0, mean_type=mean_type, dz=grid.dz)
    
    # Internal Fluxes
    flux_x = -K_face_x * ((h[:, 1:] - h[:, :-1]) / grid.dx) 
    flux_z = -K_face_z * ((h[1:, :] - h[:-1, :]) / grid.dz - 1.0)
    
    # Net Flow
    net_flow = np.zeros((grid.Nz, grid.Nx))
    net_flow[:, 1:]  += flux_x / grid.dx  
    net_flow[:, :-1] -= flux_x / grid.dx  
    net_flow[1:, :]  += flux_z / grid.dz  
    net_flow[:-1, :] -= flux_z / grid.dz  
    
    # Residual
    theta_old = theta_old_flat.reshape((grid.Nz, grid.Nx))
    residual_2d = (theta - theta_old) / dt - net_flow
    
    return residual_2d.flatten(), K_face_x, K_face_z

# ==========================================
# 2. SPARSE MATRIX ASSEMBLER
# ==========================================

def get_diagonals(grid, Tx, Tz, diag_main):
    N = grid.N_total
    
    # Offsets (+1 = Right, -1 = Left, +Nx = Bottom, -Nx = Top)
    # The reshape trick here perfectly leaves zeros at the wrap-around boundaries.
    diag_right = np.zeros(N); diag_right.reshape((grid.Nz, grid.Nx))[:, :-1] = -Tx
    diag_left = np.zeros(N);  diag_left.reshape((grid.Nz, grid.Nx))[:, 1:] = -Tx
    diag_bottom = np.zeros(N); diag_bottom.reshape((grid.Nz, grid.Nx))[:-1, :] = -Tz
    diag_top = np.zeros(N);    diag_top.reshape((grid.Nz, grid.Nx))[1:, :] = -Tz
    
    # Main Diagonal = Storage + sum of absolute values of connections
    diag_main += (np.abs(diag_right) + np.abs(diag_left) + 
                  np.abs(diag_bottom) + np.abs(diag_top))
    
    # Sliced perfectly for SciPy offsets
    return diag_right[:-1], diag_left[1:], diag_bottom[:-grid.Nx], diag_top[grid.Nx:], diag_main

# ==========================================
# 3. BOUNDARY CONDITIONS
# ==========================================

def apply_boundary_conditions(grid, diag_main, residual_flat, top_flux, bottom_head, h_flat):
    boundaries = grid.get_boundary_indices()
    
    # Top Boundary: Neumann (Flux)
    # Top Boundary: Neumann (Patch/Point Source)
    top_idx = boundaries['top']

    # Define how wide the "dripper" is (e.g., 5cm wide in the center)
    center_node = grid.Nx // 2
    half_width = 2 # This makes a 5-node wide plume (center +/- 2)
    patch_indices = top_idx[center_node - half_width : center_node + half_width + 1]

    # Apply flux ONLY to the patch
    residual_flat[patch_indices] -= (top_flux / grid.dz)
    
    # Bottom Boundary: Dirichlet (Fixed Head)
    bottom_idx = boundaries['bottom']
    
    # Robust Penalty Method ("Big Spring")
    # This forces A*dh = -residual to yield: dh = bottom_head - h_flat
    penalty = 1e12
    diag_main[bottom_idx] += penalty 
    residual_flat[bottom_idx] -= penalty * (bottom_head - h_flat[bottom_idx])
    
    return diag_main, residual_flat

# ==========================================
# 4. SOLVER LOOP
# ==========================================
def perform_timestep_2d(h_old_flat, theta_old_flat, soil, grid, dt, top_flux, bottom_head, mean_type):
    h_iter_flat = np.copy(h_old_flat)
    
    # We need the actual theta from the start of the timestep
    theta_n_flat = np.copy(theta_old_flat) 
    
    for iteration in range(50): 
        # 1. Calculate current state theta and C
        theta_m_flat = soil.calc_theta(h_iter_flat)
        C_flat = soil.calc_C(h_iter_flat)
        
        # 2. Get Residuals (This is the 'Mass Balance' part)
        # R = (theta_current - theta_old)/dt - NetFlow
        residual_flat, Kx, Kz = calculate_2d_residuals(
            h_iter_flat, h_old_flat, theta_old_flat, soil, grid, dt, mean_type
        )
        
        # 3. Build Matrix LHS (This stays essentially the same)
        Tx = Kx / (grid.dx**2)
        Tz = Kz / (grid.dz**2)
        diag_main = C_flat / dt
        
        diag_right, diag_left, diag_bottom, diag_top, diag_main = get_diagonals(grid, Tx, Tz, diag_main)
        
        # 4. Apply Boundaries
        diag_main, residual_flat = apply_boundary_conditions(
            grid, diag_main, residual_flat, top_flux, bottom_head, h_iter_flat
        )
        
        A = sp.diags(
            diagonals = [diag_main, diag_right, diag_left, diag_bottom, diag_top],
            offsets = [0, 1, -1, grid.Nx, -grid.Nx],
            format = 'csr'
        )
        
        # 5. Solve for dh
        try:
            dh_flat = spsolve(A, -residual_flat)
        except:
            return h_old_flat, theta_old_flat, 0, False

        # 6. Update h
        h_iter_flat += dh_flat 
        
        # 7. Convergence Check
        if np.max(np.abs(dh_flat)) < 1e-5: 
            theta_new_flat = soil.calc_theta(h_iter_flat)
            return h_iter_flat, theta_new_flat, iteration + 1, True
            
    return h_old_flat, theta_old_flat, iteration, False
# ==========================================
# 5. ORCHESTRATION & PLOTTING
# ==========================================

def plot_2d_moisture(theta_flat, grid):
    theta_2d = theta_flat.reshape((grid.Nz, grid.Nx))
    plt.figure(figsize=(8, 6))
    plt.imshow(theta_2d, extent=[0, grid.Lx, -grid.Lz, 0], cmap='Blues', aspect='auto')
    plt.colorbar(label='Water Content (θ)')
    plt.xlabel('Width (cm)')
    plt.ylabel('Depth (cm)')
    plt.title("2D Moisture Profile")
    plt.show()

def run_simulation():
    parser = argparse.ArgumentParser(description="2D Richards Equation Solver")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "params.json")

    parser.add_argument(
        '--config', '-c', 
        type=str, 
        default=default_config,
        help=f"Path to the JSON configuration file (default: {default_config})"
    )
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            cfg = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        return

    soil = VanGenuchtenSoil(**cfg['soil_properties'])
    grid = Grid2D(
        Lx=cfg['grid_setup']['width'], 
        Lz=cfg['grid_setup']['total_depth'], 
        dx=cfg['grid_setup']['node_spacing_x'], 
        dz=cfg['grid_setup']['node_spacing_z']
    )

    mode = cfg['solver_settings'].get('averaging_mode', 'harmonic').lower()
    
    t = 0.0
    dt = cfg['solver_settings']['initial_dt']
    
    # Initialize state
    h_state = np.full(grid.N_total, cfg['conditions']['initial_head'])
    boundaries = grid.get_boundary_indices()
    h_state[boundaries['bottom']] = cfg['conditions']['bottom_head']
    
    theta_state = soil.calc_theta(h_state)
    start_wall = time.time()

    print(f"Starting 2D Simulation ({grid.Nx}x{grid.Nz} nodes)...")
    print(mode)
    step_count = 0
    while t < cfg['solver_settings']['max_time']:
        h_new, th_new, iters, success = perform_timestep_2d(
            h_state, theta_state, soil, grid, dt, 
            cfg['conditions']['top_flux'], cfg['conditions']['bottom_head'], mode
        )
        
        if success:
            h_state, theta_state = h_new, th_new
            t += dt
            
            # --- PI CONTROLLER START ---
            target_iters = 6
            # We use a power of 0.5 to smooth the change
            # max(1, iters) prevents division by zero
            factor = (target_iters / max(1, iters))**0.5
            
            # Bound the factor so dt doesn't explode or vanish too fast
            factor = max(0.5, min(2.0, factor))
            
            dt = dt * factor
            
            # Hard limits for safety
            dt = min(dt, 0.5)  # Max dt of 30 mins
            dt = max(dt, 1e-6) # Min dt
            # --- PI CONTROLLER END ---

            if step_count % 35 == 0: 
                print(f"Time: {t:.2f} | dt: {dt:.4f} | Iters: {iters}")
            step_count += 1
        else:
            dt *= 0.2
            if dt < 1e-7:
                print("Simulation Crashed: Time step too small. Check conditions or grid.")
                break

    print(f"Simulation Complete. Runtime: {time.time() - start_wall:.2f}s")
    plot_2d_moisture(theta_state, grid)

if __name__ == "__main__":
    run_simulation()