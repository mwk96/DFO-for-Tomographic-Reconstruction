import os
import numpy as np
import imageio
import time
from PIL import Image
import astra
from matplotlib import pyplot as plt
import sys


# Parameters
size = 64
alpha = 6
phantomNo = 5
maxIterations = 10
maxFE = 100000

POP_SIZE = 2

# TV Parameter (from paper)
TV_MU = 95.0  # Optimal TV parameter from paper

# Load phantom
filename = f'input/Phantom_0{phantomNo}_{size}x{size}.bmp'
img = Image.open(filename).convert("L")
phantom = np.array(img).astype(np.uint8)
imageio.imwrite('phantom.png', phantom)

# Convert to float for ASTRA processing
phantom = phantom.astype(np.float32)

# Create projection geometry
vol_geom = astra.create_vol_geom(size, size)
angles = np.linspace(0, np.pi, alpha, endpoint=False)
proj_geom_recon = astra.create_proj_geom('parallel', 1., size, angles)
projector_id = astra.create_projector('linear', proj_geom_recon, vol_geom)

# Create sinogram for reconstruction
sinogram_id_recon, sinogram_recon = astra.create_sino(phantom, projector_id)
sinogram_recon = astra.data2d.get(sinogram_id_recon)
sinogram_recon = sinogram_recon.astype(np.float32)

# Forward project phantom for reconstruction reference
refSino = np.zeros(sinogram_recon.shape, dtype=np.float32)
W_recon = astra.OpTomo(projector_id)
W_recon.FP(phantom, out=refSino)

# Initialize boolean table (same shape as phantom)
master_boolean_table = np.ones((size, size), dtype=bool)

# Function to compute intersected pixels
def get_intersected_pixels_with_astra(size, alpha, num_detectors, projector_id):
    num_pixels = size * size
    identity_image = np.eye(num_pixels, dtype=np.float32).reshape((num_pixels, size, size))
    sinograms = np.zeros((alpha, num_detectors, num_pixels), dtype=np.float32)

    for pixel_idx in range(num_pixels):
        sinogram_id, sinogram = astra.create_sino(identity_image[pixel_idx], projector_id)
        sinograms[:, :, pixel_idx] = sinogram
        astra.data2d.delete(sinogram_id)

    intersected_pixels = {}
    for angle_idx in range(alpha):
        intersected_pixels[angle_idx] = {}
        for detector_idx in range(num_detectors):
            active_pixels = np.nonzero(sinograms[angle_idx, detector_idx, :])[0]
            intersected_pixels[angle_idx][detector_idx] = [
                (pixel_idx // size, pixel_idx % size) for pixel_idx in active_pixels
            ]
    return intersected_pixels

# Update boolean table based on the phantom
def update_boolean_table(master_boolean_table, intersected_pixels, phantom, specified_angle_indices):
    for angle_index in specified_angle_indices:
        for detector_idx, pixels in intersected_pixels[angle_index].items():
            manual_sum = sum(phantom[row, col] for row, col in pixels)
            if manual_sum == 0:
                for row, col in pixels:
                    master_boolean_table[row, col] = False

# Compute intersected pixels using ASTRA projection
num_detectors = size
intersected_pixels = get_intersected_pixels_with_astra(size, alpha, num_detectors, projector_id)

# Define the angles used for masking
specified_angle_radians = [0, 0.52359878, 1.04719755, 1.57079633, 2.0943951, 2.61799388]
specified_angle_index = [np.argmin(np.abs(angles - angle)) for angle in specified_angle_radians]

# Update Boolean Table Using the Same Reference Code Logic
update_boolean_table(master_boolean_table, intersected_pixels, phantom, specified_angle_index)

# Save the boolean table visualization for reference
output_folder = 'RARARA1'
os.makedirs(output_folder, exist_ok=True)

plt.figure(figsize=(6, 6))
plt.imshow(master_boolean_table, cmap='gray')
plt.title('Master Boolean Table')
plt.colorbar(label="True (White) / False (Black)")
boolean_table_path = os.path.join(
    output_folder,
    f'master_boolean_table_alpha{alpha:.2f}_phantom{phantomNo}.png'
)
plt.savefig(boolean_table_path)
plt.close()

print(f"Master Boolean Table saved as '{boolean_table_path}'")

# Count valid pixels
valid_pixel_count = np.sum(master_boolean_table)
total_pixels = master_boolean_table.size
valid_pixel_percentage = (valid_pixel_count / total_pixels) * 100
print(f"Valid pixels in the boolean table: {valid_pixel_count} ({valid_pixel_percentage:.2f}%)")

# Define forward projection operator for ASTRA
Astra_A = lambda x: astra.data2d.get(astra.create_sino(x.reshape(size, size), projector_id)[0]).flatten()

# Define backprojection operator for ASTRA
def Astra_AT(y, proj_geom):
    backprojection_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('BP')
    cfg['ReconstructionDataId'] = backprojection_id
    cfg['ProjectionDataId'] = astra.data2d.create('-sino', proj_geom, y.reshape(alpha, size))
    cfg['ProjectorId'] = projector_id
    algorithm_id = astra.algorithm.create(cfg)
    astra.algorithm.run(algorithm_id)
    backprojection = astra.data2d.get(backprojection_id)
    astra.algorithm.delete(algorithm_id)
    astra.data2d.delete(backprojection_id)
    return backprojection.flatten()

# SIRT Reconstruction function with iterative boolean mask application and residual tracking
def sirt(A, AT, b, x0, max_iter=20, tol=1e-6, boolean_table=None, proj_geom=None, relaxation=1.0):
    x = x0.copy()  # Initialize reconstruction
    
    # Precompute the row sums and column sums for SIRT
    ones_vol = np.ones((size, size), dtype=np.float32)
    ones_sino = A(ones_vol)
    ones_sino[ones_sino < 1e-6] = 1.0  # Avoid division by zero
    C_inv = 1.0 / ones_sino  # Inverse of row sums (for projection normalization)
    
    R_inv = AT(np.ones_like(ones_sino), proj_geom)
    R_inv = R_inv.reshape((size, size))
    R_inv[R_inv < 1e-6] = 1.0  # Avoid division by zero
    R_inv = 1.0 / R_inv  # Inverse of column sums (for backprojection normalization)
    R_inv = R_inv.flatten()
    
    # Initialize array to store residual norms at each iteration
    residual_norms = []
    
    for i in range(max_iter):
        # Forward project current estimate
        Ax = A(x)
        
        # Compute residual and normalize by row sums
        residual = b - Ax
        residual_norm = np.linalg.norm(residual)
        residual_norms.append(residual_norm)
        
        normalized_residual = residual * C_inv
        
        # Backproject normalized residual
        update = AT(normalized_residual, proj_geom)
        
        # Normalize by column sums and apply relaxation factor
        normalized_update = update * R_inv * relaxation
        
        # Update reconstruction
        x += normalized_update
        
        # Apply boolean mask, ensuring the same behavior as the reference code
        x = x.reshape((size, size))
        x[~boolean_table] = 0  # Blacken out masked regions
        x = x.flatten()  # Flatten back for the next iteration
        
        # Check for convergence using L2 norm of the residual
        if residual_norm < tol:
            break
            
        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"SIRT Iteration {i}, Residual: {residual_norm:.6f}")
    
    # Plot convergence curve
    plt.figure(figsize=(10, 6))
    plt.semilogy(residual_norms)
    plt.title('SIRT Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm (log scale)')
    plt.grid(True)
    convergence_path = os.path.join(output_folder, f'sirt_convergence_ph0{phantomNo}_{size}x{size}x{alpha}.png')
    plt.savefig(convergence_path)
    plt.close()
    print(f"SIRT convergence curve saved to: {convergence_path}")
    
    # Print final residual value
    print(f"Final SIRT residual after {i+1} iterations: {residual_norms[-1]:.6f}")
    
    return x.reshape((size, size)), residual_norms  # Return both the solution and the convergence data


# Total Variation computation function
def compute_total_variation(x):
    """
    Compute Total Variation of image x (2D array)
    TV(x) = Sum_ij sqrt(|x_i+1,j - x_i,j|^2 + |x_i,j+1 - x_i,j|^2)
    """
    # Handle boundary conditions by padding
    x_padded = np.pad(x, ((0, 1), (0, 1)), mode='edge')
    
    # Compute gradients
    dx = x_padded[1:, :-1] - x_padded[:-1, :-1]  # Horizontal gradient
    dy = x_padded[:-1, 1:] - x_padded[:-1, :-1]  # Vertical gradient
    
    # Compute TV as sum of gradient magnitudes
    tv = np.sum(np.sqrt(dx**2 + dy**2))
    return tv


# DFO evaluation function with TV regularization
def eval_with_tv(x, refSino, W, boolean_table, mu=TV_MU):
    """
    Evaluation function that includes TV regularization
    Fitness = e1 + mu * TV(x)
    """
    # Reshape to 2D for TV computation
    x_2d = x.reshape((size, size))
    
    # Compute reconstruction error (e1)
    xSino = np.zeros_like(refSino)
    W.FP(x_2d, out=xSino)
    xFlatSino = xSino.flatten()
    e1 = np.sum((refSino.flatten() - xFlatSino) ** 2)
    
    # Compute Total Variation
    tv = compute_total_variation(x_2d)
    
    # Combined fitness with TV regularization
    fitness = e1 + mu * tv
    
    return fitness, e1, tv


def dfo_refine_with_tv(initial_solution, boolean_table, refSino, W, maxFE=500000, N=2, delta=0.001, mu=TV_MU):
    """
    DFO refinement with TV regularization and simple 0-255 bounds.
    No SSE - just standard optimization bounds.
    """
    D = size * size  # Size of the image flattened
    X = np.zeros([N, D], dtype=np.float32)  # Population matrix
    
    # Fixed bounds for all dimensions
    LOWER_BOUND = 0.0
    UPPER_BOUND = 255.0
    
    # Initialize first fly with SIRT solution
    X[0] = initial_solution.flatten()
    
    # Second fly randomly initialized within bounds
    X[1] = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=D)

    # Apply boolean mask to both flies
    X[0][~boolean_table.flatten()] = 0
    X[1][~boolean_table.flatten()] = 0
    
    # Initialize remaining flies if N > 2
    for i in range(2, N):
        X[i] = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=D)
        X[i][~boolean_table.flatten()] = 0
    
    fitness = [None] * N
    e1_values = [None] * N
    tv_values = [None] * N
    FE = 0
    
    # Storage for tracking progress
    fitness_history = []
    e1_history = []
    e2_history = []
    tv_history = []
    
    # Evaluate initial fitness for each fly
    print("DEBUG: Starting initial fitness evaluation...")
    for i in range(N):
        fitness[i], e1_values[i], tv_values[i] = eval_with_tv(X[i], refSino, W, boolean_table, mu)
        FE += 1
        print(f"DEBUG: Fly {i} initial fitness: {fitness[i]:.2f}")
    
    # Find the best solution
    s = np.argmin(fitness)
    
    print(f"DFO with TV initialized:")
    print(f"   Bounds: [{LOWER_BOUND}, {UPPER_BOUND}]")
    print(f"   Initial best fitness: {fitness[s]:.2f} (e1: {e1_values[s]:.2f}, TV: {tv_values[s]:.2f})")
    
    # Iterative optimization
    progress_check_interval = maxFE // 20  # Check progress every 5%
    last_progress_check = 0
    
    print("DEBUG: Starting main optimization loop...")
    iteration_count = 0
    
    while FE < maxFE:
        iteration_count += 1
        
        # Debug output for first few iterations
        if iteration_count <= 5:
            print(f"DEBUG: Main loop iteration {iteration_count}, FE={FE}")
        
        if fitness[s] == 0:
            break  # Perfect solution found
        
        for i in range(N):
            if i == s:
                continue  # Skip best solution (elitist strategy)
                
            # Find best neighbor
            left = (i - 1) % N
            right = (i + 1) % N
            bNeighbour = right if fitness[right] < fitness[left] else left
            
            # Apply disturbances randomly within fixed bounds
            rand_vals = np.random.rand(D)
            disturbance_mask = rand_vals < delta
            X[i, disturbance_mask] = np.random.uniform(
                low=LOWER_BOUND, high=UPPER_BOUND, size=np.sum(disturbance_mask)
            )
            
            # Update equation for non-disturbed values
            non_disturbance_mask = ~disturbance_mask
            u_vals = np.random.rand(D)[non_disturbance_mask]
            X[i, non_disturbance_mask] = X[bNeighbour, non_disturbance_mask] + u_vals * (
                    X[s, non_disturbance_mask] - X[i, non_disturbance_mask])
            
            # Simple clamping to fixed bounds
            X[i] = np.clip(X[i], LOWER_BOUND, UPPER_BOUND)
            
            # Apply boolean mask
            shaped_x = X[i].reshape((size, size))
            shaped_x[~boolean_table] = 0
            X[i] = shaped_x.flatten()
            
            # Evaluate fitness with TV regularization
            fitness[i], e1_values[i], tv_values[i] = eval_with_tv(X[i], refSino, W, boolean_table, mu)
            FE += 1
            
            # Update best solution if needed
            if fitness[i] < fitness[s]:
                s = i
                print(f"NEW BEST: FE={FE}, fitness={fitness[s]:.2f}")
        
        # Progress updates
        if FE % 1000 == 0:
            print(f"  Progress: FE {FE}/{maxFE} ({FE/maxFE*100:.1f}%), Best fitness: {fitness[s]:.2f}")
        
        # Progress tracking
        if FE - last_progress_check >= progress_check_interval:
            # Compute current e2 for monitoring
            current_best_2d = X[s].reshape((size, size))
            current_e2 = np.sum(np.abs(phantom.flatten() - current_best_2d.flatten()))
            
            fitness_history.append(fitness[s])
            e1_history.append(e1_values[s])
            e2_history.append(current_e2)
            tv_history.append(tv_values[s])
            
            print(f"FE {FE}/{maxFE} ({FE/maxFE*100:.1f}%): "
                  f"fitness={fitness[s]:.2f}, e1={e1_values[s]:.2f}, "
                  f"e2={current_e2:.2f}, TV={tv_values[s]:.2f}")
            last_progress_check = FE
    
    print(f"DEBUG: Main loop completed after {iteration_count} iterations")
    
    # Final solution
    best_solution_2d = X[s].reshape((size, size))
    
    # Plot convergence curves
    if fitness_history:
        plot_convergence(fitness_history, e1_history, e2_history, tv_history)
    else:
        print("Warning: No convergence data to plot")
    
    print(f"DFO with TV completed:")
    print(f"   Final fitness: {fitness[s]:.2f} (e1: {e1_values[s]:.2f}, TV: {tv_values[s]:.2f})")
    
    return best_solution_2d


def plot_convergence(fitness_hist, e1_hist, e2_hist, tv_hist):
    """
    Plot convergence curves for DFO with TV.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    iterations = range(len(fitness_hist))
    
    # Plot 1: Combined fitness
    ax1.semilogy(iterations, fitness_hist, 'b-', linewidth=2)
    ax1.set_title('Combined Fitness (e1 + mu*TV)')
    ax1.set_xlabel('Progress Check')
    ax1.set_ylabel('Fitness (log scale)')
    ax1.grid(True)
    
    # Plot 2: e1 and e2 errors
    ax2.semilogy(iterations, e1_hist, 'r-', label='e1 (reconstruction)', linewidth=2)
    ax2.semilogy(iterations, e2_hist, 'g-', label='e2 (reproduction)', linewidth=2)
    ax2.set_title('Reconstruction vs Reproduction Errors')
    ax2.set_xlabel('Progress Check')
    ax2.set_ylabel('Error (log scale)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Total Variation
    ax3.plot(iterations, tv_hist, 'm-', linewidth=2)
    ax3.set_title('Total Variation (Smoothness)')
    ax3.set_xlabel('Progress Check')
    ax3.set_ylabel('TV Value')
    ax3.grid(True)
    
    # Plot 4: Empty or additional info
    ax4.text(0.5, 0.5, f'DFO with TV Regularization\nmu = {TV_MU}\nBounds: [0, 255]', 
             ha='center', va='center', fontsize=14, transform=ax4.transAxes)
    ax4.set_title('Configuration')
    ax4.axis('off')
    
    plt.tight_layout()
    convergence_path = os.path.join(output_folder, f'dfo_tv_convergence_ph0{phantomNo}_{size}x{size}x{alpha}.png')
    plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"DFO+TV convergence curves saved to: {convergence_path}")


# Original DFO evaluation function for comparison
def eval(x, refSino, W, FE):
    xSino = np.zeros_like(refSino)
    W.FP(x.reshape((size, size)), out=xSino)
    xFlatSino = xSino.flatten()
    fitness = np.sum((refSino.flatten() - xFlatSino) ** 2)  # Sum of squared differences
    FE += 1
    return fitness, FE


# Calculate final errors
def compute_final_errors(reconstruction, refSino, W, phantom):
    # Compute e1 by forward projecting the final reconstruction
    xSino = np.zeros_like(refSino)
    W.FP(reconstruction, out=xSino)
    e1 = np.sum(np.abs(refSino.flatten() - xSino.flatten()))
    
    # Compute e2 as pixel-wise difference from ground truth
    e2 = np.sum(np.abs(phantom.flatten() - reconstruction.flatten()))
    
    return e1, e2


# Start timing the full pipeline
start_time = time.time()

# Step 1: Run SIRT with boolean masking
print("Starting SIRT reconstruction...")
sirt_start_time = time.time()
x0 = np.zeros(size * size, dtype=np.float32)

# Run SIRT with 20 iterations
reconstruction_sirt, residual_norms = sirt(Astra_A, Astra_AT, sinogram_recon.flatten(), x0, 
                                           max_iter=20, tol=1e-4, boolean_table=master_boolean_table, 
                                           proj_geom=proj_geom_recon, relaxation=1.0)

sirt_time = time.time() - sirt_start_time
print(f"SIRT completed in {sirt_time:.2f} seconds")

# Compute SIRT errors
sirt_e1, sirt_e2 = compute_final_errors(reconstruction_sirt, refSino, W_recon, phantom)
sirt_tv = compute_total_variation(reconstruction_sirt)
print(f"SIRT errors - e1: {sirt_e1:.2f} | e2: {sirt_e2:.2f} | TV: {sirt_tv:.2f}")

# Save intermediate SIRT result
minV, maxV = np.min(reconstruction_sirt), np.max(reconstruction_sirt)
normalized_sirt = ((reconstruction_sirt - minV) / (maxV - minV) * 255).astype(np.uint8)
sirt_output_path = os.path.join(output_folder, f'reconstruction_SIRT_ph0{phantomNo}_{size}x{size}x{alpha}.bmp')
imageio.imwrite(sirt_output_path, normalized_sirt)

# Step 2: Run DFO refinement with TV (no SSE) using the SIRT solution
print("\n" + "="*60)
print("Starting DFO refinement with TV regularization (no SSE)...")
print("="*60)

dfo_tv_start_time = time.time()

reconstruction_dfo_tv = dfo_refine_with_tv(
    reconstruction_sirt, 
    master_boolean_table, 
    refSino.copy(), 
    W_recon,
    maxFE=maxFE-20,
    N=POP_SIZE,
    mu=TV_MU
)

dfo_tv_time = time.time() - dfo_tv_start_time
print(f"DFO with TV completed in {dfo_tv_time:.2f} seconds")

dfo_tv_e1, dfo_tv_e2 = compute_final_errors(reconstruction_dfo_tv, refSino, W_recon, phantom)
dfo_tv_tv = compute_total_variation(reconstruction_dfo_tv)
print(f"DFO+TV errors - e1: {dfo_tv_e1:.2f} | e2: {dfo_tv_e2:.2f} | TV: {dfo_tv_tv:.2f}")

# Save final result
minV, maxV = np.min(reconstruction_dfo_tv), np.max(reconstruction_dfo_tv)
normalized_dfo_tv = ((reconstruction_dfo_tv - minV) / (maxV - minV) * 255).astype(np.uint8)
normalized_dfo_tv[~master_boolean_table] = 0  # Final mask application
dfo_tv_output_path = os.path.join(output_folder, f'reconstruction_SIRT_DFO_TV_ph0{phantomNo}_{size}x{size}x{alpha}.bmp')
imageio.imwrite(dfo_tv_output_path, normalized_dfo_tv)
print(f"Final SIRT+DFO+TV reconstruction saved to: {dfo_tv_output_path}")

# Total pipeline time
total_time = time.time() - start_time

# Summary
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"SIRT errors - e1: {sirt_e1:.2f} | e2: {sirt_e2:.2f} | TV: {sirt_tv:.2f}")
print(f"DFO + TV:    e1={dfo_tv_e1:.2f} | e2={dfo_tv_e2:.2f} | TV: {dfo_tv_tv:.2f}")
print(f"Total time: {total_time:.2f} seconds")
print("="*60)

# Visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(phantom, cmap='gray', vmin=0, vmax=255)
plt.title('Ground Truth Phantom')
plt.axis('off')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(reconstruction_sirt, cmap='gray', vmin=0, vmax=255)
plt.title(f'SIRT\ne1={sirt_e1:.0f}, e2={sirt_e2:.0f}')
plt.axis('off')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(reconstruction_dfo_tv, cmap='gray', vmin=0, vmax=255)
plt.title(f'DFO + TV\ne1={dfo_tv_e1:.0f}, e2={dfo_tv_e2:.0f}')
plt.axis('off')
plt.colorbar()

plt.tight_layout()
comparison_path = os.path.join(output_folder, f'comparison_ph0{phantomNo}_{size}x{size}x{alpha}.png')
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Comparison figure saved to: {comparison_path}")

# Cleanup ASTRA memory
astra.data2d.delete(sinogram_id_recon)
astra.projector.delete(projector_id)