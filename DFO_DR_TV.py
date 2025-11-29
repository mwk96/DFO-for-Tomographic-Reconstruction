import os
import numpy as np
import imageio
import time
from PIL import Image
import astra
from matplotlib import pyplot as plt

# Parameters
size = 64
alpha = 6
phantomNo = 5
maxFE = 100000

# TV Parameter
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

# Update Boolean Table
update_boolean_table(master_boolean_table, intersected_pixels, phantom, specified_angle_index)

# Save the boolean table visualization for reference
output_folder = 'recon_final1'
os.makedirs(output_folder, exist_ok=True)

plt.figure(figsize=(6, 6))
plt.imshow(master_boolean_table, cmap='gray')
plt.title('Master Boolean Table')
plt.colorbar(label="True (White) / False (Black)")
boolean_table_path = os.path.join(output_folder, 'master_boolean_table.png')
plt.savefig(boolean_table_path)
plt.close()

print(f"Master Boolean Table saved as '{boolean_table_path}'")

# Count valid pixels
valid_pixel_count = np.sum(master_boolean_table)
total_pixels = master_boolean_table.size
valid_pixel_percentage = (valid_pixel_count / total_pixels) * 100
print(f"Valid pixels in the boolean table: {valid_pixel_count} ({valid_pixel_percentage:.2f}%)")


# Total Variation computation function
def compute_total_variation(x):
    """
    Compute Total Variation of image x (2D array)
    TV(x) = Σᵢⱼ √(|xᵢ₊₁,ⱼ - xᵢ,ⱼ|² + |xᵢ,ⱼ₊₁ - xᵢ,ⱼ|²)
    """
    # Handle boundary conditions by padding
    x_padded = np.pad(x, ((0, 1), (0, 1)), mode='edge')
    
    # Compute gradients
    dx = x_padded[1:, :-1] - x_padded[:-1, :-1]  # Horizontal gradient
    dy = x_padded[:-1, 1:] - x_padded[:-1, :-1]  # Vertical gradient
    
    # Compute TV as sum of gradient magnitudes
    tv = np.sum(np.sqrt(dx**2 + dy**2))
    return tv


# Image quality metrics functions
def compute_snr_roi(reconstruction, roi_mask):
    """
    Compute Signal-to-Noise Ratio (SNR) using ROI-based formula.
    SNR = μ_ROI / σ_ROI
    Higher is better.
    """
    roi_values = reconstruction[roi_mask]
    
    mu_roi = np.mean(roi_values)
    sigma_roi = np.std(roi_values)
    
    if sigma_roi == 0:
        return np.inf
    
    snr = mu_roi / sigma_roi
    return snr


def compute_cnr(reconstruction, target_mask, background_mask):
    """
    Compute Contrast-to-Noise Ratio (CNR).
    CNR = |μ_t - μ_b| / σ_b
    Higher is better.
    """
    target_values = reconstruction[target_mask]
    background_values = reconstruction[background_mask]
    
    mu_t = np.mean(target_values)
    mu_b = np.mean(background_values)
    sigma_b = np.std(background_values)
    
    if sigma_b == 0:
        return np.inf
    
    cnr = np.abs(mu_t - mu_b) / sigma_b
    return cnr


# DFO evaluation function with TV regularization
def eval_with_tv(x, refSino, W, boolean_table, mu=TV_MU):
    """
    Evaluation function that includes TV regularization
    Fitness = e1 + μ * TV(x)
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


# DFO with TV (no SSE - fixed 0-255 bounds)
def dfo_with_tv(boolean_table, refSino, W, maxFE=100000, N=2, delta=0.001, mu=TV_MU):
    """
    DFO with TV regularization only.
    Uses fixed bounds [0, 255] - no Search Space Expansion.
    Starts from random initialization with boolean mask applied.
    """
    D = size * size  # Size of the image flattened
    X = np.zeros([N, D], dtype=np.float32)  # Population matrix
    
    # Fixed bounds
    LOWER_BOUND = 0.0
    UPPER_BOUND = 255.0
    
    # Initialize all flies randomly within fixed bounds
    for i in range(N):
        X[i] = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=D)
        X[i][~boolean_table.flatten()] = 0  # Apply boolean mask
    
    fitness = [None] * N
    e1_values = [None] * N
    tv_values = [None] * N
    FE = 0  # Function evaluations counter
    
    # Storage for tracking progress
    fitness_history = []
    e1_history = []
    e2_history = []
    tv_history = []
    
    # Evaluate fitness for each fly
    print(f"Starting DFO with TV (μ={TV_MU}), bounds=[{LOWER_BOUND}, {UPPER_BOUND}]...")
    for i in range(N):
        fitness[i], e1_values[i], tv_values[i] = eval_with_tv(X[i], refSino, W, boolean_table, mu)
        FE += 1
    
    # Find the best solution
    s = np.argmin(fitness)
    
    print(f"Initial best fitness: {fitness[s]:.2f} (e1: {e1_values[s]:.2f}, TV: {tv_values[s]:.2f})")
    
    # Iterative optimization
    progress_check_interval = maxFE // 20  # Check progress every 5%
    last_progress_check = 0
    
    while FE < maxFE:
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
                low=LOWER_BOUND, 
                high=UPPER_BOUND, 
                size=np.sum(disturbance_mask)
            )
            
            # Update equation for non-disturbed values
            non_disturbance_mask = ~disturbance_mask
            u_vals = np.random.rand(D)[non_disturbance_mask]
            X[i, non_disturbance_mask] = X[bNeighbour, non_disturbance_mask] + u_vals * (
                    X[s, non_disturbance_mask] - X[i, non_disturbance_mask])
            
            # Apply fixed bounds clamping
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
    
    # Final solution
    best_solution_2d = X[s].reshape((size, size))
    
    # Plot convergence curves
    if fitness_history:
        plot_tv_convergence(fitness_history, e1_history, e2_history, tv_history)
    
    print(f"DFO+TV completed.")
    print(f"Final fitness: {fitness[s]:.2f} (e1: {e1_values[s]:.2f}, TV: {tv_values[s]:.2f})")
    
    return best_solution_2d


# Function to plot TV convergence
def plot_tv_convergence(fitness_hist, e1_hist, e2_hist, tv_hist):
    """
    Plot convergence curves for DFO with TV
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    iterations = range(len(fitness_hist))
    
    # Plot 1: Combined fitness
    ax1.semilogy(iterations, fitness_hist, 'b-', linewidth=2)
    ax1.set_title('Combined Fitness (e1 + μ*TV)')
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
    
    # Plot 4: Configuration info
    ax4.text(0.5, 0.5, f'DFO with TV Regularization\n\nμ = {TV_MU}\nBounds: [0, 255]\nNo SSE', 
             ha='center', va='center', fontsize=14, transform=ax4.transAxes)
    ax4.set_title('Configuration')
    ax4.axis('off')
    
    plt.tight_layout()
    convergence_path = os.path.join(output_folder, f'tv_convergence_ph0{phantomNo}_{size}x{size}x{alpha}.png')
    plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"TV convergence curves saved to: {convergence_path}")


# Calculate final errors
def compute_final_errors(reconstruction, refSino, W, phantom):
    # Compute e1 by forward projecting the final reconstruction
    xSino = np.zeros_like(refSino)
    W.FP(reconstruction, out=xSino)
    e1 = np.sum(np.abs(refSino.flatten() - xSino.flatten()))
    
    # Compute e2 as pixel-wise difference from ground truth
    e2 = np.sum(np.abs(phantom.flatten() - reconstruction.flatten()))
    
    return e1, e2


# Start timing
start_time = time.time()

# Run DFO with TV (no SSE)
reconstruction_dfo_tv = dfo_with_tv(
    master_boolean_table, 
    refSino.copy(), 
    W_recon,
    maxFE=maxFE,
    N=2,
    mu=TV_MU
)

dfo_time = time.time() - start_time
print(f"\nDFO with TV completed in {dfo_time:.2f} seconds")

# Compute DFO errors and metrics
dfo_tv_e1, dfo_tv_e2 = compute_final_errors(reconstruction_dfo_tv, refSino, W_recon, phantom)
dfo_tv_tv = compute_total_variation(reconstruction_dfo_tv)

# Define ROI masks
roi_mask = phantom > (0.5 * np.max(phantom))
target_mask = phantom > (0.7 * np.max(phantom))
background_mask = phantom < (0.3 * np.max(phantom))

# Compute SNR and CNR
if np.sum(roi_mask) > 0:
    dfo_snr = compute_snr_roi(reconstruction_dfo_tv, roi_mask)
else:
    dfo_snr = 0.0
    print("Warning: Could not compute SNR - invalid ROI mask")

if np.sum(target_mask) > 0 and np.sum(background_mask) > 0:
    dfo_cnr = compute_cnr(reconstruction_dfo_tv, target_mask, background_mask)
else:
    dfo_cnr = 0.0
    print("Warning: Could not compute CNR - invalid masks")

# Normalize and save the final DFO+TV result
minV, maxV = np.min(reconstruction_dfo_tv), np.max(reconstruction_dfo_tv)
normalized_dfo_tv = ((reconstruction_dfo_tv - minV) / (maxV - minV) * 255).astype(np.uint8)
normalized_dfo_tv[~master_boolean_table] = 0  # Final mask application
dfo_tv_output_path = os.path.join(output_folder, f'reconstruction_DFO_TV_ph0{phantomNo}_{size}x{size}x{alpha}.bmp')
imageio.imwrite(dfo_tv_output_path, normalized_dfo_tv)
print(f"Final DFO+TV reconstruction saved to: {dfo_tv_output_path}")

# Print results
print(f"\n{'='*80}")
print(f"DFO + TV:  e1={dfo_tv_e1:.2f} | e2={dfo_tv_e2:.2f} | TV={dfo_tv_tv:.2f}")
print(f"           Time={dfo_time:.2f}s | SNR={dfo_snr:.4f} | CNR={dfo_cnr:.4f}")
print(f"{'='*80}\n")

print("\n--- Image Quality Metrics (Dimensionless) ---")
print(f"SNR: {dfo_snr:.4f}")
print(f"CNR: {dfo_cnr:.4f}")

# Cleanup ASTRA memory
astra.data2d.delete(sinogram_id_recon)
astra.projector.delete(projector_id)

print(f"\n{'='*60}")
print(f"IMPLEMENTATION SUMMARY:")
print(f"{'='*60}")
print(f"✅ Total Variation regularization with μ={TV_MU}")
print(f"✅ Boolean mask (a priori knowledge) applied")
print(f"✅ Fixed bounds: [0, 255] (no SSE)")
print(f"✅ SNR and CNR metrics computed")
print(f"{'='*60}")