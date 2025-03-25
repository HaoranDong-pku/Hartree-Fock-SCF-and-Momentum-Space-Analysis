import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, fftshift
import os
import glob

def load_wavefunction_data(orbital_idx):
    """
    Load the saved 3D wave function data
    
    Parameters:
    -----------
    orbital_idx : int
        Index of the orbital
        
    Returns:
    --------
    tuple
        (psi_3d, x, y, z, orbital_idx, E_B) - Wave function values, grid coordinates, orbital index, and binding energy
    """
    data_dir = 'wavefunction_data'
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Run hartree_fock.py first.")
        return None, None, None, None, None, None
    
    # Load wave function data
    psi_file = os.path.join(data_dir, f'orbital_{orbital_idx}_3d.npy')
    if not os.path.exists(psi_file):
        # Try to find any orbital file
        orbital_files = glob.glob(os.path.join(data_dir, 'orbital_*_3d.npy'))
        if not orbital_files:
            print(f"No wave function data found in {data_dir}. Run hartree_fock.py first.")
            return None, None, None, None, None, None
        print(f"Orbital {orbital_idx} not found. Using {os.path.basename(orbital_files[0])} instead.")
        psi_file = orbital_files[0]
        orbital_idx = int(os.path.basename(psi_file).split('_')[1])
    
    psi_3d = np.load(psi_file)
    
    # Load grid coordinates
    x = np.load(os.path.join(data_dir, 'grid_x.npy'))
    y = np.load(os.path.join(data_dir, 'grid_y.npy'))
    z = np.load(os.path.join(data_dir, 'grid_z.npy'))
    
    # Try to load orbital energies
    try:
        orbital_energies_file = os.path.join(data_dir, 'orbital_energies.npy')
        if os.path.exists(orbital_energies_file):
            orbital_energies = np.load(orbital_energies_file)
            # Binding energy is negative of orbital energy
            if orbital_idx < len(orbital_energies):
                E_B = -orbital_energies[orbital_idx]
                print(f"Binding energy (E_B) calculated: {E_B:.6f} a.u.")
            else:
                print(f"Orbital index {orbital_idx} out of range for energy data. Using default E_B.")
                E_B = 0.5  # Default value if orbital energy isn't available
        else:
            print("Orbital energies file not found. Using default E_B.")
            E_B = 0.5  # Default value
    except Exception as e:
        print(f"Error loading orbital energies: {e}. Using default E_B.")
        E_B = 0.5  # Default value
    
    print(f"Loaded wave function data for orbital {orbital_idx}")
    print(f"Grid shape: {psi_3d.shape}")
    
    return psi_3d, x, y, z, orbital_idx, E_B

def get_momentum_grid_parameters():
    """
    Get custom grid parameters for momentum space calculations
    
    Returns:
    --------
    tuple
        (num_bins, p_cutoff) - Number of bins for radial distribution and momentum cutoff
    """
    print("\nEnter momentum grid parameters:")
    print("-" * 40)
    
    # Get number of bins
    while True:
        try:
            num_bins = int(input("Enter number of bins for radial distribution (default 100): ") or "100")
            if num_bins <= 0:
                print("Number of bins must be positive!")
                continue
            break
        except ValueError:
            print("Please enter a valid integer!")
    
    # Get momentum cutoff
    while True:
        try:
            p_cutoff = float(input("Enter momentum cutoff (0-1, fraction of max momentum to keep, default 1.0): ") or "1.0")
            if p_cutoff <= 0 or p_cutoff > 1:
                print("Momentum cutoff must be between 0 and 1!")
                continue
            break
        except ValueError:
            print("Please enter a valid number!")
    
    return num_bins, p_cutoff

def compute_momentum_wavefunction(psi_3d, x, y, z):
    """
    Compute the wave function in momentum space using Fourier transform
    
    Parameters:
    -----------
    psi_3d : array
        Wave function values in position space
    x, y, z : arrays
        Grid coordinates in position space
        
    Returns:
    --------
    tuple
        (phi_3d, px, py, pz) - Wave function values and grid coordinates in momentum space
    """
    # Compute grid spacing and size
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    
    # Perform 3D Fourier transform
    phi_3d = fftshift(fftn(psi_3d))
    
    # Normalize
    phi_3d *= dx * dy * dz
    
    # Compute momentum space coordinates
    # The momentum grid is determined by the position grid spacing
    nx, ny, nz = psi_3d.shape
    Lx, Ly, Lz = x[-1] - x[0], y[-1] - y[0], z[-1] - z[0]
    
    # Momentum grid spacing
    dpx = 2 * np.pi / Lx
    dpy = 2 * np.pi / Ly
    dpz = 2 * np.pi / Lz
    
    # Momentum grid coordinates
    px = np.fft.fftshift(np.fft.fftfreq(nx, dx) * 2 * np.pi)
    py = np.fft.fftshift(np.fft.fftfreq(ny, dy) * 2 * np.pi)
    pz = np.fft.fftshift(np.fft.fftfreq(nz, dz) * 2 * np.pi)
    
    return phi_3d, px, py, pz

def compute_radial_momentum_distribution(phi_3d, px, py, pz, num_bins=100, p_cutoff=1.0):
    """
    Compute the radial momentum distribution from the 3D momentum space wave function
    
    Parameters:
    -----------
    phi_3d : array
        Wave function values in momentum space
    px, py, pz : arrays
        Grid coordinates in momentum space
    num_bins : int
        Number of bins for radial distribution
    p_cutoff : float
        Fraction of maximum momentum to include
        
    Returns:
    --------
    tuple
        (p_radial, phi_radial) - Radial momentum and corresponding wave function values
    """
    # Create 3D momentum grid
    PX, PY, PZ = np.meshgrid(px, py, pz, indexing='ij')
    
    # Compute magnitude of momentum vector
    P = np.sqrt(PX**2 + PY**2 + PZ**2)
    
    # Flatten arrays
    p_flat = P.flatten()
    phi_flat = np.abs(phi_3d)**2  # Probability density in momentum space
    phi_flat = phi_flat.flatten()
    
    # Create radial grid with cutoff
    p_max = np.max(p_flat) * p_cutoff
    p_radial = np.linspace(0, p_max, num_bins)
    phi_radial = np.zeros(num_bins)
    
    # Bin the data
    bin_width = p_max / num_bins
    for i in range(num_bins):
        # Find points in this radial bin
        mask = (p_flat >= i * bin_width) & (p_flat < (i + 1) * bin_width)
        if np.any(mask):
            phi_radial[i] = np.mean(phi_flat[mask])
    
    # Normalize
    if np.sum(phi_radial) > 0:
        phi_radial /= np.max(phi_radial)
    
    # Apply p^2 factor for the radial distribution in 3D
    phi_radial *= p_radial**2
    
    return p_radial, phi_radial

def plot_momentum_wavefunction(p_radial, phi_radial, orbital_idx, figures):
    """
    Create figure for the radial momentum distribution
    
    Parameters:
    -----------
    p_radial : array
        Radial momentum values
    phi_radial : array
        Radial wave function values in momentum space
    orbital_idx : int
        Index of the orbital
    figures : list
        List to store figure objects
    """
    fig = plt.figure(figsize=(10, 6))
    plt.plot(p_radial, phi_radial, 'b-', linewidth=2)
    plt.xlabel('Momentum p (a.u.)')
    plt.ylabel('p²|φ(p)|²')
    plt.title(f'Momentum Space Wave Function - Orbital {orbital_idx}')
    plt.grid(True)
    
    # Save the figure
    os.makedirs('momentum_data', exist_ok=True)
    plt.savefig(f'momentum_data/orbital_{orbital_idx}_momentum.png')
    
    # Also save numerical data
    np.save(f'momentum_data/orbital_{orbital_idx}_p_radial.npy', p_radial)
    np.save(f'momentum_data/orbital_{orbital_idx}_phi_radial.npy', phi_radial)
    
    print(f"Momentum space wave function saved to momentum_data/orbital_{orbital_idx}_momentum.png")
    print(f"Numerical data saved to momentum_data/")
    
    figures.append(fig)

def plot_momentum_slices(phi_3d, px, py, pz, orbital_idx, figures):
    """
    Create figures for slices of the 3D momentum space wave function
    
    Parameters:
    -----------
    phi_3d : array
        Wave function values in momentum space
    px, py, pz : arrays
        Grid coordinates in momentum space
    orbital_idx : int
        Index of the orbital
    figures : list
        List to store figure objects
    """
    # Get midpoints
    mid_x = len(px) // 2
    mid_y = len(py) // 2
    mid_z = len(pz) // 2
    
    # Plot xy-plane slice
    fig1 = plt.figure(figsize=(10, 8))
    phi_xy = np.abs(phi_3d[:, :, mid_z])**2
    plt.pcolormesh(px, py, phi_xy.T, cmap='viridis', shading='auto')
    plt.colorbar(label='|φ(p)|²')
    plt.xlabel('px (a.u.)')
    plt.ylabel('py (a.u.)')
    plt.title(f'Momentum Space Wave Function (xy-plane) - Orbital {orbital_idx}')
    plt.axis('equal')
    
    # Save the figure
    os.makedirs('momentum_data', exist_ok=True)
    plt.savefig(f'momentum_data/orbital_{orbital_idx}_momentum_xy.png')
    figures.append(fig1)
    
    # Plot xz-plane slice
    fig2 = plt.figure(figsize=(10, 8))
    phi_xz = np.abs(phi_3d[:, mid_y, :])**2
    plt.pcolormesh(px, pz, phi_xz.T, cmap='viridis', shading='auto')
    plt.colorbar(label='|φ(p)|²')
    plt.xlabel('px (a.u.)')
    plt.ylabel('pz (a.u.)')
    plt.title(f'Momentum Space Wave Function (xz-plane) - Orbital {orbital_idx}')
    plt.axis('equal')
    
    # Save the figure
    plt.savefig(f'momentum_data/orbital_{orbital_idx}_momentum_xz.png')
    figures.append(fig2)
    
    print(f"Momentum space slices saved to momentum_data/")

def calculate_compton_profile(p_radial, phi_radial, E_B):
    """
    Calculate the formula p(p²/2+E_B)²|φ(p)|² for Compton profile analysis
    
    Parameters:
    -----------
    p_radial : array
        Radial momentum values
    phi_radial : array
        Radial wave function probability density in momentum space
    E_B : float
        Binding energy parameter in atomic units
        
    Returns:
    --------
    array
        Calculated Compton profile values
    """
    # Calculate the formula: p(p²/2+E_B)²|φ(p)|²
    compton_values = p_radial * (p_radial**2/2 + E_B)**2 * phi_radial
    
    # Normalize to maximum value for better visualization
    if np.max(np.abs(compton_values)) > 0:
        compton_values = compton_values / np.max(np.abs(compton_values))
    
    return compton_values

def get_plotting_range():
    """
    Get plotting range parameters from user input
    
    Returns:
    --------
    tuple
        (p_min, p_max) - Plotting range for momentum
    """
    print("\nEnter plotting range for Compton profile:")
    print("-" * 40)
    
    # Get plotting range
    while True:
        try:
            p_min = float(input("Enter minimum momentum p_min for plotting: "))
            p_max = float(input("Enter maximum momentum p_max for plotting: "))
            if p_min >= p_max:
                print("p_min must be less than p_max!")
                continue
            if p_min < 0:
                print("Warning: p_min should typically be non-negative for physical interpretation.")
                confirm = input("Continue anyway? (y/n): ")
                if confirm.lower() != 'y':
                    continue
            break
        except ValueError:
            print("Please enter valid numbers!")
    
    return p_min, p_max

def plot_compton_profile(p_radial, compton_values, E_B, p_min, p_max, orbital_idx, figures):
    """
    Create figure for the Compton profile formula
    
    Parameters:
    -----------
    p_radial : array
        Radial momentum values
    compton_values : array
        Calculated Compton profile values
    E_B : float
        Binding energy parameter used in calculation
    p_min, p_max : float
        Range of momentum to plot
    orbital_idx : int
        Index of the orbital
    figures : list
        List to store figure objects
    """
    # Filter values within the specified range
    mask = (p_radial >= p_min) & (p_radial <= p_max)
    p_plot = p_radial[mask]
    compton_plot = compton_values[mask]
    
    # Convert momentum to energy (E = p²/2m, in atomic units m=1)
    energy_plot = p_plot**2 / 2
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(energy_plot, compton_plot, 'r-', linewidth=2)
    plt.xlabel('$E=p^2/2m$ (a.u.)')
    plt.ylabel(r'$p(\frac{p^2}{2}+E_B)^2|\phi(p)|^2$')
    plt.title(f'Compton Profile (E_B = {E_B:.4f} a.u.) - Orbital {orbital_idx}')
    plt.grid(True)
    
    # Save the figure
    os.makedirs('momentum_data', exist_ok=True)
    filename = f'momentum_data/orbital_{orbital_idx}_compton_profile_EB_{E_B:.4f}.png'
    plt.savefig(filename)
    
    # Also save numerical data
    np.save(f'momentum_data/orbital_{orbital_idx}_compton_p.npy', p_plot)
    np.save(f'momentum_data/orbital_{orbital_idx}_compton_values_EB_{E_B:.4f}.npy', compton_plot)
    np.save(f'momentum_data/orbital_{orbital_idx}_compton_energy.npy', energy_plot)
    
    print(f"Compton profile saved to {filename}")
    print(f"Numerical data saved to momentum_data/")
    
    figures.append(fig)

def create_momentum_plot(p_radial, phi_radial, orbital_idx):
    """
    Create figure for the radial momentum distribution
    
    Parameters:
    -----------
    p_radial : array
        Radial momentum values
    phi_radial : array
        Radial wave function values in momentum space
    orbital_idx : int
        Index of the orbital
    
    Returns:
    --------
    matplotlib.figure.Figure
        The plot figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p_radial, phi_radial, 'b-', linewidth=2)
    ax.set_xlabel('Momentum p (a.u.)')
    ax.set_ylabel('p²|φ(p)|²')
    ax.set_title(f'Momentum Space Wave Function - Orbital {orbital_idx}')
    ax.grid(True)
    
    # Save data if output directory exists
    if os.path.exists('momentum_data'):
        os.makedirs('momentum_data', exist_ok=True)
        np.save(f'momentum_data/orbital_{orbital_idx}_p_radial.npy', p_radial)
        np.save(f'momentum_data/orbital_{orbital_idx}_phi_radial.npy', phi_radial)
        fig.savefig(f'momentum_data/orbital_{orbital_idx}_momentum.png')
    
    return fig

def create_momentum_slice_plots(phi_3d, px, py, pz, orbital_idx):
    """
    Create figures for slices of the 3D momentum space wave function
    
    Parameters:
    -----------
    phi_3d : array
        Wave function values in momentum space
    px, py, pz : arrays
        Grid coordinates in momentum space
    orbital_idx : int
        Index of the orbital
    
    Returns:
    --------
    tuple
        (fig1, fig2) - Matplotlib figures for xy and xz slices
    """
    # Get midpoints
    mid_x = len(px) // 2
    mid_y = len(py) // 2
    mid_z = len(pz) // 2
    
    # xy-plane slice
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    phi_xy = np.abs(phi_3d[:, :, mid_z])**2
    im1 = ax1.pcolormesh(px, py, phi_xy.T, cmap='viridis', shading='auto')
    plt.colorbar(im1, ax=ax1, label='|φ(p)|²')
    ax1.set_xlabel('px (a.u.)')
    ax1.set_ylabel('py (a.u.)')
    ax1.set_title(f'Momentum Space Wave Function (xy-plane) - Orbital {orbital_idx}')
    ax1.axis('equal')
    
    # xz-plane slice
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    phi_xz = np.abs(phi_3d[:, mid_y, :])**2
    im2 = ax2.pcolormesh(px, pz, phi_xz.T, cmap='viridis', shading='auto')
    plt.colorbar(im2, ax=ax2, label='|φ(p)|²')
    ax2.set_xlabel('px (a.u.)')
    ax2.set_ylabel('pz (a.u.)')
    ax2.set_title(f'Momentum Space Wave Function (xz-plane) - Orbital {orbital_idx}')
    ax2.axis('equal')
    
    # Save figures if output directory exists
    if os.path.exists('momentum_data'):
        os.makedirs('momentum_data', exist_ok=True)
        fig1.savefig(f'momentum_data/orbital_{orbital_idx}_momentum_xy.png')
        fig2.savefig(f'momentum_data/orbital_{orbital_idx}_momentum_xz.png')
    
    return fig1, fig2

def create_compton_profile_plot(p_radial, compton_values, E_B, p_min, p_max, orbital_idx):
    """
    Create figure for the Compton profile formula
    
    Parameters:
    -----------
    p_radial : array
        Radial momentum values
    compton_values : array
        Calculated Compton profile values
    E_B : float
        Binding energy parameter used in calculation
    p_min, p_max : float
        Range of momentum to plot
    orbital_idx : int
        Index of the orbital
    
    Returns:
    --------
    matplotlib.figure.Figure
        The plot figure
    """
    # Filter values within the specified range
    mask = (p_radial >= p_min) & (p_radial <= p_max)
    p_plot = p_radial[mask]
    compton_plot = compton_values[mask]
    
    # Convert momentum to energy (E = p²/2m, in atomic units m=1)
    energy_plot = p_plot**2 / 2
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(energy_plot, compton_plot, 'r-', linewidth=2)
    ax.set_xlabel('$E=p^2/2m$ (a.u.)')
    ax.set_ylabel(r'$p(\frac{p^2}{2}+E_B)^2|\phi(p)|^2$')
    ax.set_title(f'Compton Profile (E_B = {E_B:.4f} a.u.) - Orbital {orbital_idx}')
    ax.grid(True)
    
    # Save data if output directory exists
    if os.path.exists('momentum_data'):
        os.makedirs('momentum_data', exist_ok=True)
        filename = f'momentum_data/orbital_{orbital_idx}_compton_profile_EB_{E_B:.4f}.png'
        fig.savefig(filename)
        np.save(f'momentum_data/orbital_{orbital_idx}_compton_p.npy', p_plot)
        np.save(f'momentum_data/orbital_{orbital_idx}_compton_values_EB_{E_B:.4f}.npy', compton_plot)
        np.save(f'momentum_data/orbital_{orbital_idx}_compton_energy.npy', energy_plot)
    
    return fig

def get_plotting_range(p_max):
    """
    Get plotting range parameters from user input
    
    Parameters:
    -----------
    p_max : float
        Maximum momentum value
    
    Returns:
    --------
    tuple
        (p_min, p_max) - Plotting range for momentum
    """
    print("\nEnter plotting range for Compton profile:")
    print("-" * 40)
    
    # Get plotting range
    while True:
        try:
            p_min = float(input(f"Enter minimum momentum p_min (0-{p_max:.2f}): "))
            p_max_user = float(input(f"Enter maximum momentum p_max (0-{p_max:.2f}): "))
            if p_min >= p_max_user:
                print("p_min must be less than p_max!")
                continue
            if p_min < 0:
                print("Warning: p_min should typically be non-negative for physical interpretation.")
                confirm = input("Continue anyway? (y/n): ")
                if confirm.lower() != 'y':
                    continue
            break
        except ValueError:
            print("Please enter valid numbers!")
    
    return p_min, p_max_user

def run_momentum_analysis(orbital_idx, num_bins=100, p_cutoff=1.0, p_min=None, p_max=None):
    """
    Run the complete momentum space analysis for a given orbital
    
    Parameters:
    -----------
    orbital_idx : int
        Index of the orbital to analyze
    num_bins : int
        Number of bins for radial distribution
    p_cutoff : float
        Fraction of maximum momentum to include
    p_min, p_max : float
        Range of momentum to plot (if None, will use 0 and max momentum)
    
    Returns:
    --------
    dict
        Dictionary containing all results and figures
    """
    # Load data
    psi_3d, x, y, z, orbital_idx, E_B = load_wavefunction_data(orbital_idx)
    
    if psi_3d is None:
        return None
    
    # Compute momentum space wave function
    print("\nCalculating momentum space wave function...")
    phi_3d, px, py, pz = compute_momentum_wavefunction(psi_3d, x, y, z)
    
    # Compute radial distribution
    print("Computing radial momentum distribution...")
    p_radial, phi_radial = compute_radial_momentum_distribution(
        phi_3d, px, py, pz, num_bins=num_bins, p_cutoff=p_cutoff
    )
    
    # Set default p_min, p_max if not provided
    p_max_available = np.max(p_radial)
    if p_min is None:
        p_min = 0.0
    if p_max is None:
        p_max = min(p_max_available, 5.0)
    
    # Calculate Compton profile
    print("Calculating Compton profile...")
    compton_values = calculate_compton_profile(p_radial, phi_radial, E_B)
    
    # Create figures
    print("Generating plots...")
    momentum_fig = create_momentum_plot(p_radial, phi_radial, orbital_idx)
    xy_slice_fig, xz_slice_fig = create_momentum_slice_plots(phi_3d, px, py, pz, orbital_idx)
    compton_fig = create_compton_profile_plot(p_radial, compton_values, E_B, p_min, p_max, orbital_idx)
    
    # Return all results in a dictionary
    return {
        'orbital_idx': orbital_idx,
        'E_B': E_B,
        'phi_3d': phi_3d,
        'px': px, 'py': py, 'pz': pz,
        'p_radial': p_radial,
        'phi_radial': phi_radial,
        'compton_values': compton_values,
        'momentum_fig': momentum_fig,
        'xy_slice_fig': xy_slice_fig,
        'xz_slice_fig': xz_slice_fig,
        'compton_fig': compton_fig
    }

def main():
    print("Momentum Space Wave Function Calculator")
    print("=" * 40)
    
    try:
        # Ask for the orbital index
        while True:
            try:
                orbital_idx = int(input("Enter orbital index (default is outermost orbital): ") or "-1")
                break
            except ValueError:
                print("Please enter a valid integer!")
        
        # Ask for momentum grid settings
        print("\nMomentum Grid Settings:")
        print("1. Use default parameters")
        print("2. Enter custom grid parameters")
        
        while True:
            try:
                grid_mode = int(input("\nEnter your choice (1 or 2): "))
                if grid_mode not in [1, 2]:
                    print("Please enter 1 or 2!")
                    continue
                break
            except ValueError:
                print("Please enter a valid number!")
        
        # Get grid parameters if custom mode is chosen
        if grid_mode == 2:
            # Get number of bins
            while True:
                try:
                    num_bins = int(input("Enter number of bins for radial distribution (default 100): ") or "100")
                    if num_bins <= 0:
                        print("Number of bins must be positive!")
                        continue
                    break
                except ValueError:
                    print("Please enter a valid integer!")
            
            # Get momentum cutoff
            while True:
                try:
                    p_cutoff = float(input("Enter momentum cutoff (0-1, fraction of max momentum to keep, default 1.0): ") or "1.0")
                    if p_cutoff <= 0 or p_cutoff > 1:
                        print("Momentum cutoff must be between 0 and 1!")
                        continue
                    break
                except ValueError:
                    print("Please enter a valid number!")
        else:
            num_bins, p_cutoff = 100, 1.0  # Default values
        
        # Load data to get max momentum value
        temp_psi, temp_x, temp_y, temp_z, temp_idx, temp_E_B = load_wavefunction_data(orbital_idx)
        if temp_psi is not None:
            temp_phi, temp_px, temp_py, temp_pz = compute_momentum_wavefunction(temp_psi, temp_x, temp_y, temp_z)
            temp_p_radial, _ = compute_radial_momentum_distribution(temp_phi, temp_px, temp_py, temp_pz)
            p_max = np.max(temp_p_radial)
            
            # Get plotting range for Compton profile
            p_min, p_max_user = get_plotting_range(p_max)
            
            # Run the analysis
            results = run_momentum_analysis(
                orbital_idx, 
                num_bins=num_bins, 
                p_cutoff=p_cutoff, 
                p_min=p_min, 
                p_max=p_max_user
            )
            
            if results:
                print("\nCalculation complete. All results saved in the 'momentum_data' folder.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 