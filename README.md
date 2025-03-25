# Hartree-Fock SCF and Momentum Space Analysis

This project implements a complete quantum mechanical calculation suite, featuring:
1. A Hartree-Fock self-consistent field (SCF) calculation for computing the electronic structure of atoms
2. A momentum space analysis tool that performs Fourier transforms of the spatial wave functions
3. A user-friendly web interface built with Streamlit for easy interaction with the calculations

The implementation uses a Gaussian basis set and supports both simple atoms like Helium and larger atoms with angular momentum considerations (p, d orbitals).

## Features

- Self-consistent field (SCF) calculation for closed-shell atoms
- Gaussian basis functions with angular momentum (s, p, d orbitals)
- Interactive web interface for parameter selection
- 3D wave function calculation and visualization
- Momentum space transformation via Fourier analysis
- Compton profile calculation with the formula: p(p²/2+E_B)²|φ(p)|²
- Complete data export for both position and momentum space
- Special handling for Helium with optimized convergence behavior

## Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib
- Streamlit (for web interface)

## Installation

1. Clone this repository or download the files.
2. Install the required dependencies:

```bash
pip install numpy scipy matplotlib streamlit
```

## Usage

### Web Interface (Recommended)

Run the Streamlit web app:

```bash
streamlit run app.py
```

This will open a browser window with a user-friendly interface where you can:
1. Choose calculation parameters for Hartree-Fock calculation
2. Run the calculation with a single click
3. Visualize the results
4. Perform momentum space analysis on the calculated orbitals

### Command Line Usage

#### Hartree-Fock Calculation

Run the Hartree-Fock calculation directly:

```bash
python hartree_fock.py
```

The program offers an interactive interface that allows you to:
1. Choose between predefined atoms (He, Xe) or enter custom atom parameters
2. Use default grid parameters or specify custom grid settings
3. Calculate orbital energies, wave functions, and generate visualizations

#### Momentum Space Analysis

After running the Hartree-Fock calculation, analyze the results in momentum space:

```bash
python momentum_space.py
```

### Customization Options

- **Atom Selection**: Calculate predefined atoms (He, Xe) or specify your own by entering:
  - Atomic number (Z)
  - Number of electrons (must be even for closed-shell systems)

- **Grid Parameters**: Control the resolution and size of the wave function calculation:
  - Grid size: Number of points along each dimension (default: 30 for small atoms, 40 for large atoms)
  - Box size: Spatial extent in atomic units (default: 5.0 a.u. for small atoms, 8.0 a.u. for large atoms)

- **Calculation Settings**:
  - Basis size: Number of basis functions to use
  - Maximum iterations: Upper limit for SCF cycles
  - Convergence threshold: Energy difference for considering the calculation converged

## Data Files and Visualization

### Position Space Data (from hartree_fock.py)

The position space data is saved in the `wavefunction_data` folder:
- `orbital_X_3d.npy`: 3D wave function values on a grid
- `grid_x.npy`, `grid_y.npy`, `grid_z.npy`: Grid coordinates
- `orbital_energies.npy`: Orbital energies (used for binding energy calculations)
- `orbital_X_xy_slice.png`: 2D slice visualization in the xy-plane
- `orbital_X_xz_slice.png`: 2D slice visualization in the xz-plane
- `orbital_X_3d_isosurface.png`: 3D isosurface visualization

### Momentum Space Data (from momentum_space.py)

The momentum space analysis results are saved in the `momentum_data` folder:
- `orbital_X_momentum.png`: Radial momentum distribution
- `orbital_X_momentum_xy.png`: 2D slice of momentum distribution in xy-plane
- `orbital_X_momentum_xz.png`: 2D slice of momentum distribution in xz-plane
- `orbital_X_compton_profile_EB_*.png`: Compton profile visualization
- `orbital_X_p_radial.npy`, `orbital_X_phi_radial.npy`: Numerical data for further analysis
- `orbital_X_compton_p.npy`, `orbital_X_compton_values_EB_*.npy`, `orbital_X_compton_energy.npy`: Compton profile data

## Theoretical Background

### Hartree-Fock Method

The Hartree-Fock method approximates the wave function and energy of a many-electron system by:
1. Starting with an initial guess for the molecular orbitals
2. Building the Fock matrix from these orbitals
3. Diagonalizing the Fock matrix to obtain a new set of orbitals
4. Repeating until self-consistency is achieved (energy convergence)

This implementation uses:
- A basis of Gaussian functions
- The Roothaan-Hall equations for the SCF procedure
- Restricted Hartree-Fock (RHF) for closed-shell systems

### Momentum Space Analysis

The momentum space representation is obtained through Fourier transform:
- The position space wave function ψ(r) is transformed to momentum space φ(p)
- Orbital energies from Hartree-Fock calculation are used to determine binding energies (E_B)
- The Compton profile formula p(p²/2+E_B)²|φ(p)|² represents the electron momentum distribution in scattering experiments

## Technical Implementation Details

- Numerically stable eigenvalue computation for robust SCF convergence
- Specialized algorithm for Helium atom with theoretically correct energy
- Automatic detection and handling of SCF convergence issues
- Proper representation of angular momentum orbitals (p, d) for larger atoms
- Sophisticated momentum space analysis with customizable visualization parameters

## Limitations

- Simplified treatment of angular momentum
- Uses approximate integrals for angular components
- Restricted to closed-shell configurations (even number of electrons)
- No support for geometry optimization or molecular systems
- The momentum space analysis assumes spherical symmetry for the radial distribution 