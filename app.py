import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import subprocess
import time
import io
import zipfile
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import base64
import sys

# Import functions from our modules
from hartree_fock import run_hartree_fock_calculation
import momentum_space as ms

# Set page config
st.set_page_config(
    page_title="Hartree-Fock and Momentum Space Analysis",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #26A69A;
    }
    .info-text {
        font-size: 1rem;
        color: #616161;
    }
    .results-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #5E35B1;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Hartree-Fock and Momentum Space Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="info-text">This application performs quantum mechanical calculations using the Hartree-Fock method and momentum space analysis.</p>', unsafe_allow_html=True)

def run_hartree_fock():
    """Run Hartree-Fock calculation with UI progress indicators"""
    # Create a custom progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Cleaning up previous calculation results...")
        progress_bar.progress(5)
        
        status_text.text("Starting Hartree-Fock calculation...")
        progress_bar.progress(10)
        
        # Get parameters from session state
        mode = 'predefined' if st.session_state.atom_choice == "Predefined atoms" else 'custom'
        
        # Create a dictionary to store calculation parameters
        calc_params = {
            'mode': mode,
            'max_iterations': st.session_state.max_iterations,
            'convergence_threshold': st.session_state.convergence_threshold,
            'grid_size': st.session_state.grid_size,
            'box_size': st.session_state.box_size,
            'verbose': True,
            'selected_orbitals': st.session_state.orbital_selection  # Pass the orbital selection
        }
        
        # Set basis size if it exists in session state (default is None to allow auto-selection)
        if 'basis_size' in st.session_state and st.session_state.basis_size > 0:
            calc_params['basis_size'] = st.session_state.basis_size
            
        # Add atom-specific parameters
        if mode == 'predefined':
            calc_params['atom_name'] = 'He' if 'He' in st.session_state.atom_type else 'Xe'
        else:
            calc_params['atomic_number'] = st.session_state.atomic_number
            calc_params['num_electrons'] = st.session_state.electron_count
        
        # Print parameters for debugging
        st.write(f"Using calculation parameters: {calc_params}")
        
        # Run calculation
        results = run_hartree_fock_calculation(**calc_params)
        
        # Update progress
        progress_bar.progress(100)
        status_text.text("Hartree-Fock calculation completed successfully!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
        # Display basic results
        st.success(f"Calculation completed for atom with Z={results['Z']}!")
        st.info(f"Total energy: {results['total_energy']:.8f} a.u.")
        
        # Store results in session state
        st.session_state.hf_results = results
        
        # Clear previous momentum space analysis (if any)
        if 'momentum_results' in st.session_state:
            del st.session_state.momentum_results
        
        return True
        
    except Exception as e:
        st.error(f"An error occurred while running Hartree-Fock calculation: {e}")
        progress_bar.empty()
        status_text.empty()
        return False

# Main layout and functionality
tab1, tab2 = st.tabs(["Hartree-Fock Calculation", "Momentum Space Analysis"])

with tab1:
    st.markdown('<p class="sub-header">Hartree-Fock Self-Consistent Field Calculation</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Atom Parameters")
        atom_options = ["Predefined atoms", "Custom atom"]
        atom_choice = st.radio("Atom selection", atom_options, key="atom_choice")
        
        if atom_choice == "Predefined atoms":
            atom_type = st.selectbox("Select atom", ["He - Helium", "Xe - Xenon"], key="atom_type")
        else:
            atomic_number = st.number_input("Atomic number (Z)", min_value=1, max_value=118, value=2, key="atomic_number")
            electron_count = st.number_input("Number of electrons", min_value=2, max_value=atomic_number*2, value=2, step=2, key="electron_count")
            st.caption("Note: For closed-shell systems, electron count must be even")
    
    with col2:
        st.markdown("### Grid Parameters")
        grid_size = st.slider("Grid size", min_value=20, max_value=400, value=40, step=10, 
                             help="Number of points along each dimension", key="grid_size")
        box_size = st.slider("Box size (a.u.)", min_value=1.0, max_value=50.0, value=8.0, step=0.5,
                            help="Spatial extent in atomic units", key="box_size")
    
    st.markdown("### Calculation Settings")
    col3, col4 = st.columns(2)
    
    with col3:
        basis_size = st.number_input("Basis size", min_value=5, max_value=50, value=30, key="basis_size")
        max_iterations = st.number_input("Maximum SCF iterations", min_value=10, max_value=500, value=100, key="max_iterations")
    
    with col4:
        convergence_threshold = st.number_input("Convergence threshold", min_value=1e-10, max_value=1e-1, value=1e-6, 
                                               format="%.1e", step=1e-1, key="convergence_threshold")
    
    # Orbital calculation selection
    st.markdown("### Orbital Calculation Options")
    
    # Determine number of electrons based on selection
    if st.session_state.atom_choice == "Predefined atoms":
        num_electrons = 2 if "He" in st.session_state.atom_type else 54  # He or Xe
    else:
        num_electrons = st.session_state.electron_count
    
    num_occupied = num_electrons // 2  # For closed shell systems
    
    # Create a list of available orbitals
    orbital_options = [
        {"name": f"Orbital {i} ({label})", "index": i, "selected": i < 3 or i == num_occupied-1}
        for i, label in [
            (j, "core" if j < num_occupied//2 else 
                  "HOMO" if j == num_occupied-1 else 
                  "valence" if j < num_occupied else "virtual")
            for j in range(min(num_occupied+3, basis_size))  # Occupied + a few virtual
        ]
    ]
    
    # Compute default selections
    if 'orbital_selection' not in st.session_state:
        # By default select: first 3 orbitals, HOMO, and first virtual orbital
        default_selections = [o["index"] for o in orbital_options if o["selected"]]
        # Always include HOMO
        homo_idx = num_occupied - 1
        if homo_idx not in default_selections and homo_idx < len(orbital_options):
            default_selections.append(homo_idx)
        # Add first virtual if available
        if num_occupied < len(orbital_options) and num_occupied not in default_selections:
            default_selections.append(num_occupied)
        
        st.session_state.orbital_selection = default_selections
    
    # Create multiselect widget
    selected_orbitals = st.multiselect(
        "Select orbitals to calculate (calculating all orbitals can be time-consuming)",
        options=[o["index"] for o in orbital_options],
        default=st.session_state.orbital_selection,
        format_func=lambda x: next((o["name"] for o in orbital_options if o["index"] == x), f"Orbital {x}"),
        key="orbital_selection"
    )
    
    # Warning if no orbitals selected
    if not selected_orbitals:
        st.warning("Please select at least one orbital to calculate.")
    else:
        # Calculate how many orbitals of each type are selected
        total_orbitals = len(selected_orbitals)
        occupied_count = len([idx for idx in selected_orbitals if idx < num_occupied])
        virtual_count = total_orbitals - occupied_count
        has_homo = (num_occupied - 1) in selected_orbitals
        
        # Create info message
        info_message = f"Selected {total_orbitals} orbitals: {occupied_count} occupied"
        if has_homo:
            info_message += " (including HOMO)"
        if virtual_count > 0:
            info_message += f", {virtual_count} virtual"
        
        # Calculate estimated time based on number of orbitals and grid size
        # Higher grid sizes make calculation more time-consuming
        base_time_per_orbital = 1.0  # Base time per orbital in seconds
        grid_factor = (st.session_state.grid_size / 40) ** 2  # Quadratic increase with grid size
        estimated_time = total_orbitals * base_time_per_orbital * grid_factor
        
        # Format time
        time_str = ""
        if estimated_time < 60:
            time_str = f"~{estimated_time:.1f} seconds"
        else:
            time_str = f"~{estimated_time/60:.1f} minutes"
            
        info_message += f" (Estimated calculation time: {time_str})"
        
        st.info(info_message)
    
    # Run calculation button
    if st.button("Run Hartree-Fock Calculation", key="run_hf", type="primary") and selected_orbitals:
        success = run_hartree_fock()
        if success:
            st.success("Calculation completed! You can now proceed to the Momentum Space Analysis tab.")
            
    # Display previous results if available
    if os.path.exists('wavefunction_data'):
        st.markdown("### Previous Calculation Results")
        if os.path.exists('scf_convergence.png') and os.path.exists('hf_orbitals.png'):
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                st.image('scf_convergence.png', caption="SCF Convergence", use_column_width=True)
            
            with img_col2:
                st.image('hf_orbitals.png', caption="Radial Orbitals", use_column_width=True)
                
        # Find orbital wavefunction visualization
        st.markdown("#### 3D Wave Function Visualizations")
        
        # Get all xy and xz slice images
        xy_files = sorted(glob.glob('wavefunction_data/*_xy_slice.png'))
        xz_files = sorted(glob.glob('wavefunction_data/*_xz_slice.png'))
        
        if xy_files:
            # Display xy-plane slices in a grid (3 per row)
            st.markdown("##### XY-Plane Slices")
            
            # Calculate number of rows needed
            items_per_row = 3
            num_rows = (len(xy_files) + items_per_row - 1) // items_per_row
            
            for row in range(num_rows):
                # Create a row of columns
                cols = st.columns(items_per_row)
                
                # Fill each column in this row
                for col in range(items_per_row):
                    idx = row * items_per_row + col
                    if idx < len(xy_files):
                        orbital_name = os.path.basename(xy_files[idx]).split('_')[1]
                        cols[col].image(xy_files[idx], caption=f"Orbital {orbital_name}", use_column_width=True)
        
        if xz_files:
            # Display xz-plane slices in a grid (3 per row)
            st.markdown("##### XZ-Plane Slices")
            
            # Calculate number of rows needed
            items_per_row = 3
            num_rows = (len(xz_files) + items_per_row - 1) // items_per_row
            
            for row in range(num_rows):
                # Create a row of columns
                cols = st.columns(items_per_row)
                
                # Fill each column in this row
                for col in range(items_per_row):
                    idx = row * items_per_row + col
                    if idx < len(xz_files):
                        orbital_name = os.path.basename(xz_files[idx]).split('_')[1]
                        cols[col].image(xz_files[idx], caption=f"Orbital {orbital_name}", use_column_width=True)
        
        # Find and display 3D isosurface plots if available
        iso_files = sorted(glob.glob('wavefunction_data/*_isosurface.png'))
        if iso_files:
            st.markdown("##### 3D Isosurface Plots")
            
            # Calculate number of rows needed
            items_per_row = 3
            num_rows = (len(iso_files) + items_per_row - 1) // items_per_row
            
            for row in range(num_rows):
                # Create a row of columns
                cols = st.columns(items_per_row)
                
                # Fill each column in this row
                for col in range(items_per_row):
                    idx = row * items_per_row + col
                    if idx < len(iso_files):
                        orbital_name = os.path.basename(iso_files[idx]).split('_')[1]
                        cols[col].image(iso_files[idx], caption=f"Orbital {orbital_name}", use_column_width=True)

with tab2:
    st.markdown('<p class="sub-header">Momentum Space Analysis</p>', unsafe_allow_html=True)
    
    # Check if wavefunction data exists
    if not os.path.exists('wavefunction_data'):
        st.warning("No wavefunction data found. Please run a Hartree-Fock calculation first.")
    else:
        # Find available orbitals
        orbital_files = glob.glob('wavefunction_data/orbital_*_3d.npy')
        available_orbitals = sorted([int(os.path.basename(f).split('_')[1]) for f in orbital_files])
        
        if not available_orbitals:
            st.warning("No orbital data found. Please run a Hartree-Fock calculation first.")
        else:
            # Sidebar for parameters
            st.sidebar.markdown("### Momentum Space Parameters")
            
            # Orbital selection with improved display
            st.sidebar.markdown("#### Orbital Selection")
            
            # Get the highest occupied orbital (outermost electron)
            if hasattr(st.session_state, 'hf_results') and 'num_electrons' in st.session_state.hf_results:
                num_electrons = st.session_state.hf_results['num_electrons']
                highest_occupied = num_electrons // 2 - 1  # Zero-indexed
                st.sidebar.info(f"Highest occupied orbital index: {highest_occupied}")
            else:
                highest_occupied = max(available_orbitals)
            
            # Create a user-friendly format function for the orbital dropdown
            def format_orbital(orbital_idx):
                if orbital_idx == highest_occupied:
                    return f"Orbital {orbital_idx} (HOMO - highest occupied)"
                elif orbital_idx < highest_occupied:
                    return f"Orbital {orbital_idx} (occupied)"
                else:
                    return f"Orbital {orbital_idx} (virtual)"
            
            selected_orbital = st.sidebar.selectbox(
                "Select orbital to visualize",
                available_orbitals,
                index=available_orbitals.index(highest_occupied) if highest_occupied in available_orbitals else 0,
                format_func=format_orbital
            )
            
            # Momentum grid parameters
            st.sidebar.markdown("#### Grid Parameters")
            num_bins = st.sidebar.slider("Number of bins", min_value=50, max_value=500, value=100, step=10)
            p_cutoff = st.sidebar.slider("Momentum cutoff", min_value=0.1, max_value=1.0, value=1.0, step=0.05)
            
            # Compton profile parameters
            st.sidebar.markdown("#### Compton Profile Parameters")
            
            # Load orbital data to get p_max for setting plot range
            temp_psi, temp_x, temp_y, temp_z, _, temp_E_B = ms.load_wavefunction_data(selected_orbital)
            
            if temp_psi is not None:
                # Calculate momentum grid to set reasonable plot range limits
                temp_phi, temp_px, temp_py, temp_pz = ms.compute_momentum_wavefunction(temp_psi, temp_x, temp_y, temp_z)
                temp_p_radial, _ = ms.compute_radial_momentum_distribution(temp_phi, temp_px, temp_py, temp_pz)
                temp_p_max = np.max(temp_p_radial)
                
                # Compton profile plot range
                plot_range_col1, plot_range_col2 = st.sidebar.columns(2)
                with plot_range_col1:
                    p_min = st.number_input("Minimum p", min_value=0.0, max_value=temp_p_max, value=0.0, format="%.2f")
                with plot_range_col2:
                    p_max = st.number_input("Maximum p", min_value=0.01, max_value=temp_p_max, value=min(temp_p_max, 5.0), format="%.2f")
                
                # Calculate and display results
                if st.button("Calculate Momentum Space Properties", type="primary"):
                    # Show a progress indicator
                    ms_progress = st.progress(0)
                    ms_status = st.empty()
                    ms_status.text("Cleaning previous momentum space results...")
                    
                    # Clear previous results if they exist
                    if os.path.exists('momentum_data'):
                        for file in glob.glob(f'momentum_data/orbital_{selected_orbital}_*'):
                            os.remove(file)
                            
                    ms_progress.progress(20)
                    ms_status.text("Running momentum space analysis...")
                    
                    # Use the run_momentum_analysis function from momentum_space module
                    results = ms.run_momentum_analysis(
                        selected_orbital, 
                        num_bins=num_bins, 
                        p_cutoff=p_cutoff,
                        p_min=p_min,
                        p_max=p_max
                    )
                    
                    ms_progress.progress(100)
                    ms_status.text("Analysis complete!")
                    time.sleep(1)
                    ms_status.empty()
                    ms_progress.empty()
                    
                    if results:
                        # Store in session state for reuse
                        st.session_state.momentum_results = results
                        
                        # Display results
                        st.markdown('<p class="results-header">Momentum Space Results</p>', unsafe_allow_html=True)
                        
                        # Display radial distribution plot
                        st.markdown("### Radial Momentum Distribution")
                        st.pyplot(results['momentum_fig'])
                        
                        # Display slice plots in a more organized way
                        st.markdown("### Momentum Space Slices")
                        slice_cols = st.columns(2)
                        with slice_cols[0]:
                            st.markdown("#### XY-Plane Slice")
                            st.pyplot(results['xy_slice_fig'])
                        with slice_cols[1]:
                            st.markdown("#### XZ-Plane Slice")
                            st.pyplot(results['xz_slice_fig'])
                        
                        # Display Compton profile
                        st.markdown("### Compton Profile")
                        st.pyplot(results['compton_fig'])
                        
                        # Display data download options
                        st.markdown("### Download Results")
                        if not os.path.exists('momentum_data'):
                            os.makedirs('momentum_data', exist_ok=True)
                        
                        # Create download buttons in a more organized layout
                        st.markdown("#### Download Visualization Images")
                        download_cols = st.columns(2)
                        
                        with download_cols[0]:
                            with open(f'momentum_data/orbital_{selected_orbital}_momentum.png', 'rb') as f:
                                st.download_button(
                                    label="Download Momentum Distribution Plot",
                                    data=f,
                                    file_name=f"orbital_{selected_orbital}_momentum.png",
                                    mime="image/png"
                                )
                        
                            # Use the simpler filename for the Compton profile
                            compton_filename = f'momentum_data/orbital_{selected_orbital}_compton.png'
                            if os.path.exists(compton_filename):
                                with open(compton_filename, 'rb') as f:
                                    st.download_button(
                                        label="Download Compton Profile",
                                        data=f,
                                        file_name=f"orbital_{selected_orbital}_compton_profile.png",
                                        mime="image/png"
                                    )
                            else:
                                # Fallback to the longer filename if the simple one doesn't exist
                                compton_filename = f'momentum_data/orbital_{selected_orbital}_compton_profile_EB_{results["E_B"]:.4f}.png'
                                if os.path.exists(compton_filename):
                                    with open(compton_filename, 'rb') as f:
                                        st.download_button(
                                            label="Download Compton Profile",
                                            data=f,
                                            file_name=f"orbital_{selected_orbital}_compton_profile.png",
                                            mime="image/png"
                                        )
                                else:
                                    st.warning(f"Compton profile image not found")
                        
                        with download_cols[1]:
                            with open(f'momentum_data/orbital_{selected_orbital}_momentum_xy.png', 'rb') as f:
                                st.download_button(
                                    label="Download XY-Plane Slice",
                                    data=f,
                                    file_name=f"orbital_{selected_orbital}_momentum_xy.png",
                                    mime="image/png"
                                )
                        
                            with open(f'momentum_data/orbital_{selected_orbital}_momentum_xz.png', 'rb') as f:
                                st.download_button(
                                    label="Download XZ-Plane Slice",
                                    data=f,
                                    file_name=f"orbital_{selected_orbital}_momentum_xz.png",
                                    mime="image/png"
                                )
                        
                        # Add a button to download all data
                        st.markdown("#### Download Complete Dataset")
                        # Create a zip file with all data
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            # Add all files from momentum_data directory
                            for file in os.listdir('momentum_data'):
                                if file.startswith(f'orbital_{selected_orbital}'):
                                    zip_file.write(os.path.join('momentum_data', file), file)
                        
                        zip_buffer.seek(0)
                        st.download_button(
                            label="Download All Data (zip)",
                            data=zip_buffer,
                            file_name=f"orbital_{selected_orbital}_all_data.zip",
                            mime="application/zip"
                        )

# Footer
st.markdown("---")
st.markdown('<p class="info-text">Hartree-Fock and Momentum Space Analysis Web UI - Developed with Streamlit</p>', unsafe_allow_html=True) 