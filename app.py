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
            'verbose': True
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
    
    # Run calculation button
    if st.button("Run Hartree-Fock Calculation", key="run_hf", type="primary"):
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
        wf_files = glob.glob('wavefunction_data/*_xy_slice.png')
        if wf_files:
            st.markdown("#### 3D Wave Function Visualizations")
            wf_cols = st.columns(len(wf_files))
            
            for i, wf_file in enumerate(wf_files):
                with wf_cols[i]:
                    orbital_name = os.path.basename(wf_file).split('_')[1]
                    st.image(wf_file, caption=f"Orbital {orbital_name} (xy-plane)", use_column_width=True)

with tab2:
    st.markdown('<p class="sub-header">Momentum Space Analysis</p>', unsafe_allow_html=True)
    
    # Check if wavefunction data exists
    if not os.path.exists('wavefunction_data'):
        st.warning("No wavefunction data found. Please run a Hartree-Fock calculation first.")
    else:
        # Find available orbitals
        orbital_files = glob.glob('wavefunction_data/orbital_*_3d.npy')
        available_orbitals = [int(os.path.basename(f).split('_')[1]) for f in orbital_files]
        
        if not available_orbitals:
            st.warning("No orbital data found. Please run a Hartree-Fock calculation first.")
        else:
            # Sidebar for parameters
            st.sidebar.markdown("### Momentum Space Parameters")
            
            # Orbital selection
            default_orbital = max(available_orbitals) if -1 in available_orbitals else available_orbitals[0]
            selected_orbital = st.sidebar.selectbox(
                "Select orbital", 
                available_orbitals,
                index=available_orbitals.index(default_orbital) if default_orbital in available_orbitals else 0,
                format_func=lambda x: f"Orbital {x}" + (" (outermost)" if x == max(available_orbitals) else "")
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
                    # Use the run_momentum_analysis function from momentum_space module
                    results = ms.run_momentum_analysis(
                        selected_orbital, 
                        num_bins=num_bins, 
                        p_cutoff=p_cutoff,
                        p_min=p_min,
                        p_max=p_max
                    )
                    
                    if results:
                        # Display results
                        st.markdown('<p class="results-header">Momentum Space Results</p>', unsafe_allow_html=True)
                        
                        # Display radial distribution plot
                        st.pyplot(results['momentum_fig'])
                        
                        # Display slice plots
                        st.markdown("### Momentum Space Slices")
                        slice_cols = st.columns(2)
                        with slice_cols[0]:
                            st.pyplot(results['xy_slice_fig'])
                        with slice_cols[1]:
                            st.pyplot(results['xz_slice_fig'])
                        
                        # Display Compton profile
                        st.markdown("### Compton Profile")
                        st.pyplot(results['compton_fig'])
                        
                        # Display data download options
                        st.markdown("### Download Results")
                        if not os.path.exists('momentum_data'):
                            os.makedirs('momentum_data', exist_ok=True)
                        
                        # Create download buttons
                        download_cols = st.columns(4)
                        
                        with download_cols[0]:
                            with open(f'momentum_data/orbital_{selected_orbital}_momentum.png', 'rb') as f:
                                st.download_button(
                                    label="Download Momentum Plot",
                                    data=f,
                                    file_name=f"orbital_{selected_orbital}_momentum.png",
                                    mime="image/png"
                                )
                        
                        with download_cols[1]:
                            with open(f'momentum_data/orbital_{selected_orbital}_momentum_xy.png', 'rb') as f:
                                st.download_button(
                                    label="Download XY Slice",
                                    data=f,
                                    file_name=f"orbital_{selected_orbital}_momentum_xy.png",
                                    mime="image/png"
                                )
                                
                        with download_cols[2]:
                            with open(f'momentum_data/orbital_{selected_orbital}_compton_profile_EB_{results["E_B"]:.4f}.png', 'rb') as f:
                                st.download_button(
                                    label="Download Compton Profile",
                                    data=f,
                                    file_name=f"orbital_{selected_orbital}_compton_profile.png",
                                    mime="image/png"
                                )
                                
                        with download_cols[3]:
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