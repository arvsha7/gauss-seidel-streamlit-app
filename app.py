"""
Gauss-Seidel Power Flow Analysis - Interactive Web Application
Author: Educational Tool for Power Systems Analysis
Framework: Streamlit
"""

import streamlit as st
import numpy as np
import pandas as pd
import cmath
from io import StringIO
import json

# Page configuration
st.set_page_config(
    page_title="Gauss-Seidel Power Flow Solver",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .iteration-box {
        background-color: #f8f9fa;
        border-left: 5px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def complex_to_string(c, precision=4):
    """Format complex number as string"""
    real = round(c.real, precision)
    imag = round(abs(c.imag), precision)
    sign = "+" if c.imag >= 0 else "-"
    return f"{real} {sign} j{imag}"

def polar_to_string(magnitude, angle_deg, precision=4):
    """Format polar form as string"""
    return f"{round(magnitude, precision)} ∠ {round(angle_deg, precision)}°"

def build_ybus(bus_data, line_data):
    """
    Build Y-bus matrix from line data
    
    Parameters:
    - bus_data: DataFrame with bus information
    - line_data: DataFrame with line impedance data (From, To, R, X)
    
    Returns:
    - Y_bus: Complex admittance matrix
    """
    num_buses = len(bus_data)
    Y_bus = np.zeros((num_buses, num_buses), dtype=complex)
    
    for _, line in line_data.iterrows():
        from_bus = int(line['From']) - 1  # Convert to 0-indexed
        to_bus = int(line['To']) - 1
        
        # Line impedance: z = R + jX
        z = complex(line['R'], line['X'])
        
        # Line admittance: y = 1/z
        y = 1.0 / z
        
        # Build Y-bus (symmetric matrix)
        Y_bus[from_bus, to_bus] -= y
        Y_bus[to_bus, from_bus] -= y
        Y_bus[from_bus, from_bus] += y
        Y_bus[to_bus, to_bus] += y
    
    return Y_bus

def initialize_voltages(bus_data, initial_settings):
    """
    Initialize voltage vector based on bus types and initial settings
    
    Parameters:
    - bus_data: DataFrame with bus information
    - initial_settings: Dictionary with initial voltage settings
    
    Returns:
    - V: Complex voltage vector
    """
    num_buses = len(bus_data)
    V = np.zeros(num_buses, dtype=complex)
    
    for i, bus in bus_data.iterrows():
        bus_type = bus['Type']
        
        if bus_type == 1:  # Slack bus
            # Use specified voltage and angle
            V[i] = bus['V'] * cmath.exp(1j * np.radians(bus['Angle']))
        
        elif bus_type == 2:  # PV bus
            # Use specified voltage magnitude, initial angle
            mag = bus['V']
            angle = initial_settings.get(f'pv_angle_{i}', 0.0)
            V[i] = mag * cmath.exp(1j * np.radians(angle))
        
        else:  # PQ bus (Type 3)
            # Use initial voltage settings
            mag = initial_settings.get(f'pq_mag_{i}', 1.0)
            angle = initial_settings.get(f'pq_angle_{i}', 0.0)
            V[i] = mag * cmath.exp(1j * np.radians(angle))
    
    return V

def calculate_net_power(bus_data):
    """
    Calculate net power injection for each bus
    S_net = (P_Gen - P_Load) + j(Q_Gen - Q_Load)
    
    Returns:
    - S_net: Complex power injection vector
    """
    num_buses = len(bus_data)
    S_net = np.zeros(num_buses, dtype=complex)
    
    for i, bus in bus_data.iterrows():
        P_net = bus['P_Gen'] - bus['P_Load']
        Q_net = bus['Q_Gen'] - bus['Q_Load']
        S_net[i] = complex(P_net, Q_net)
    
    return S_net

def gauss_seidel_iteration(V, Y_bus, S_net, bus_data):
    """
    Perform one iteration of Gauss-Seidel method for power flow
    
    Parameters:
    - V: Current voltage vector
    - Y_bus: Admittance matrix
    - S_net: Net power injection vector
    - bus_data: Bus information DataFrame
    
    Returns:
    - V_new: Updated voltage vector
    - calculation_details: List of calculation details for each bus
    """
    num_buses = len(V)
    V_new = V.copy()
    calculation_details = []
    
    for i in range(num_buses):
        bus_type = bus_data.iloc[i]['Type']
        
        # Skip Slack Bus (Type 1)
        if bus_type == 1:
            details = {
                'bus': i + 1,
                'type': 'Slack',
                'fixed': True,
                'V_old': V[i],
                'V_new': V[i],
                'error': 0.0
            }
            calculation_details.append(details)
            continue
        
        # Store old value
        V_old = V_new[i]
        
        # Calculate sum of Y_ij * V_j for all j != i
        sum_YV = 0.0
        sum_terms = []
        for j in range(num_buses):
            if i != j:
                term = Y_bus[i, j] * V_new[j]
                sum_YV += term
                sum_terms.append({
                    'j': j + 1,
                    'Y_ij': Y_bus[i, j],
                    'V_j': V_new[j],
                    'product': term
                })
        
        # Calculate power term: (S_i*) / (V_i*)
        S_conj = np.conj(S_net[i])
        V_conj = np.conj(V_old)
        power_term = S_conj / V_conj if abs(V_conj) > 1e-10 else 0.0
        
        # Calculate new voltage: V_i = (1/Y_ii) * [power_term - sum_YV]
        Y_ii = Y_bus[i, i]
        V_new[i] = (power_term - sum_YV) / Y_ii
        
        # For PV buses, maintain voltage magnitude
        if bus_type == 2:
            target_mag = bus_data.iloc[i]['V']
            current_angle = cmath.phase(V_new[i])
            V_new[i] = target_mag * cmath.exp(1j * current_angle)
        
        # Calculate error
        error = abs(V_new[i] - V_old)
        
        # Store calculation details
        details = {
            'bus': i + 1,
            'type': 'PV' if bus_type == 2 else 'PQ',
            'fixed': False,
            'V_old': V_old,
            'S_net': S_net[i],
            'S_conj': S_conj,
            'V_conj': V_conj,
            'power_term': power_term,
            'sum_YV': sum_YV,
            'sum_terms': sum_terms,
            'Y_ii': Y_ii,
            'V_new': V_new[i],
            'error': error,
            'target_mag': bus_data.iloc[i]['V'] if bus_type == 2 else None
        }
        calculation_details.append(details)
    
    return V_new, calculation_details

def calculate_slack_bus_power(slack_bus_idx, V, Y_bus):
    """
    Calculate active and reactive power at slack bus after convergence
    
    S_slack = V_slack * conj(I_slack)
    where I_slack = sum(Y_slack,j * V_j) for all j
    """
    I_slack = 0.0
    for j in range(len(V)):
        I_slack += Y_bus[slack_bus_idx, j] * V[j]
    
    S_slack = V[slack_bus_idx] * np.conj(I_slack)
    P_slack = S_slack.real
    Q_slack = S_slack.imag
    
    return P_slack, Q_slack

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'bus_data' not in st.session_state:
    # Default bus data
    st.session_state.bus_data = pd.DataFrame({
        'Bus': [1, 2, 3],
        'Type': [1, 2, 3],  # 1=Slack, 2=PV, 3=PQ
        'V': [1.05, 1.00, 1.00],
        'Angle': [0.0, 0.0, 0.0],
        'P_Gen': [0.0, 0.50, 0.0],
        'Q_Gen': [0.0, 0.30, 0.0],
        'P_Load': [0.0, 0.20, 0.45],
        'Q_Load': [0.0, 0.10, 0.15]
    })

if 'line_data' not in st.session_state:
    # Default line data
    st.session_state.line_data = pd.DataFrame({
        'From': [1, 2, 1],
        'To': [2, 3, 3],
        'R': [0.02, 0.01, 0.03],
        'X': [0.04, 0.05, 0.06]
    })

if 'iteration_history' not in st.session_state:
    st.session_state.iteration_history = []

if 'current_iteration' not in st.session_state:
    st.session_state.current_iteration = 0

if 'V_current' not in st.session_state:
    st.session_state.V_current = None

if 'is_converged' not in st.session_state:
    st.session_state.is_converged = False

if 'Y_bus' not in st.session_state:
    st.session_state.Y_bus = None

if 'S_net' not in st.session_state:
    st.session_state.S_net = None

if 'tolerance' not in st.session_state:
    st.session_state.tolerance = 0.0001

if 'max_iterations' not in st.session_state:
    st.session_state.max_iterations = 50

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Header
st.markdown("""
    <div class="main-header">
        <h1>⚡ Gauss-Seidel Power Flow Analysis</h1>
        <p>Interactive Load Flow Solver for Educational Purposes</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar - Tutorial and Settings
with st.sidebar:
    st.header("📚 Tutorial")
    
    with st.expander("🔍 What is Power Flow Analysis?"):
        st.markdown("""
        **Power Flow (Load Flow) Analysis** determines the steady-state operating 
        condition of a power system. It calculates:
        
        - **Voltage magnitudes** at all buses
        - **Voltage angles** at all buses
        - **Active power flows** in transmission lines
        - **Reactive power flows** in transmission lines
        
        **Bus Types:**
        - **Type 1 (Slack/Swing):** V and δ specified
        - **Type 2 (PV/Generator):** P and |V| specified
        - **Type 3 (PQ/Load):** P and Q specified
        """)
    
    with st.expander("🧮 Gauss-Seidel Method"):
        st.markdown("""
        The **Gauss-Seidel** method is an iterative algorithm:
        
        **Formula:**
        ```
        V_i^(k+1) = (1/Y_ii) × [(S_i*/V_i^(k)*) - Σ(Y_ij × V_j)]
        ```
        
        Where:
        - `V_i` = Complex voltage at bus i
        - `Y_ij` = Admittance matrix elements
        - `S_i` = Net complex power injection
        - `*` = Complex conjugate
        
        **Steps:**
        1. Build Y-bus matrix from line data
        2. Initialize voltages
        3. Iterate until convergence
        4. Calculate slack bus power
        """)
    
    with st.expander("⚙️ How to Use This App"):
        st.markdown("""
        1. **Enter Bus Data:** Add/edit bus information
        2. **Enter Line Data:** Define network topology
        3. **Set Initial Conditions:** Choose starting values
        4. **Initialize Solver:** Prepare for iterations
        5. **Run Iterations:** Step-by-step or solve all
        6. **View Results:** Analyze convergence and final values
        """)
    
    st.divider()
    
    st.header("⚙️ Settings")
    st.session_state.tolerance = st.number_input(
        "Tolerance", 
        value=0.0001, 
        format="%.6f",
        min_value=0.000001,
        max_value=0.01,
        help="Convergence criterion for voltage mismatch"
    )
    
    st.session_state.max_iterations = st.number_input(
        "Max Iterations", 
        value=50, 
        min_value=10,
        max_value=200,
        help="Maximum number of iterations before stopping"
    )

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Input", 
    "🎮 Solver Control", 
    "📈 Results & Iterations",
    "📄 Final Report"
])

# ============================================================================
# TAB 1: DATA INPUT
# ============================================================================

with tab1:
    st.header("📊 System Data Input")
    
    # Bus Data Section
    st.subheader("1️⃣ Bus Data")
    
    st.markdown("""
    <div class="info-box">
    <strong>Bus Types:</strong> 1 = Slack, 2 = PV (Generator), 3 = PQ (Load)<br>
    <strong>Units:</strong> Voltage in p.u., Angle in degrees, Power in p.u.
    </div>
    """, unsafe_allow_html=True)
    
    # Display bus data editor
    edited_bus_data = st.data_editor(
        st.session_state.bus_data,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Bus": st.column_config.NumberColumn("Bus #", min_value=1, max_value=100),
            "Type": st.column_config.SelectboxColumn(
                "Type",
                options=[1, 2, 3],
                help="1=Slack, 2=PV, 3=PQ"
            ),
            "V": st.column_config.NumberColumn("V (p.u.)", min_value=0.8, max_value=1.2, format="%.4f"),
            "Angle": st.column_config.NumberColumn("Angle (°)", format="%.2f"),
            "P_Gen": st.column_config.NumberColumn("P_Gen (p.u.)", format="%.4f"),
            "Q_Gen": st.column_config.NumberColumn("Q_Gen (p.u.)", format="%.4f"),
            "P_Load": st.column_config.NumberColumn("P_Load (p.u.)", format="%.4f"),
            "Q_Load": st.column_config.NumberColumn("Q_Load (p.u.)", format="%.4f"),
        }
    )
    st.session_state.bus_data = edited_bus_data
    
    st.divider()
    
    # Line Data Section
    st.subheader("2️⃣ Line (Branch) Data")
    
    st.markdown("""
    <div class="info-box">
    <strong>Format:</strong> From bus → To bus with series impedance R + jX<br>
    <strong>Units:</strong> R and X in p.u.
    </div>
    """, unsafe_allow_html=True)
    
    # Display line data editor
    edited_line_data = st.data_editor(
        st.session_state.line_data,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "From": st.column_config.NumberColumn("From Bus", min_value=1),
            "To": st.column_config.NumberColumn("To Bus", min_value=1),
            "R": st.column_config.NumberColumn("R (p.u.)", format="%.4f"),
            "X": st.column_config.NumberColumn("X (p.u.)", format="%.4f"),
        }
    )
    st.session_state.line_data = edited_line_data
    
    st.divider()
    
    # Initial Conditions Section
    st.subheader("3️⃣ Initial Conditions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**PQ Bus Initial Values**")
        for i, bus in st.session_state.bus_data.iterrows():
            if bus['Type'] == 3:  # PQ bus
                st.markdown(f"**Bus {bus['Bus']}:**")
                st.number_input(
                    f"Initial |V| (p.u.) - Bus {bus['Bus']}",
                    value=1.0,
                    key=f"pq_mag_{i}",
                    format="%.4f"
                )
                st.number_input(
                    f"Initial δ (degrees) - Bus {bus['Bus']}",
                    value=0.0,
                    key=f"pq_angle_{i}",
                    format="%.2f"
                )
    
    with col2:
        st.markdown("**PV Bus Initial Values**")
        for i, bus in st.session_state.bus_data.iterrows():
            if bus['Type'] == 2:  # PV bus
                st.markdown(f"**Bus {bus['Bus']}:**")
                st.number_input(
                    f"Initial δ (degrees) - Bus {bus['Bus']}",
                    value=0.0,
                    key=f"pv_angle_{i}",
                    format="%.2f"
                )
                st.info(f"Voltage magnitude fixed at {bus['V']:.4f} p.u.")

# ============================================================================
# TAB 2: SOLVER CONTROL
# ============================================================================

with tab2:
    st.header("🎮 Solver Control Panel")
    
    # Initialize button
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔧 Initialize Solver", use_container_width=True, type="primary"):
            # Build Y-bus
            st.session_state.Y_bus = build_ybus(
                st.session_state.bus_data, 
                st.session_state.line_data
            )
            
            # Calculate net power
            st.session_state.S_net = calculate_net_power(st.session_state.bus_data)
            
            # Initialize voltages
            initial_settings = {key: st.session_state[key] for key in st.session_state.keys() 
                              if key.startswith(('pq_mag_', 'pq_angle_', 'pv_angle_'))}
            st.session_state.V_current = initialize_voltages(
                st.session_state.bus_data,
                initial_settings
            )
            
            # Reset iteration counter
            st.session_state.current_iteration = 0
            st.session_state.iteration_history = []
            st.session_state.is_converged = False
            
            st.success("✅ Solver initialized successfully!")
            st.rerun()
    
    with col2:
        if st.button("▶️ Next Iteration", use_container_width=True, 
                     disabled=(st.session_state.V_current is None or st.session_state.is_converged)):
            # Run one iteration
            V_new, calc_details = gauss_seidel_iteration(
                st.session_state.V_current,
                st.session_state.Y_bus,
                st.session_state.S_net,
                st.session_state.bus_data
            )
            
            # Calculate max error
            max_error = max([d['error'] for d in calc_details])
            
            # Update iteration counter
            st.session_state.current_iteration += 1
            
            # Store iteration history
            st.session_state.iteration_history.append({
                'iteration': st.session_state.current_iteration,
                'V': V_new.copy(),
                'details': calc_details,
                'max_error': max_error
            })
            
            # Update current voltage
            st.session_state.V_current = V_new
            
            # Check convergence
            if max_error < st.session_state.tolerance:
                st.session_state.is_converged = True
                st.success(f"🎉 Converged in {st.session_state.current_iteration} iterations!")
            
            st.rerun()
    
    with col3:
        if st.button("⏩ Solve All", use_container_width=True,
                     disabled=(st.session_state.V_current is None or st.session_state.is_converged)):
            # Run iterations until convergence
            iteration_count = 0
            while iteration_count < st.session_state.max_iterations:
                V_new, calc_details = gauss_seidel_iteration(
                    st.session_state.V_current,
                    st.session_state.Y_bus,
                    st.session_state.S_net,
                    st.session_state.bus_data
                )
                
                max_error = max([d['error'] for d in calc_details])
                st.session_state.current_iteration += 1
                iteration_count += 1
                
                st.session_state.iteration_history.append({
                    'iteration': st.session_state.current_iteration,
                    'V': V_new.copy(),
                    'details': calc_details,
                    'max_error': max_error
                })
                
                st.session_state.V_current = V_new
                
                if max_error < st.session_state.tolerance:
                    st.session_state.is_converged = True
                    st.success(f"🎉 Converged in {st.session_state.current_iteration} iterations!")
                    break
            
            if not st.session_state.is_converged:
                st.warning(f"⚠️ Did not converge in {st.session_state.max_iterations} iterations")
            
            st.rerun()
    
    st.divider()
    
    # Reset button
    if st.button("🔄 Reset Solver", use_container_width=True):
        st.session_state.V_current = None
        st.session_state.current_iteration = 0
        st.session_state.iteration_history = []
        st.session_state.is_converged = False
        st.session_state.Y_bus = None
        st.session_state.S_net = None
        st.success("Reset complete. Configure data and initialize solver.")
        st.rerun()
    
    st.divider()
    
    # Display status
    if st.session_state.V_current is not None:
        st.subheader("📊 Current Status")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Iteration", st.session_state.current_iteration)
        with col2:
            if st.session_state.iteration_history:
                last_error = st.session_state.iteration_history[-1]['max_error']
                st.metric("Max Error", f"{last_error:.8f}")
            else:
                st.metric("Max Error", "N/A")
        with col3:
            st.metric("Status", "Converged ✅" if st.session_state.is_converged else "Running ⏳")
        
        # Display Y-bus matrix
        st.subheader("Y-bus Matrix")
        with st.expander("View Y-bus Matrix"):
            num_buses = len(st.session_state.bus_data)
            ybus_display = []
            for i in range(num_buses):
                row = {}
                for j in range(num_buses):
                    row[f'Y[{i+1},{j+1}]'] = complex_to_string(st.session_state.Y_bus[i, j])
                ybus_display.append(row)
            st.dataframe(pd.DataFrame(ybus_display), use_container_width=True)

# ============================================================================
# TAB 3: RESULTS & ITERATIONS
# ============================================================================

with tab3:
    st.header("📈 Iteration Results")
    
    if not st.session_state.iteration_history:
        st.info("No iterations yet. Initialize solver and run iterations in the Solver Control panel.")
    else:
        # Display iterations (newest first)
        for iter_data in reversed(st.session_state.iteration_history):
            with st.expander(
                f"🔄 Iteration {iter_data['iteration']} - Max Error: {iter_data['max_error']:.8f}",
                expanded=(iter_data['iteration'] == st.session_state.current_iteration)
            ):
                st.markdown(f"### Iteration {iter_data['iteration']}")
                
                for detail in iter_data['details']:
                    bus_num = detail['bus']
                    bus_type = detail['type']
                    
                    st.markdown(f"#### Bus {bus_num} ({bus_type})")
                    
                    if detail['fixed']:
                        st.info(f"Slack bus - Voltage fixed at {complex_to_string(detail['V_new'])}")
                    else:
                        # Show detailed calculations
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.markdown("**Previous Value:**")
                            st.code(f"V[{bus_num}](old) = {complex_to_string(detail['V_old'])}")
                            st.code(f"|V| = {abs(detail['V_old']):.6f}, δ = {np.degrees(cmath.phase(detail['V_old'])):.4f}°")
                        
                        with col2:
                            st.markdown("**New Value:**")
                            st.code(f"V[{bus_num}](new) = {complex_to_string(detail['V_new'])}")
                            st.code(f"|V| = {abs(detail['V_new']):.6f}, δ = {np.degrees(cmath.phase(detail['V_new'])):.4f}°")
                        
                        st.markdown("**Step 1: Power Term**")
                        st.latex(r"S_i^* / V_i^{(old)*}")
                        st.code(f"S_net[{bus_num}] = {complex_to_string(detail['S_net'])}")
                        st.code(f"S_net[{bus_num}]* = {complex_to_string(detail['S_conj'])}")
                        st.code(f"V[{bus_num}](old)* = {complex_to_string(detail['V_conj'])}")
                        st.code(f"Power Term = {complex_to_string(detail['power_term'])}")
                        
                        st.markdown("**Step 2: Neighbor Sum**")
                        st.latex(r"\sum_{j \neq i} Y_{ij} \times V_j")
                        
                        # Display neighbor contributions in a table
                        if detail['sum_terms']:
                            neighbor_df = pd.DataFrame([
                                {
                                    'Bus j': term['j'],
                                    'Y_ij': complex_to_string(term['Y_ij']),
                                    'V_j': complex_to_string(term['V_j']),
                                    'Y_ij × V_j': complex_to_string(term['product'])
                                }
                                for term in detail['sum_terms']
                            ])
                            st.dataframe(neighbor_df, use_container_width=True)
                        
                        st.code(f"Neighbor Sum = {complex_to_string(detail['sum_YV'])}")
                        
                        st.markdown("**Step 3: Calculate New Voltage**")
                        st.latex(r"V_i^{(new)} = \frac{1}{Y_{ii}} \times [PowerTerm - NeighborSum]")
                        st.code(f"Y[{bus_num},{bus_num}] = {complex_to_string(detail['Y_ii'])}")
                        st.code(f"Bracket = {complex_to_string(detail['power_term'])} - {complex_to_string(detail['sum_YV'])}")
                        st.code(f"V[{bus_num}](new) = {complex_to_string(detail['V_new'])}")
                        
                        if detail['target_mag'] is not None:
                            st.markdown(f"**PV Bus Correction:** Magnitude adjusted to {detail['target_mag']:.4f} p.u.")
                        
                        st.markdown(f"**Error:** {detail['error']:.8f}")
                    
                    st.divider()
                
                # Iteration summary
                st.markdown("### Iteration Summary")
                summary_data = []
                for detail in iter_data['details']:
                    summary_data.append({
                        'Bus': detail['bus'],
                        'Type': detail['type'],
                        'V (Rectangular)': complex_to_string(detail['V_new']),
                        '|V| (p.u.)': f"{abs(detail['V_new']):.6f}",
                        'δ (degrees)': f"{np.degrees(cmath.phase(detail['V_new'])):.4f}",
                        'Error': f"{detail['error']:.8f}"
                    })
                
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                
                st.markdown(f"**Maximum Error:** {iter_data['max_error']:.8f}")
                st.markdown(f"**Tolerance:** {st.session_state.tolerance}")
                
                if iter_data['max_error'] < st.session_state.tolerance:
                    st.success("✅ CONVERGED")
                else:
                    st.warning("⏳ NOT CONVERGED")

# ============================================================================
# TAB 4: FINAL REPORT
# ============================================================================

with tab4:
    st.header("📄 Final Report")
    
    if not st.session_state.is_converged:
        st.warning("⚠️ Solution has not converged yet. Run iterations until convergence.")
    else:
        st.success(f"✅ Solution converged in {st.session_state.current_iteration} iterations")
        
        # Final voltage profile
        st.subheader("🔌 Final Voltage Profile")
        
        voltage_profile = []
        for i, bus in st.session_state.bus_data.iterrows():
            V = st.session_state.V_current[i]
            voltage_profile.append({
                'Bus': bus['Bus'],
                'Type': ['Slack', 'PV', 'PQ'][bus['Type'] - 1],
                'V (Rectangular)': complex_to_string(V),
                '|V| (p.u.)': f"{abs(V):.6f}",
                'δ (degrees)': f"{np.degrees(cmath.phase(V)):.4f}"
            })
        
        st.dataframe(pd.DataFrame(voltage_profile), use_container_width=True)
        
        st.divider()
        
        # Slack bus power calculation
        st.subheader("⚡ Slack Bus Power")
        
        # Find slack bus
        slack_idx = st.session_state.bus_data[st.session_state.bus_data['Type'] == 1].index[0]
        slack_bus_num = st.session_state.bus_data.iloc[slack_idx]['Bus']
        
        P_slack, Q_slack = calculate_slack_bus_power(
            slack_idx,
            st.session_state.V_current,
            st.session_state.Y_bus
        )
        
        st.markdown("**Calculation:**")
        st.latex(r"S_{slack} = V_{slack} \times I_{slack}^*")
        st.latex(r"I_{slack} = \sum_{j=1}^{n} Y_{slack,j} \times V_j")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Active Power (P)", f"{P_slack:.6f} p.u.")
        with col2:
            st.metric("Reactive Power (Q)", f"{Q_slack:.6f} p.u.")
        
        st.info(f"Slack Bus {slack_bus_num} supplies P = {P_slack:.6f} p.u. and Q = {Q_slack:.6f} p.u.")
        
        st.divider()
        
        # System summary
        st.subheader("📊 System Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**System Statistics:**")
            st.write(f"Number of Buses: {len(st.session_state.bus_data)}")
            st.write(f"Number of Lines: {len(st.session_state.line_data)}")
            st.write(f"Convergence Tolerance: {st.session_state.tolerance}")
            st.write(f"Iterations to Convergence: {st.session_state.current_iteration}")
        
        with col2:
            st.markdown("**Power Summary:**")
            total_gen_P = st.session_state.bus_data['P_Gen'].sum() + P_slack
            total_gen_Q = st.session_state.bus_data['Q_Gen'].sum() + Q_slack
            total_load_P = st.session_state.bus_data['P_Load'].sum()
            total_load_Q = st.session_state.bus_data['Q_Load'].sum()
            
            st.write(f"Total Generation P: {total_gen_P:.6f} p.u.")
            st.write(f"Total Generation Q: {total_gen_Q:.6f} p.u.")
            st.write(f"Total Load P: {total_load_P:.6f} p.u.")
            st.write(f"Total Load Q: {total_load_Q:.6f} p.u.")
            st.write(f"Losses P: {(total_gen_P - total_load_P):.6f} p.u.")
            st.write(f"Losses Q: {(total_gen_Q - total_load_Q):.6f} p.u.")

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>⚡ Gauss-Seidel Power Flow Solver | Educational Tool for Power Systems Analysis</p>
        <p>Built with Streamlit | Free and Open Source</p>
    </div>
    """, unsafe_allow_html=True)
