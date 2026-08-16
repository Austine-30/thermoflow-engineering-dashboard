"""
AI DEVELOPMENT DOCUMENTATION

AI tool used:
1. ChatGPT – used for application planning, Python/Streamlit code generation,
   debugging assistance, and engineering logic review.

Key prompts used:
1. "Design a Streamlit thermodynamics engineering application suitable for a
   Petroleum Engineering student and include interactive engineering inputs."
2. "Build a Peng-Robinson equation-of-state calculator in Python that calculates
   the compressibility factor of real gases."
3. "Add Plotly graphs, Pandas results tables, input validation and user-friendly
   error handling to the Streamlit thermodynamics application."

Manual verification and fixes:
The engineering equations, units, critical properties, calculated results,
input limits, and error-handling behaviour were manually reviewed and tested.
The application was also checked using different pressures, temperatures,
and gases to ensure that the results were physically reasonable.
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="ThermoFlow Engineering Dashboard",
    page_icon="⚙️",
    layout="wide",
)


# ---------------------------------------------------------
# GAS DATABASE
# Critical properties:
# Tc = critical temperature [K]
# Pc = critical pressure [bar]
# omega = acentric factor [-]
# MW = molecular weight [g/mol]
# ---------------------------------------------------------

GASES = {
    "Methane": {
        "Tc": 190.56,
        "Pc": 45.99,
        "omega": 0.01142,
        "MW": 16.043,
    },
    "Ethane": {
        "Tc": 305.32,
        "Pc": 48.72,
        "omega": 0.0995,
        "MW": 30.070,
    },
    "Propane": {
        "Tc": 369.83,
        "Pc": 42.48,
        "omega": 0.1523,
        "MW": 44.097,
    },
    "Carbon Dioxide": {
        "Tc": 304.13,
        "Pc": 73.77,
        "omega": 0.22394,
        "MW": 44.010,
    },
}


# Universal gas constant
# L·bar/(mol·K)
R = 0.08314462618


# ---------------------------------------------------------
# PENG-ROBINSON FUNCTIONS
# ---------------------------------------------------------

def peng_robinson_parameters(T, P, gas):
    """
    Calculate Peng-Robinson EOS parameters A and B.
    """

    props = GASES[gas]

    Tc = props["Tc"]
    Pc = props["Pc"]
    omega = props["omega"]

    Tr = T / Tc

    kappa = (
        0.37464
        + 1.54226 * omega
        - 0.26992 * omega**2
    )

    alpha = (
        1
        + kappa * (1 - math.sqrt(Tr))
    ) ** 2

    a = (
        0.45724
        * (R**2 * Tc**2 / Pc)
        * alpha
    )

    b = 0.07780 * R * Tc / Pc

    A = a * P / (R**2 * T**2)
    B = b * P / (R * T)

    return A, B, a, b


def calculate_z_factor(T, P, gas):
    """
    Solve the Peng-Robinson cubic EOS and return the largest
    real compressibility-factor root, which represents the
    vapour/gas phase.
    """

    A, B, _, _ = peng_robinson_parameters(T, P, gas)

    coefficients = [
        1,
        -(1 - B),
        A - 3 * B**2 - 2 * B,
        -(A * B - B**2 - B**3),
    ]

    roots = np.roots(coefficients)

    real_roots = [
        root.real
        for root in roots
        if abs(root.imag) < 1e-8
    ]

    positive_roots = [
        root
        for root in real_roots
        if root > B
    ]

    if not positive_roots:
        raise ValueError(
            "No physically meaningful compressibility-factor root was found."
        )

    return max(positive_roots)


def fugacity_coefficient(T, P, gas, Z):
    """
    Calculate the fugacity coefficient for a pure gas
    using the Peng-Robinson EOS.
    """

    A, B, _, _ = peng_robinson_parameters(T, P, gas)

    if Z <= B:
        raise ValueError("Invalid Z-factor/B relationship.")

    sqrt2 = math.sqrt(2)

    term1 = Z - 1 - math.log(Z - B)

    term2 = (
        A
        / (2 * sqrt2 * B)
        * math.log(
            (Z + (1 + sqrt2) * B)
            / (Z + (1 - sqrt2) * B)
        )
    )

    ln_phi = term1 - term2

    return math.exp(ln_phi)


def real_gas_density(T, P, gas, Z):
    """
    Calculate gas density using:
        rho = P * MW / (ZRT)

    P in bar
    MW in g/mol
    R in L bar/(mol K)

    Result is converted to kg/m3.
    """

    MW = GASES[gas]["MW"]

    density_g_per_L = P * MW / (Z * R * T)

    # 1 g/L = 1 kg/m3
    return density_g_per_L


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.header("⚙️ Engineering Inputs")

st.sidebar.markdown(
    "Enter the operating conditions and select the gas "
    "to perform real-gas thermodynamic analysis."
)

gas = st.sidebar.selectbox(
    "Select gas",
    list(GASES.keys()),
)

temperature = st.sidebar.slider(
    "Temperature (K)",
    min_value=150.0,
    max_value=700.0,
    value=300.0,
    step=5.0,
)

pressure = st.sidebar.number_input(
    "Operating pressure (bar)",
    min_value=0.1,
    max_value=1000.0,
    value=100.0,
    step=5.0,
)

pressure_range = st.sidebar.slider(
    "Pressure range for analysis (bar)",
    min_value=10.0,
    max_value=1000.0,
    value=(10.0, 500.0),
    step=10.0,
)

number_of_points = st.sidebar.slider(
    "Number of analysis points",
    min_value=10,
    max_value=100,
    value=30,
    step=5,
)

show_properties = st.sidebar.checkbox(
    "Show selected gas properties",
    value=True,
)


# ---------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------

st.title("⚙️ ThermoFlow")
st.subheader("Real-Gas Thermodynamics Engineering Dashboard")

st.write(
    """
    **ThermoFlow** is an interactive thermodynamics application for
    analysing real-gas behaviour using the **Peng–Robinson Equation of State (EOS)**.

    It calculates the compressibility factor, fugacity coefficient,
    gas density and other thermodynamic properties over a selected
    pressure and temperature range.
    """
)

st.info(
    """
    **How to use:** Select a gas from the sidebar, enter the operating
    temperature and pressure, then examine the calculated properties
    and interactive pressure-analysis chart below.
    """
)


# ---------------------------------------------------------
# GAS PROPERTY INFORMATION
# ---------------------------------------------------------

if show_properties:

    props = GASES[gas]

    st.markdown("### Selected Gas Properties")

    property_col1, property_col2, property_col3, property_col4 = st.columns(4)

    property_col1.metric(
        "Critical Temperature",
        f"{props['Tc']:.2f} K",
    )

    property_col2.metric(
        "Critical Pressure",
        f"{props['Pc']:.2f} bar",
    )

    property_col3.metric(
        "Molecular Weight",
        f"{props['MW']:.3f} g/mol",
    )

    property_col4.metric(
        "Acentric Factor",
        f"{props['omega']:.4f}",
    )


# ---------------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------------

if temperature <= 0:
    st.warning(
        "Temperature must be greater than 0 K. "
        "Please enter a valid temperature."
    )
    st.stop()

if pressure <= 0:
    st.warning(
        "Pressure must be greater than 0 bar. "
        "Please enter a valid operating pressure."
    )
    st.stop()

if pressure_range[1] <= pressure_range[0]:
    st.warning(
        "The maximum pressure must be greater than the minimum pressure."
    )
    st.stop()


# ---------------------------------------------------------
# MAIN CALCULATION
# ---------------------------------------------------------

try:

    Z = calculate_z_factor(
        temperature,
        pressure,
        gas,
    )

    phi = fugacity_coefficient(
        temperature,
        pressure,
        gas,
        Z,
    )

    density = real_gas_density(
        temperature,
        pressure,
        gas,
        Z,
    )

    fugacity = phi * pressure

except (ValueError, ZeroDivisionError, OverflowError) as error:

    st.warning(
        f"Unable to complete the calculation for the selected conditions: "
        f"{error}"
    )

    st.stop()


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

st.markdown("### Current Operating Conditions")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Compressibility Factor, Z",
    f"{Z:.4f}",
)

col2.metric(
    "Fugacity Coefficient, φ",
    f"{phi:.4f}",
)

col3.metric(
    "Fugacity",
    f"{fugacity:.2f} bar",
)

col4.metric(
    "Gas Density",
    f"{density:.3f} kg/m³",
)


# ---------------------------------------------------------
# ENGINEERING INTERPRETATION
# ---------------------------------------------------------

st.markdown("### Engineering Interpretation")

if Z < 0.95:
    interpretation = (
        "The Z-factor is below 0.95, indicating noticeable deviation "
        "from ideal-gas behaviour."
    )
elif Z <= 1.05:
    interpretation = (
        "The Z-factor is close to 1, indicating behaviour relatively "
        "close to the ideal-gas assumption."
    )
else:
    interpretation = (
        "The Z-factor is above 1.05, indicating significant deviation "
        "from ideal-gas behaviour."
    )

st.success(interpretation)


# ---------------------------------------------------------
# PRESSURE RANGE ANALYSIS
# ---------------------------------------------------------

st.markdown("### Pressure-Dependent Real-Gas Analysis")

pressures = np.linspace(
    pressure_range[0],
    pressure_range[1],
    number_of_points,
)

records = []

for P in pressures:

    try:

        z_value = calculate_z_factor(
            temperature,
            P,
            gas,
        )

        phi_value = fugacity_coefficient(
            temperature,
            P,
            gas,
            z_value,
        )

        density_value = real_gas_density(
            temperature,
            P,
            gas,
            z_value,
        )

        fugacity_value = phi_value * P

        records.append(
            {
                "Pressure (bar)": P,
                "Temperature (K)": temperature,
                "Z-Factor": z_value,
                "Fugacity Coefficient": phi_value,
                "Fugacity (bar)": fugacity_value,
                "Density (kg/m³)": density_value,
            }
        )

    except (ValueError, ZeroDivisionError, OverflowError):
        continue


if not records:

    st.warning(
        "No valid results could be generated for the selected pressure range."
    )
    st.stop()


results_df = pd.DataFrame(records)


# ---------------------------------------------------------
# PLOTLY CHART
# ---------------------------------------------------------

st.markdown("### Interactive Compressibility-Factor Plot")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=results_df["Pressure (bar)"],
        y=results_df["Z-Factor"],
        mode="lines+markers",
        name="Z-Factor",
        hovertemplate=(
            "Pressure: %{x:.2f} bar"
            "<br>Z-Factor: %{y:.4f}"
            "<extra></extra>"
        ),
    )
)

fig.add_hline(
    y=1.0,
    line_dash="dash",
    annotation_text="Ideal gas: Z = 1",
)

fig.update_layout(
    xaxis_title="Pressure (bar)",
    yaxis_title="Compressibility Factor, Z",
    hovermode="x unified",
    height=500,
)

st.plotly_chart(
    fig,
    width="stretch",
)


# ---------------------------------------------------------
# RESULTS TABLE
# ---------------------------------------------------------

st.markdown("### Detailed Results Table")

st.dataframe(
    results_df.round(4),
    width="stretch",
)


# ---------------------------------------------------------
# ENGINEERING NOTES
# ---------------------------------------------------------

st.markdown("### Engineering Notes")

st.write(
    """
    The Peng–Robinson equation of state is commonly used to describe
    the pressure-volume-temperature behaviour of real fluids, particularly
    hydrocarbons and gases encountered in petroleum engineering.

    The compressibility factor is defined as:

    Z = PV / RT

    For an ideal gas, Z = 1. Deviations from unity indicate non-ideal
    behaviour caused by intermolecular interactions and finite molecular
    volume.

    The application uses the Peng–Robinson EOS to estimate the real-gas
    compressibility factor and related properties.
    """
)

st.caption(
    "ThermoFlow | Petroleum Engineering Thermodynamics Application"
)

