import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import streamlit as st
from typing import Dict, Tuple, List

# Set random seed for reproducibility
np.random.seed(42)

# Original simulation functions (keep these unchanged)
def simulate_ifco_cycles(
    initial_pool_size: int = 10000,
    true_shrinkage_rate: float = 0.05,
    mean_trip_duration: int = 100,
    simulation_days: int = 1825,  # 5 years
    trips_per_day: int = 10,
    return_window_factor: float = 3.0
) -> Dict:
    """
    Simulate the IFCO Smart Cycle for RPCs (Reusable Packaging Containers)
    """
    # Calculate total trips in simulation
    total_trips = trips_per_day * simulation_days
    
    # Simulate trip durations (exponential distribution around mean_trip_duration)
    trip_durations = np.random.exponential(scale=mean_trip_duration, size=total_trips)
    
    # Simulate if trip is lost (shrinkage) - Bernoulli trial
    lost_flags = np.random.binomial(1, true_shrinkage_rate, total_trips)
    
    # Rental dates (days from start)
    rental_dates = np.arange(total_trips) // trips_per_day
    
    # Return dates for non-lost trips
    return_dates = rental_dates + trip_durations
    return_dates[lost_flags == 1] = np.nan  # lost trips never return
    
    # Create DataFrame for easier manipulation
    trips_df = pd.DataFrame({
        'rental_date': rental_dates,
        'trip_duration': trip_durations,
        'return_date': return_dates,
        'is_lost': lost_flags,
    })
    
    # Track pool size over time
    daily_stats = []
    
    # Initialize pool size
    pool_size = initial_pool_size
    
    # Track cumulative metrics
    cumulative_trips = 0
    cumulative_returns = 0
    cumulative_lost = 0
    
    # For each day in the simulation
    for day in range(simulation_days):
        # Count new trips starting today
        new_trips = len(trips_df[trips_df['rental_date'] == day])
        cumulative_trips += new_trips
        
        # Count returns today
        returns_today = len(trips_df[(trips_df['return_date'] >= day) & (trips_df['return_date'] < day + 1)])
        cumulative_returns += returns_today
        
        # Count trips that we know are lost
        known_lost_today = len(trips_df[(trips_df['rental_date'] + return_window_factor * mean_trip_duration < day) & 
                                      (trips_df['is_lost'] == 1) & 
                                      (trips_df['rental_date'] <= day)])
        
        # Update pool size
        pool_size = initial_pool_size + cumulative_returns - cumulative_trips
        
        # Estimated shrinkage rate based on information available up to this day
        if cumulative_trips > 0:
            estimated_shrinkage = sum(trips_df[(trips_df['rental_date'] + return_window_factor * mean_trip_duration < day) & 
                                         (trips_df['is_lost'] == 1)]['is_lost']) / cumulative_trips
        else:
            estimated_shrinkage = 0
            
        # Store daily statistics
        daily_stats.append({
            'day': day,
            'new_trips': new_trips,
            'returns': returns_today,
            'known_lost': known_lost_today,
            'cumulative_trips': cumulative_trips,
            'cumulative_returns': cumulative_returns,
            'pool_size': pool_size,
            'estimated_shrinkage_rate': estimated_shrinkage
        })
    
    # Convert to DataFrame
    daily_df = pd.DataFrame(daily_stats)
    
    # Calculate overall shrinkage rate estimate
    # Define a cutoff date to ensure all non-lost RPCs have had time to return
    cutoff_date = simulation_days - return_window_factor * mean_trip_duration
    
    # Trips that started before the cutoff
    trips_before_cutoff = trips_df[trips_df['rental_date'] < cutoff_date]
    
    # Count total trips and lost trips before cutoff
    total_trips_before_cutoff = len(trips_before_cutoff)
    lost_trips_before_cutoff = sum(trips_before_cutoff['is_lost'])
    
    # Calculate estimated shrinkage rate
    if total_trips_before_cutoff > 0:
        estimated_shrinkage_rate = lost_trips_before_cutoff / total_trips_before_cutoff
    else:
        estimated_shrinkage_rate = 0
    
    # Calculate final pool size
    final_pool_size = initial_pool_size + len(trips_df[~np.isnan(trips_df['return_date'])]) - total_trips
    
    # Return results
    return {
        'true_shrinkage_rate': true_shrinkage_rate,
        'estimated_shrinkage_rate': estimated_shrinkage_rate,
        'initial_pool_size': initial_pool_size,
        'total_trips': total_trips,
        'final_pool_size': final_pool_size,
        'trips_df': trips_df,
        'daily_stats': daily_df
    }

def simulate_partial_visibility(
    full_simulation_results: Dict,
    observed_fraction: float = 0.3,
    confidence_level: float = 0.95
) -> Dict:
    """
    Simulate scenario two where only a fraction of rental dates are observed
    """
    trips_df = full_simulation_results['trips_df']
    total_trips = len(trips_df)
    observed_trips = int(total_trips * observed_fraction)
    
    # Sample from total trips
    sample_indices = np.random.choice(total_trips, observed_trips, replace=False)
    
    # Observed lost flags and rental dates
    observed_trips_df = trips_df.iloc[sample_indices].copy()
    
    # Calculate shrinkage rate from observed data
    cutoff_date = trips_df['rental_date'].max() - 3 * trips_df['trip_duration'].mean()
    observed_trips_before_cutoff = observed_trips_df[observed_trips_df['rental_date'] < cutoff_date]
    
    total_observed_before_cutoff = len(observed_trips_before_cutoff)
    lost_observed_before_cutoff = sum(observed_trips_before_cutoff['is_lost'])
    
    if total_observed_before_cutoff > 0:
        sample_shrinkage_rate = lost_observed_before_cutoff / total_observed_before_cutoff
    else:
        sample_shrinkage_rate = 0
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    
    # Using normal approximation for CI
    p_hat = sample_shrinkage_rate
    z = stats.norm.ppf(1 - alpha/2)
    
    margin_of_error = z * np.sqrt(p_hat * (1 - p_hat) / total_observed_before_cutoff)
    ci_lower = max(0, p_hat - margin_of_error)
    ci_upper = min(1, p_hat + margin_of_error)
    
    # Estimate pool size using sample shrinkage rate
    # Scale up the observed trips to estimate total
    estimated_total_trips = total_observed_before_cutoff / observed_fraction
    
    # Calculate estimated pool size
    initial_pool_size = full_simulation_results['initial_pool_size']
    estimated_pool_size = initial_pool_size - (sample_shrinkage_rate * estimated_total_trips)
    
    # Calculate pool size bounds based on CI
    pool_size_lower_bound = initial_pool_size - (ci_upper * estimated_total_trips)
    pool_size_upper_bound = initial_pool_size - (ci_lower * estimated_total_trips)
    
    return {
        'observed_fraction': observed_fraction,
        'sample_shrinkage_rate': sample_shrinkage_rate,
        'confidence_interval': (ci_lower, ci_upper),
        'estimated_pool_size': estimated_pool_size,
        'pool_size_ci': (pool_size_lower_bound, pool_size_upper_bound),
        'observed_trips_df': observed_trips_df
    }

# Streamlit-adapted visualization functions
def st_plot_trip_durations(full_results: Dict):
    """Plot trip durations histogram using Streamlit"""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(full_results['trips_df']['trip_duration'], kde=True, ax=ax)
    ax.set_title('Distribution of Trip Durations')
    ax.set_xlabel('Duration (days)')
    ax.set_ylabel('Count')
    st.pyplot(fig)

def st_plot_pool_size(full_results: Dict):
    """Plot pool size over time using Streamlit"""
    fig, ax = plt.subplots(figsize=(10, 6))
    daily_stats = full_results['daily_stats']
    ax.plot(daily_stats['day'], daily_stats['pool_size'])
    ax.set_title('Pool Size Over Time')
    ax.set_xlabel('Day')
    ax.set_ylabel('Number of RPCs')
    st.pyplot(fig)

def st_plot_cumulative_metrics(full_results: Dict):
    """Plot cumulative metrics using Streamlit"""
    fig, ax = plt.subplots(figsize=(10, 6))
    daily_stats = full_results['daily_stats']
    ax.plot(daily_stats['day'], daily_stats['cumulative_trips'], label='Total Trips')
    ax.plot(daily_stats['day'], daily_stats['cumulative_returns'], label='Total Returns')
    ax.set_title('Cumulative Metrics')
    ax.set_xlabel('Day')
    ax.set_ylabel('Count')
    ax.legend()
    st.pyplot(fig)

def st_plot_shrinkage_comparison(full_results: Dict, partial_results: Dict = None):
    """Plot shrinkage rate comparison using Streamlit"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # For full visibility
    ax.axhline(y=full_results['true_shrinkage_rate'], color='r', linestyle='-', label='True Shrinkage Rate')
    ax.axhline(y=full_results['estimated_shrinkage_rate'], color='b', linestyle='--', 
              label=f'Full Visibility Estimate: {full_results["estimated_shrinkage_rate"]:.4f}')
    
    # For partial visibility (if provided)
    if partial_results:
        ax.axhline(y=partial_results['sample_shrinkage_rate'], color='g', linestyle=':', 
                  label=f'Partial Visibility Estimate: {partial_results["sample_shrinkage_rate"]:.4f}')
        
        # Add confidence interval
        ci_lower, ci_upper = partial_results['confidence_interval']
        ax.axhspan(ci_lower, ci_upper, alpha=0.2, color='g', 
                  label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
    
    ax.set_title('Shrinkage Rate Estimates')
    ax.set_xlabel('Method')
    ax.set_ylabel('Shrinkage Rate')
    ax.legend()
    st.pyplot(fig)

def st_sensitivity_analysis(
    base_shrinkage_rate: float = 0.05,
    shrinkage_range: List[float] = None,
    initial_pool_size: int = 10000,
    mean_trip_duration: int = 100,
    simulation_days: int = 1825,
    trips_per_day: int = 10
):
    """Perform sensitivity analysis for different shrinkage rates and display in Streamlit"""
    if shrinkage_range is None:
        shrinkage_range = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15]
    
    results = []
    progress_bar = st.progress(0)
    
    for i, shrinkage_rate in enumerate(shrinkage_range):
        sim_result = simulate_ifco_cycles(
            initial_pool_size=initial_pool_size,
            true_shrinkage_rate=shrinkage_rate,
            mean_trip_duration=mean_trip_duration,
            simulation_days=simulation_days,
            trips_per_day=trips_per_day
        )
        
        results.append({
            'shrinkage_rate': shrinkage_rate,
            'final_pool_size': sim_result['final_pool_size'],
            'estimated_shrinkage': sim_result['estimated_shrinkage_rate']
        })
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(shrinkage_range))
    
    # Convert to DataFrame
    sensitivity_df = pd.DataFrame(results)
    
    # Display the dataframe
    st.subheader("Sensitivity Analysis Results")
    st.dataframe(sensitivity_df)
    
    # Plot results - Pool size vs shrinkage rate
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(sensitivity_df['shrinkage_rate'], sensitivity_df['final_pool_size'], 'o-')
    ax1.set_title('Impact of Shrinkage Rate on Pool Size')
    ax1.set_xlabel('True Shrinkage Rate')
    ax1.set_ylabel('Final Pool Size')
    ax1.grid(True)
    st.pyplot(fig1)
    
    # Plot results - Estimation accuracy
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(sensitivity_df['shrinkage_rate'], sensitivity_df['estimated_shrinkage'], 'o-', label='Estimated')
    ax2.plot(sensitivity_df['shrinkage_rate'], sensitivity_df['shrinkage_rate'], 'r--', label='True Value')
    ax2.set_title('Shrinkage Rate Estimation Accuracy')
    ax2.set_xlabel('True Shrinkage Rate')
    ax2.set_ylabel('Estimated Shrinkage Rate')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)
    
    return sensitivity_df

def st_observation_fraction_impact(
    full_sim_results: Dict,
    fractions: List[float] = None
):
    """Analyze the impact of different observation fractions and display in Streamlit"""
    if fractions is None:
        fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    
    results = []
    progress_bar = st.progress(0)
    
    for i, fraction in enumerate(fractions):
        partial_result = simulate_partial_visibility(
            full_sim_results,
            observed_fraction=fraction
        )
        
        ci_lower, ci_upper = partial_result['confidence_interval']
        ci_width = ci_upper - ci_lower
        
        results.append({
            'observed_fraction': fraction,
            'estimated_shrinkage': partial_result['sample_shrinkage_rate'],
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_width,
            'true_rate_in_ci': (ci_lower <= full_sim_results['true_shrinkage_rate'] <= ci_upper)
        })
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(fractions))
    
    # Convert to DataFrame
    fraction_df = pd.DataFrame(results)
    
    # Display the dataframe
    st.subheader("Impact of Observation Fraction")
    st.dataframe(fraction_df)
    
    # Plot shrinkage estimate vs fraction
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.errorbar(
        fraction_df['observed_fraction'], 
        fraction_df['estimated_shrinkage'],
        yerr=[fraction_df['estimated_shrinkage'] - fraction_df['ci_lower'], 
              fraction_df['ci_upper'] - fraction_df['estimated_shrinkage']],
        fmt='o-', 
        capsize=5
    )
    ax1.axhline(y=full_sim_results['true_shrinkage_rate'], color='r', linestyle='--', label='True Rate')
    ax1.set_title('Shrinkage Rate Estimates with Confidence Intervals')
    ax1.set_xlabel('Fraction of Observed Trips')
    ax1.set_ylabel('Estimated Shrinkage Rate')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)
    
    # Plot CI width vs fraction
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(fraction_df['observed_fraction'], fraction_df['ci_width'], 'o-')
    ax2.set_title('Confidence Interval Width')
    ax2.set_xlabel('Fraction of Observed Trips')
    ax2.set_ylabel('95% CI Width')
    ax2.grid(True)
    st.pyplot(fig2)
    
    return fraction_df

# Main Streamlit app
def main():
    st.set_page_config(page_title="IFCO Smart Cycle Simulation", layout="wide")
    
    st.title("IFCO Smart Cycle Simulation")
    st.markdown("""
    This app simulates the movement of IFCO's Reusable Packaging Containers (RPCs) through the food supply chain.
    Analyze shrinkage rates and pool sizes under different scenarios.
    """)
    
    # Add image of the SmartCycle
    st.image("IFCO_smart_cycle_image.png", 
             caption="IFCO Smart Cycle", use_column_width=True)

    # Sidebar for inputs
    st.sidebar.header("Simulation Parameters")
    
    # Create tabs for scenarios
    tab1, tab2, tab3 = st.tabs(["Scenario One: 100% Visibility", 
                               "Scenario Two: Partial Visibility", 
                               "Sensitivity Analysis"])
    
    # Common parameters
    initial_pool_size = st.sidebar.number_input("Initial Pool Size", value=10000, min_value=1000, max_value=100000)
    true_shrinkage_rate = st.sidebar.slider("True Shrinkage Rate", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    mean_trip_duration = st.sidebar.number_input("Mean Trip Duration (days)", value=100, min_value=10, max_value=365)
    simulation_days = st.sidebar.number_input("Simulation Days", value=1825, min_value=365, max_value=3650)
    trips_per_day = st.sidebar.number_input("Trips Per Day", value=10, min_value=1, max_value=100)
    
    # Scenario One tab
    with tab1:
        st.header("Scenario One: 100% Visibility")
        st.markdown("""
        In this scenario, we assume we can measure the rental date for 100% of trips.
        """)
        
        if st.button("Run Scenario One Simulation"):
            with st.spinner("Running simulation..."):
                # Run the simulation with 100% visibility
                full_sim_results = simulate_ifco_cycles(
                    initial_pool_size=initial_pool_size,
                    true_shrinkage_rate=true_shrinkage_rate,
                    mean_trip_duration=mean_trip_duration,
                    simulation_days=simulation_days,
                    trips_per_day=trips_per_day
                )
                
                # Store results in session state to use in other tabs
                st.session_state.full_sim_results = full_sim_results
                
                # Display key metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("True Shrinkage Rate", f"{full_sim_results['true_shrinkage_rate']:.1%}")
                with col2:
                    st.metric("Estimated Shrinkage Rate", f"{full_sim_results['estimated_shrinkage_rate']:.1%}")
                with col3:
                    st.metric("Final Pool Size", f"{int(full_sim_results['final_pool_size']):,}")
                
                # Display visualizations
                st.subheader("Trip Duration Distribution")
                st_plot_trip_durations(full_sim_results)
                
                st.subheader("Pool Size Over Time")
                st_plot_pool_size(full_sim_results)
                
                st.subheader("Cumulative Metrics")
                st_plot_cumulative_metrics(full_sim_results)
                
                st.subheader("Shrinkage Rate Estimation")
                st_plot_shrinkage_comparison(full_sim_results)
    
    # Scenario Two tab
    with tab2:
        st.header("Scenario Two: Partial Visibility")
        st.markdown("""
        In this scenario, we can only measure the rental date for a percentage of trips, substantially smaller than 100%.
        """)
        
        # Additional parameter for scenario two
        observed_fraction = st.slider("Observed Fraction of Trips", min_value=0.1, max_value=0.9, value=0.3, step=0.1)
        
        if st.button("Run Scenario Two Simulation"):
            # Check if full simulation results exist
            if 'full_sim_results' not in st.session_state:
                st.warning("Please run Scenario One simulation first.")
            else:
                with st.spinner("Running simulation..."):
                    # Run the partial visibility simulation
                    partial_sim_results = simulate_partial_visibility(
                        st.session_state.full_sim_results,
                        observed_fraction=observed_fraction
                    )
                    
                    # Display key metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("True Shrinkage Rate", f"{st.session_state.full_sim_results['true_shrinkage_rate']:.1%}")
                    with col2:
                        st.metric("Estimated Shrinkage Rate", f"{partial_sim_results['sample_shrinkage_rate']:.1%}")
                    with col3:
                        ci_lower, ci_upper = partial_sim_results['confidence_interval']
                        st.metric("95% Confidence Interval", f"[{ci_lower:.1%}, {ci_upper:.1%}]")
                    
                    # Show shrinkage rate comparison
                    st.subheader("Shrinkage Rate Estimation")
                    st_plot_shrinkage_comparison(st.session_state.full_sim_results, partial_sim_results)
                    
                    # Analyze impact of different observation fractions
                    st.subheader("Impact of Observation Fraction")
                    observation_results = st_observation_fraction_impact(st.session_state.full_sim_results)
    
    # Sensitivity Analysis tab
    with tab3:
        st.header("Sensitivity Analysis")
        st.markdown("""
        This analysis tests how different shrinkage rates affect the pool size and estimation accuracy.
        """)
        
        if st.button("Run Sensitivity Analysis"):
            with st.spinner("Running analysis..."):
                # Define shrinkage range
                shrinkage_range = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15]
                
                # Run sensitivity analysis
                sensitivity_results = st_sensitivity_analysis(
                    base_shrinkage_rate=true_shrinkage_rate,
                    shrinkage_range=shrinkage_range,
                    initial_pool_size=initial_pool_size,
                    mean_trip_duration=mean_trip_duration,
                    simulation_days=simulation_days,
                    trips_per_day=trips_per_day
                )

    # Documentation section
    st.sidebar.header("Documentation")
    st.sidebar.markdown("""
    ### Key Terms:
    - **Shrinkage Rate**: Probability an RPC is lost during a trip.
    - **Pool Size**: Number of RPCs available at a given time.
    - **Trip**: Movement of an RPC from when it leaves a Service Center until it returns.
    
    ### Assumptions:
    - Trip durations follow an exponential distribution with specified mean.
    - All non-lost RPCs eventually return.
    - Trips/losses occur randomly and independently.
    """)

if __name__ == "__main__":
    main()
