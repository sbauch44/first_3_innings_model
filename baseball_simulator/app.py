from datetime import date

import streamlit as st

import storage

st.title("Baseball Inning Simulator Results")

selected_date = st.date_input("Select Date", date.today())
date_str = selected_date.strftime("%Y-%m-%d")

if st.button("Load Results"):
    st.write(f"Loading results for {date_str}...")
    results_df = storage.load_simulation_results_for_date(date_str) # Loads combined results for the day
    if results_df is not None and not results_df.is_empty():
        st.write("Simulation Probabilities:")
        # Display formatted (maybe pivot?) DataFrame
        st.dataframe(results_df)
    else:
        st.write("No simulation results found for this date.")
