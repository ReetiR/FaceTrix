import streamlit as st
import pandas as pd
from datetime import datetime

st.title("Attendance Viewer")

# Get current timestamp
ts = datetime.now()

# Format date and timestamp
date = ts.strftime("%d_%m_%Y")
timestamp = ts.strftime("%H_%M_%S")

# Construct the file path using an f-string
file_path = f"Attendance/Attendance_{date}.csv"

# Read the CSV file
try:
    df = pd.read_csv(file_path)
    st.write(f"Viewing attendance for: {date}")
    # Display the dataframe with highlighting
    st.dataframe(df.style.highlight_max(axis=0))
except FileNotFoundError:
    st.error(f"File '{file_path}' not found.")

