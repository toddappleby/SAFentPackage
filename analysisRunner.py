#!/usr/bin/env python
"""
Example Usage of Enhanced MedPC Analysis

This script demonstrates how to use the enhanced MedPC analysis framework
with example data files.
"""

import os
from pathlib import Path
from enhanced_medpc_analysis import EnhancedMedPCAnalyzer

def main():
    """Run example analysis on MedPC data."""
    # Set up paths
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    data_dir = current_dir / "data"
    output_dir = current_dir / "analysis_results"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Looking for MedPC data in: {data_dir}")
    print(f"Saving results to: {output_dir}")
    
    # Initialize the enhanced analyzer
    analyzer = EnhancedMedPCAnalyzer(data_dir, output_dir)
    
    # List available dates
    dates = analyzer.list_available_dates()
    print("Available experiment dates:")
    for date in dates:
        print(f"  {date}")
    
    # Define time segments to analyze (in minutes)
    # Note: The analyzer will automatically handle different session lengths between phases
    # SelfAdmin and EXT are 180 minutes, while REI is 60 minutes
    time_segments = [
        (0, 60),   # First hour (common to all phases)
       # (0, 30),   # First 30 minutes
        #(30, 60),  # Second 30 minutes
       # (0, 180),  # Full SelfAdmin/EXT session
       # (60, 120), # Second hour of SelfAdmin/EXT
        #(120, 180) # Third hour of SelfAdmin/EXT
    ]
    
    # Run enhanced analysis for all data
    analyzer.run_enhanced_analysis(time_segments=time_segments)
    
    # Example: Run analysis for specific dates and subjects
    # Uncomment and modify as needed
    """
    # Specific dates
    selected_dates = ['2025-04-14']
    
    # Specific subjects
    selected_subjects = [83]
    
    # Run targeted analysis
    analyzer.run_enhanced_analysis(
        time_segments=time_segments,
        selected_dates=selected_dates,
        selected_subjects=selected_subjects
    )
    """
    
    print("\nAnalysis complete! Results saved to:", output_dir)

if __name__ == "__main__":
    main()