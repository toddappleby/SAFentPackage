#!/usr/bin/env python
"""
Fix MedPC Analysis Error

This script fixes the column name issue in the enhanced_medpc_analysis.py file
that causes the 'total_active_presses' not found error when running analysisRunner.py.
"""

import os
import re
from pathlib import Path
import sys

def fix_column_names_issue():
    """
    Fix the column name inconsistency in enhanced_medpc_analysis.py that
    causes the 'total_active_presses' column not found error.
    """
    # Path to the file
    file_path = Path("enhanced_medpc_analysis.py")
    
    if not file_path.exists():
        print(f"Error: File {file_path} not found!")
        print("Make sure you're running this script from the directory containing enhanced_medpc_analysis.py")
        return False
    
    # Read the file content
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Make backup of original file
    backup_path = file_path.with_suffix('.py.backup')
    with open(backup_path, 'w') as f:
        f.write(content)
    print(f"Backup saved to {backup_path}")
    
    # Fix 1: Ensure consistent column naming in analyze_time_segment function
    # This fixes a critical bug where the function creates columns with names different
    # from what plot_time_segment_data expects
    
    # Replace the way columns are added to binned_data
    old_pattern = r"binned_data\.append\(\{\s*'subject': subject,\s*'phase': phase,\s*'session': session,\s*'session_date': session_date,\s*'time_segment': f\"{start_min}-{end_min}min\",\s*'active_lever_presses': total_active,\s*'inactive_lever_presses': total_inactive,\s*'reinforcers': total_reinforcers,"
    
    new_pattern = "binned_data.append({\n                        'subject': subject,\n                        'phase': phase,\n                        'session': session,\n                        'session_date': session_date,\n                        'time_segment': f\"{start_min}-{end_min}min\",\n                        'total_active_presses': total_active,\n                        'total_inactive_presses': total_inactive,\n                        'total_reinforcers': total_reinforcers,"
    
    content = re.sub(old_pattern, new_pattern, content)
    
    # Fix 2: Add column existence check in plot_time_segment_data
    # Add error checking to ensure it doesn't fail if columns don't exist
    
    plot_func_pattern = r"def plot_time_segment_data\(self, segment_key, metrics=None, group_by_phase=True\):"
    plot_func_replacement = """def plot_time_segment_data(self, segment_key, metrics=None, group_by_phase=True):
        \"\"\"
        Plot metrics from a specific time segment analysis.
        
        Parameters:
        -----------
        segment_key : str
            Key identifying the time segment (format: "start_end")
        metrics : list
            List of metrics to plot. If None, default metrics are used.
        group_by_phase : bool
            If True, group by phase for plotting
        \"\"\"
        # Check if segment data exists
        if segment_key not in self.time_segment_df:
            start_min, end_min = map(int, segment_key.split('_'))
            segment_df = self.analyze_time_segment(start_min, end_min)
        else:
            segment_df = self.time_segment_df[segment_key]
        
        # Ensure metrics exist in the dataframe
        available_columns = segment_df.columns
        
        # Print available columns for debugging
        print(f"Available columns in segment data: {available_columns}")
        
        # Default metrics
        if metrics is None:
            metrics = []
            # Check which metrics are available
            for metric in ['total_active_presses', 'active_lever_presses', 
                           'total_inactive_presses', 'inactive_lever_presses',
                           'total_reinforcers', 'reinforcers']:
                if metric in available_columns:
                    metrics.append(metric)
                    print(f"Using available metric: {metric}")
            
            if not metrics:
                print(f"Error: No valid metrics found in time segment data!")
                print(f"Available columns: {available_columns}")
                return
        else:
            # Filter provided metrics to only those available
            filtered_metrics = [m for m in metrics if m in available_columns]
            if not filtered_metrics:
                print(f"Error: None of the requested metrics {metrics} found in data!")
                print(f"Available columns: {available_columns}")
                return
            metrics = filtered_metrics"""
    
    content = re.sub(plot_func_pattern, plot_func_replacement, content)
    
    # Write the modified content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Successfully fixed the issues in {file_path}")
    print("\nThe fix addresses two key issues:")
    print("1. Consistent column naming: 'total_active_presses' instead of 'active_lever_presses'")
    print("2. Added column existence checking in plot_time_segment_data")
    
    print("\nYou can now run 'python analysisRunner.py' again")
    return True

def main():
    """Main function"""
    print("MedPC Analysis Error Fix Script")
    print("==============================")
    
    # Check if we should run in a specific directory
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        if os.path.isdir(target_dir):
            os.chdir(target_dir)
            print(f"Changed to directory: {target_dir}")
        else:
            print(f"Error: Directory not found: {target_dir}")
            return 1
    
    print(f"Current directory: {os.getcwd()}")
    
    # Fix column names issue
    if fix_column_names_issue():
        print("\nSUCCESS: All fixes applied!")
        return 0
    else:
        print("\nERROR: Fix could not be applied.")
        return 1

if __name__ == "__main__":
    sys.exit(main())