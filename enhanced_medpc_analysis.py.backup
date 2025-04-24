#!/usr/bin/env python
"""
Enhanced MedPC Data Analysis

This script extends the existing MedPC analysis framework to provide:
1. Time-segmented analysis (analyze specific time periods of experimental sessions)
2. Longitudinal metric graphs (plot metrics across all experimental phases)

Built to work with the existing improved_medpc_parser.py and analyze_medpc.py files.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from pathlib import Path
import re
from datetime import datetime
from matplotlib.dates import DateFormatter, DayLocator

# Import the existing parser and analyzer
from improved_medpc_parser import MedPCDataParser
from analyze_medpc_with_date_selection import MedPCAnalyzer


class EnhancedMedPCAnalyzer(MedPCAnalyzer):
    """
    Enhanced analyzer that extends MedPCAnalyzer with additional analysis capabilities.
    Handles phase-specific differences in experimental design:
    - SelfAdmin: 180 min session, reinforcement delivered
    - EXT: 180 min session, no reinforcement
    - REI: 60 min session, reinforcement delivered
    """
    
    def __init__(self, data_dir="./data", output_dir="./enhanced_analysis_output"):
        """
        Initialize the enhanced MedPC data analyzer.
        
        Parameters:
        -----------
        data_dir : str or Path
            Directory containing MedPC data files
        output_dir : str or Path
            Directory to save analysis outputs
        """
        super().__init__(data_dir, output_dir)
        
        # Additional data containers
        self.longitudinal_df = None
        self.time_segment_df = {}
        
        # Phase-specific session lengths (in minutes)
        self.phase_session_lengths = {
            'SelfAdmin': 180,
            'EXT': 180,
            'REI': 60
        }
    
    def analyze_time_segment(self, start_min=0, end_min=60, bin_size=5):
        """
        Analyze a specific time segment of experimental sessions.
        Takes into account phase-specific session lengths.
        
        Parameters:
        -----------
        start_min : int
            Start time in minutes (inclusive)
        end_min : int
            End time in minutes (exclusive)
        bin_size : int
            Size of time bins in minutes for analysis
            
        Returns:
        --------
        Polars DataFrame with time-segmented data
        """
        if self.time_series_df is None:
            raise ValueError("No time series data available. Call load_and_process_data() first.")
        
        # Convert time boundaries to seconds
        start_sec = start_min * 60
        end_sec = end_min * 60
        
        # Filter time series data to the specified time segment
        segment_df = self.time_series_df.filter(
            (pl.col('time_seconds') >= start_sec) & 
            (pl.col('time_seconds') < end_sec)
        )
        
        # Check if the time segment exceeds phase-specific session length
        # and add a warning note if necessary
        phase_warnings = []
        for phase, length in self.phase_session_lengths.items():
            if end_min > length:
                phase_data = segment_df.filter(pl.col('phase') == phase)
                if len(phase_data) > 0:
                    phase_warnings.append(
                        f"Warning: Time segment {start_min}-{end_min}min exceeds {phase} "
                        f"session length ({length}min). Data beyond {length}min will be empty."
                    )
                    
        if phase_warnings:
            for warning in phase_warnings:
                print(warning)
        
        print(f"Analyzing time segment {start_min}-{end_min} minutes with {len(segment_df)} events")
        
        # Create a unique key for this time segment
        segment_key = f"{start_min}_{end_min}"
        
        # Store the segment DataFrame
        self.time_segment_df[segment_key] = segment_df
        
        # Calculate binned metrics for this segment
        binned_data = []
        
        # Group by subject, phase, and time bin
        bin_size_sec = bin_size * 60
        num_bins = int(np.ceil((end_sec - start_sec) / bin_size_sec))
        
        # Process each subject and phase separately
        subjects = segment_df.select('subject').unique().to_series().to_list()
        phases = segment_df.select('phase').unique().to_series().to_list()
        
        for subject in subjects:
            for phase in phases:
                if phase is None:
                    continue
                
                # Filter data for this subject and phase
                subject_phase_df = segment_df.filter(
                    (pl.col('subject') == subject) & 
                    (pl.col('phase') == phase)
                )
                
                if len(subject_phase_df) == 0:
                    continue
                
                # Get all sessions for this subject and phase
                sessions = subject_phase_df.select('filename').unique().to_series().to_list()
                
                for session in sessions:
                    # Filter data for this session
                    session_df = subject_phase_df.filter(pl.col('filename') == session)
                    
                    # Extract session date from filename (format: YYYY-MM-DD_HHhMMm_Subject XX.txt)
                    session_date = None
                    date_match = re.match(r'(\d{4}-\d{2}-\d{2})_', session)
                    if date_match:
                        session_date = date_match.group(1)
                    
                    # Calculate bin counts
                    active_bins = [0] * num_bins
                    inactive_bins = [0] * num_bins
                    reinforcer_bins = [0] * num_bins
                    
                    for row in session_df.iter_rows(named=True):
                        time_sec = row['time_seconds']
                        response_type = row['response_type']
                        
                        # Calculate bin index relative to start_sec
                        bin_idx = int((time_sec - start_sec) // bin_size_sec)
                        
                        if 0 <= bin_idx < num_bins:
                            if response_type == 'active_lever':
                                active_bins[bin_idx] += 1
                            elif response_type == 'inactive_lever':
                                inactive_bins[bin_idx] += 1
                            elif response_type == 'reinforced':
                                reinforcer_bins[bin_idx] += 1
                    
                    # Calculate totals for the entire segment
                    total_active = sum(active_bins)
                    total_inactive = sum(inactive_bins)
                    total_reinforcers = sum(reinforcer_bins)
                    
                    # Add a summary row for this session
                    binned_data.append({
                        'subject': subject,
                        'phase': phase,
                        'session': session,
                        'session_date': session_date,
                        'time_segment': f"{start_min}-{end_min}min",
                        'total_active_presses': total_active,
                        'total_inactive_presses': total_inactive,
                        'total_reinforcers': total_reinforcers,
                        'active_bins': active_bins,
                        'inactive_bins': inactive_bins,
                        'reinforcer_bins': reinforcer_bins
                    })
        
        # Create DataFrame from binned data
        segment_summary_df = pl.DataFrame(binned_data)
        
        # Save the segment summary to a CSV file
        output_path = self.output_dir / f"time_segment_{start_min}_{end_min}min_summary.csv"
        
        # Filter out list columns for CSV output
        list_cols = ['active_bins', 'inactive_bins', 'reinforcer_bins']
        csv_df = segment_summary_df.select([col for col in segment_summary_df.columns if col not in list_cols])
        csv_df.write_csv(output_path)
        
        print(f"Saved time segment summary to {output_path}")
        
        return segment_summary_df
    
    def create_longitudinal_dataframe(self):
        """
        Create a longitudinal DataFrame with metrics across all experimental phases.
        Adds phase-specific information and normalizes metrics by session length.
        
        Returns:
        --------
        Polars DataFrame with longitudinal data
        """
        if self.summary_df is None:
            raise ValueError("No summary data available. Call load_and_process_data() first.")
        
        # Extract subject, phase, date, and metrics from summary_df
        longitudinal_data = []
        
        for row in self.summary_df.iter_rows(named=True):
            subject = row['subject']
            phase = row['phase']
            filename = row['filename']
            
            if phase is None:
                continue
            
            # Extract date from filename (format: YYYY-MM-DD_HHhMMm_Subject XX.txt)
            session_date = None
            date_match = re.match(r'(\d{4}-\d{2}-\d{2})_', filename)
            if date_match:
                session_date = date_match.group(1)
                # Convert to datetime for sorting
                session_datetime = datetime.strptime(session_date, '%Y-%m-%d')
            else:
                # Use a default date if not found
                session_date = "Unknown"
                session_datetime = datetime.min
            
            # Get phase-specific session length
            session_length = self.phase_session_lengths.get(phase, 180)  # Default to 180 min
            
            # Calculate rate metrics (per hour)
            active_presses = row.get('active_lever_presses', 0)
            inactive_presses = row.get('inactive_lever_presses', 0)
            reinforcers = row.get('reinforcers', 0)
            
            active_rate = (active_presses / session_length) * 60  # Per hour
            inactive_rate = (inactive_presses / session_length) * 60  # Per hour
            reinforcer_rate = (reinforcers / session_length) * 60  # Per hour
            
            # Add a row for this session
            longitudinal_data.append({
                'subject': subject,
                'phase': phase,
                'session': filename,
                'session_date': session_date,
                'session_datetime': session_datetime,
                'session_length_min': session_length,
                'active_lever_presses': active_presses,
                'inactive_lever_presses': inactive_presses,
                'reinforcers': reinforcers,
                'active_presses_per_hour': active_rate,
                'inactive_presses_per_hour': inactive_rate,
                'reinforcers_per_hour': reinforcer_rate,
                # Add ratio of active to inactive
                'active_inactive_ratio': active_presses / max(1, inactive_presses)
            })
        
        # Create DataFrame and sort by subject, date, and phase
        self.longitudinal_df = pl.DataFrame(longitudinal_data)
        
        # Sort the DataFrame
        self.longitudinal_df = self.longitudinal_df.sort([
            pl.col('subject'),
            pl.col('session_datetime'),
            pl.col('phase')
        ])
        
        # Save to CSV
        output_path = self.output_dir / "longitudinal_data.csv"
        
        # Remove datetime column for CSV (not easily serializable)
        csv_df = self.longitudinal_df.select([col for col in self.longitudinal_df.columns 
                                             if col != 'session_datetime'])
        csv_df.write_csv(output_path)
        
        print(f"Created longitudinal DataFrame with {len(self.longitudinal_df)} rows")
        print(f"Saved longitudinal data to {output_path}")
        
        return self.longitudinal_df
    
    def plot_longitudinal_metrics(self, subjects=None, metrics=None, group_by_phase=True):
        """
        Plot metrics across all experimental phases for each subject.
        
        Parameters:
        -----------
        subjects : list
            List of subjects to include. If None, all subjects are included.
        metrics : list
            List of metrics to plot. If None, all metrics are plotted.
        group_by_phase : bool
            If True, group sessions by phase for display
        """
        if self.longitudinal_df is None:
            self.create_longitudinal_dataframe()
        
        # Default metrics if not specified
        if metrics is None:
            metrics = ['active_lever_presses', 'inactive_lever_presses', 'reinforcers']
        
        # Get unique subjects
        if subjects is None:
            subjects = self.longitudinal_df.select('subject').unique().to_series().to_list()
        else:
            # Filter to ensure subjects exist in the data
            existing_subjects = self.longitudinal_df.select('subject').unique().to_series().to_list()
            subjects = [s for s in subjects if s in existing_subjects]
        
        # Create figures for each subject
        for subject in subjects:
            # Filter data for this subject
            subject_df = self.longitudinal_df.filter(pl.col('subject') == subject)
            
            if len(subject_df) == 0:
                print(f"No data for subject {subject}")
                continue
            
            # Get phases for this subject
            phases = subject_df.select('phase').unique().to_series().to_list()
            
            # Create a figure for each metric
            for metric in metrics:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                if group_by_phase:
                    # Group by phase and calculate mean and SEM
                    for phase_idx, phase in enumerate(phases):
                        phase_df = subject_df.filter(pl.col('phase') == phase)
                        
                        # Convert to lists for plotting
                        session_numbers = list(range(1, len(phase_df) + 1))
                        metric_values = phase_df.select(metric).to_series().to_list()
                        
                        # Plot individual sessions
                        ax.plot(session_numbers, metric_values, 'o-', label=f"{phase}")
                        
                        # Add phase label
                        mid_idx = len(session_numbers) // 2
                        if mid_idx < len(session_numbers):
                            ax.text(session_numbers[mid_idx], max(metric_values) * 1.1, 
                                   phase, ha='center', fontweight='bold')
                        
                        # Add vertical separators between phases (except after the last one)
                        if phase_idx < len(phases) - 1:
                            ax.axvline(x=session_numbers[-1] + 0.5, color='gray', linestyle='--', alpha=0.5)
                else:
                    # Plot all sessions in sequence
                    metric_values = subject_df.select(metric).to_series().to_list()
                    session_numbers = list(range(1, len(metric_values) + 1))
                    
                    # Get phase for each session for coloring
                    session_phases = subject_df.select('phase').to_series().to_list()
                    phase_colors = {phase: plt.cm.tab10(i) for i, phase in enumerate(phases)}
                    
                    # Plot with colors by phase
                    for i, (x, y, phase) in enumerate(zip(session_numbers, metric_values, session_phases)):
                        if i == 0:
                            ax.plot(x, y, 'o', color=phase_colors[phase], label=phase)
                        else:
                            # Only add to legend if phase changed
                            if session_phases[i-1] != phase:
                                ax.plot(x, y, 'o', color=phase_colors[phase], label=phase)
                            else:
                                ax.plot(x, y, 'o', color=phase_colors[phase])
                        
                        # Connect points with lines
                        if i > 0:
                            ax.plot([session_numbers[i-1], x], [metric_values[i-1], y], '-', 
                                   color=phase_colors[session_phases[i-1]], alpha=0.7)
                
                # Customize plot
                metric_label = metric.replace('_', ' ').title()
                ax.set_title(f"Subject {subject}: {metric_label} Over Time")
                ax.set_xlabel("Session Number")
                ax.set_ylabel(metric_label)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                if not group_by_phase:
                    ax.legend(title="Phase")
                
                # Save the figure
                output_path = self.output_dir / f"longitudinal_{metric}_{subject}.png"
                plt.tight_layout()
                plt.savefig(output_path, dpi=300)
                plt.close()
                
                print(f"Saved longitudinal plot for subject {subject}, metric {metric}")
    
    # Replace the plot_time_segment_data method in enhanced_medpc_analysis.py with this fixed version:

    def plot_time_segment_data(self, segment_key, metrics=None, group_by_phase=True):
        """
        Plot metrics from a specific time segment analysis.
        
        Parameters:
        -----------
        segment_key : str
            Key identifying the time segment (format: "start_end")
        metrics : list
            List of metrics to plot. If None, default metrics are used.
        group_by_phase : bool
            If True, group by phase for plotting
        """
        # Check if segment data exists
        if segment_key not in self.time_segment_df:
            start_min, end_min = map(int, segment_key.split('_'))
            segment_df = self.analyze_time_segment(start_min, end_min)
        else:
            segment_df = self.time_segment_df[segment_key]
        
        # Default metrics
        if metrics is None:
            metrics = [
                'total_active_presses',
                'total_inactive_presses',
                'total_reinforcers'
            ]
        
        # Parse segment key to get time range
        start_min, end_min = map(int, segment_key.split('_'))
        time_range = f"{start_min}-{end_min}min"
        
        # Get unique subjects and phases
        subjects = segment_df.select('subject').unique().to_series().to_list()
        phases = segment_df.select('phase').unique().to_series().to_list()
        
        # Plot metrics by phase for each subject
        for subject in subjects:
            subject_df = segment_df.filter(pl.col('subject') == subject)
            
            if len(subject_df) == 0:
                continue
            
            # Create a figure for each metric
            for metric in metrics:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Prepare for phase grouping or sequential plotting
                if group_by_phase:
                    # Plot by phase
                    for phase in phases:
                        phase_df = subject_df.filter(pl.col('phase') == phase)
                        
                        if len(phase_df) == 0:
                            continue
                        
                        # Sort by session (use filename to ensure ordering if session_date isn't available)
                        # First check if session_date column exists
                        if 'session_date' in phase_df.columns:
                            try:
                                phase_df = phase_df.sort('session_date')
                            except:
                                # If sorting fails, try to sort by session/filename
                                if 'session' in phase_df.columns:
                                    phase_df = phase_df.sort('session')
                        elif 'session' in phase_df.columns:
                            phase_df = phase_df.sort('session')
                        
                        # Get values
                        session_nums = list(range(1, len(phase_df) + 1))
                        values = phase_df.select(metric).to_series().to_list()
                        
                        # Plot
                        ax.plot(session_nums, values, 'o-', label=phase)
                        
                        # Add phase label
                        if len(values) > 0:
                            mid_idx = len(values) // 2
                            if mid_idx < len(values):
                                y_pos = max(values) * 1.1 if max(values) > 0 else 1
                                ax.text(session_nums[mid_idx], y_pos, 
                                    phase, ha='center', fontweight='bold')
                else:
                    # Sort all sessions (using session/filename if session_date isn't available)
                    if 'session_date' in subject_df.columns:
                        try:
                            subject_df = subject_df.sort('session_date')
                        except:
                            if 'session' in subject_df.columns:
                                subject_df = subject_df.sort('session')
                    elif 'session' in subject_df.columns:
                        subject_df = subject_df.sort('session')
                    
                    # Get values
                    values = subject_df.select(metric).to_series().to_list()
                    session_phases = subject_df.select('phase').to_series().to_list()
                    session_nums = list(range(1, len(values) + 1))
                    
                    # Create a color map for phases
                    phase_colors = {phase: plt.cm.tab10(i) for i, phase in enumerate(phases)}
                    
                    # Plot with colors by phase
                    for i, (x, y, phase) in enumerate(zip(session_nums, values, session_phases)):
                        if i == 0 or (i > 0 and session_phases[i-1] != phase):
                            ax.plot(x, y, 'o', color=phase_colors[phase], label=phase)
                        else:
                            ax.plot(x, y, 'o', color=phase_colors[phase])
                        
                        # Connect points with lines
                        if i > 0:
                            ax.plot([session_nums[i-1], x], [values[i-1], y], '-', 
                                color=phase_colors[session_phases[i-1]], alpha=0.7)
                
                # Customize plot
                metric_label = metric.replace('total_', '').replace('_', ' ').title()
                ax.set_title(f"Subject {subject}: {metric_label} ({time_range})")
                ax.set_xlabel("Session Number")
                ax.set_ylabel(metric_label)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                if not group_by_phase:
                    ax.legend(title="Phase")
                
                # Save figure
                metric_name = metric.replace('total_', '')
                output_path = self.output_dir / f"time_segment_{segment_key}_{metric_name}_{subject}.png"
                plt.tight_layout()
                plt.savefig(output_path, dpi=300)
                plt.close()
                
                print(f"Saved time segment plot for subject {subject}, metric {metric_name}")
    
    def plot_time_segment_comparison(self, segment_keys, metric='total_active_presses'):
        """
        Compare a metric across different time segments.
        
        Parameters:
        -----------
        segment_keys : list
            List of time segment keys (format: "start_end")
        metric : str
            Metric to compare across segments
        """
        # Ensure all segment data is available
        for key in segment_keys:
            if key not in self.time_segment_df:
                start_min, end_min = map(int, key.split('_'))
                self.analyze_time_segment(start_min, end_min)
        
        # Get unique subjects and phases
        all_subjects = set()
        all_phases = set()
        
        for key in segment_keys:
            segment_df = self.time_segment_df[key]
            all_subjects.update(segment_df.select('subject').unique().to_series().to_list())
            all_phases.update(segment_df.select('phase').unique().to_series().to_list())
        
        subjects = sorted(list(all_subjects))
        phases = sorted(list(all_phases))
        
        # Create comparison plots for each subject and phase
        for subject in subjects:
            for phase in phases:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Process each time segment
                for key in segment_keys:
                    segment_df = self.time_segment_df[key]
                    
                    # Filter for this subject and phase
                    filtered_df = segment_df.filter(
                        (pl.col('subject') == subject) & 
                        (pl.col('phase') == phase)
                    )
                    
                    if len(filtered_df) == 0:
                        continue
                    
                    # Sort by date
                    filtered_df = filtered_df.sort('session_date')
                    
                    # Get values
                    session_nums = list(range(1, len(filtered_df) + 1))
                    values = filtered_df.select(metric).to_series().to_list()
                    
                    # Format time range for label
                    start_min, end_min = map(int, key.split('_'))
                    time_range = f"{start_min}-{end_min}min"
                    
                    # Plot
                    ax.plot(session_nums, values, 'o-', label=time_range)
                
                # Customize plot
                metric_label = metric.replace('total_', '').replace('_', ' ').title()
                ax.set_title(f"Subject {subject}, Phase {phase}: {metric_label} by Time Segment")
                ax.set_xlabel("Session Number")
                ax.set_ylabel(metric_label)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend(title="Time Segment")
                
                # Save figure
                metric_name = metric.replace('total_', '')
                segment_str = '_'.join(segment_keys)
                output_path = self.output_dir / f"segment_comparison_{metric_name}_subject{subject}_{phase}.png"
                plt.tight_layout()
                plt.savefig(output_path, dpi=300)
                plt.close()
                
                print(f"Saved time segment comparison for subject {subject}, phase {phase}")
    
    def plot_phase_comparison_for_segment(self, segment_key, metrics=None):
        """
        Create a direct comparison of metrics across phases for a specific time segment.
        This is useful for comparing the first 60 minutes of each phase.
        
        Parameters:
        -----------
        segment_key : str
            Key identifying the time segment (format: "start_end")
        metrics : list
            List of metrics to plot. If None, default metrics are used.
        """
        # Check if segment data exists
        if segment_key not in self.time_segment_df:
            start_min, end_min = map(int, segment_key.split('_'))
            segment_df = self.analyze_time_segment(start_min, end_min)
        else:
            segment_df = self.time_segment_df[segment_key]
        
        # Default metrics
        if metrics is None:
            metrics = [
                'total_active_presses',
                'total_inactive_presses',
                'total_reinforcers'
            ]
        
        # Parse segment key to get time range
        start_min, end_min = map(int, segment_key.split('_'))
        time_range = f"{start_min}-{end_min}min"
        
        # Get unique subjects and phases
        subjects = segment_df.select('subject').unique().to_series().to_list()
        phases = segment_df.select('phase').unique().to_series().to_list()
        
        # Create a figure for each metric
        for metric in metrics:
            # Create a figure for all subjects combined (group data)
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Process each phase
            x_positions = []
            labels = []
            all_phase_data = {}  # Store data by phase for statistical comparison
            
            for i, phase in enumerate(phases):
                phase_df = segment_df.filter(pl.col('phase') == phase)
                
                if len(phase_df) == 0:
                    continue
                
                # Get values for this phase
                phase_values = []
                
                # Get all sessions for this phase
                for subject in subjects:
                    subject_phase_df = phase_df.filter(pl.col('subject') == subject)
                    
                    if len(subject_phase_df) == 0:
                        continue
                    
                    # Get the mean for this subject's sessions in this phase
                    subject_values = subject_phase_df.select(metric).to_series().to_list()
                    if subject_values:
                        subject_mean = np.mean(subject_values)
                        phase_values.append(subject_mean)
                
                # Store for statistical tests
                all_phase_data[phase] = phase_values
                
                # Calculate mean and SEM
                if phase_values:
                    mean_val = np.mean(phase_values)
                    sem_val = np.std(phase_values) / np.sqrt(len(phase_values))
                    
                    x_positions.append(i)
                    labels.append(phase)
                    
                    # Plot bar with error bar
                    ax.bar(i, mean_val, yerr=sem_val, capsize=10, alpha=0.7, label=phase)
                    
                    # Add jittered data points
                    x_jitter = np.random.normal(i, 0.1, size=len(phase_values))
                    ax.scatter(x_jitter, phase_values, alpha=0.6, s=30, color='black')
            
            # Customize plot
            metric_label = metric.replace('total_', '').replace('_', ' ').title()
            ax.set_title(f"Phase Comparison: {metric_label} ({time_range})")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_ylabel(metric_label)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Add p-values if there are multiple phases (consider adding statistical tests here)
            
            # Save figure
            metric_name = metric.replace('total_', '')
            output_path = self.output_dir / f"phase_comparison_{segment_key}_{metric_name}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            print(f"Saved phase comparison plot for {metric_name} ({time_range})")
    
    def plot_group_means(self, time_segment=None, metrics=None):
        """
        Plot group means with SEM error bars for each phase.
        
        Parameters:
        -----------
        time_segment : str
            Time segment key (format: "start_end"). If None, use full session data.
        metrics : list
            List of metrics to plot. If None, default metrics are used.
        """
        # Default metrics
        if metrics is None:
            if time_segment:
                metrics = [
                    'total_active_presses',
                    'total_inactive_presses',
                    'total_reinforcers'
                ]
            else:
                metrics = [
                    'active_lever_presses',
                    'inactive_lever_presses',
                    'reinforcers'
                ]
        
        # Get the appropriate data source
        if time_segment:
            if time_segment not in self.time_segment_df:
                start_min, end_min = map(int, time_segment.split('_'))
                self.analyze_time_segment(start_min, end_min)
            
            source_df = self.time_segment_df[time_segment]
            title_suffix = f" ({start_min}-{end_min}min)"
        else:
            if self.longitudinal_df is None:
                self.create_longitudinal_dataframe()
            
            source_df = self.longitudinal_df
            title_suffix = " (Full Session)"
        
        # Get unique phases
        phases = source_df.select('phase').unique().to_series().to_list()
        
        # Create a figure for each metric
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Process each phase
            x_positions = []
            labels = []
            means = []
            sems = []
            
            for i, phase in enumerate(phases):
                phase_df = source_df.filter(pl.col('phase') == phase)
                
                if len(phase_df) == 0:
                    continue
                
                # Group by session to get per-session values
                grouped = phase_df.group_by('session_date')
                session_means = grouped.agg(pl.col(metric).mean())
                
                # Calculate overall mean and SEM
                values = session_means.select(metric).to_series().to_list()
                if values:
                    mean_val = np.mean(values)
                    sem_val = np.std(values) / np.sqrt(len(values))
                    
                    x_positions.append(i)
                    labels.append(phase)
                    means.append(mean_val)
                    sems.append(sem_val)
            
            # Plot bars with error bars
            ax.bar(x_positions, means, yerr=sems, capsize=10, alpha=0.7)
            
            # Add data points for individual sessions
            for i, phase in enumerate(labels):
                phase_df = source_df.filter(pl.col('phase') == phase)
                session_dates = phase_df.select('session_date').unique().to_series().to_list()
                
                for date in session_dates:
                    date_df = phase_df.filter(pl.col('session_date') == date)
                    values = date_df.select(metric).to_series().to_list()
                    
                    # Calculate mean for this session date
                    if values:
                        session_mean = np.mean(values)
                        # Add jitter
                        jitter = np.random.normal(0, 0.05)
                        ax.plot(i + jitter, session_mean, 'o', color='black', alpha=0.5)
            
            # Customize plot
            metric_label = metric.replace('total_', '').replace('_', ' ').title()
            ax.set_title(f"Group Mean {metric_label} by Phase{title_suffix}")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(labels)
            ax.set_ylabel(metric_label)
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Save figure
            metric_name = metric.replace('total_', '')
            segment_str = f"_{time_segment}" if time_segment else ""
            output_path = self.output_dir / f"group_mean_{metric_name}{segment_str}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            print(f"Saved group mean plot for {metric_name}{segment_str}")
    
    def run_enhanced_analysis(self, time_segments=None, selected_dates=None, selected_subjects=None):
        """
        Run enhanced analysis with time segments and longitudinal metrics.
        Automatically includes phase-appropriate time segments based on session lengths.
        
        Parameters:
        -----------
        time_segments : list
            List of time segments to analyze, each as (start_min, end_min)
        selected_dates : list
            List of dates to include in the analysis (format: "YYYY-MM-DD")
        selected_subjects : list
            List of subjects to analyze. If None, all subjects will be analyzed.
        """
        # Load and process data
        self.load_and_process_data(bin_sizes=[5, 10, 30], selected_dates=selected_dates)
        
        # Create longitudinal data
        self.create_longitudinal_dataframe()
        
        # Plot longitudinal metrics
        self.plot_longitudinal_metrics(subjects=selected_subjects, group_by_phase=True)
        self.plot_longitudinal_metrics(subjects=selected_subjects, group_by_phase=False)
        
        # Plot group means for full sessions
        self.plot_group_means()
        
        # Add phase-appropriate time segments if not provided
        if not time_segments:
            # Default time segments
            time_segments = [
                (0, 60),   # First hour (matches REI session length)
                (0, 30),   # First 30 minutes
                (30, 60),  # Second 30 minutes
            ]
            
            # Add segments for longer sessions
            for phase, length in self.phase_session_lengths.items():
                if length > 60:
                    # Add full session length segment
                    time_segments.append((0, length))
                    # Add third hour if applicable
                    if length >= 180:
                        time_segments.append((120, 180))
            
            # Remove duplicates
            time_segments = list(set(time_segments))
            print(f"Using default time segments: {time_segments}")
        
        # Process each time segment
        segment_keys = []
        
        for start_min, end_min in time_segments:
            segment_key = f"{start_min}_{end_min}"
            segment_keys.append(segment_key)
            
            # Analyze segment
            self.analyze_time_segment(start_min, end_min)
            
            # Plot segment data
            self.plot_time_segment_data(segment_key, group_by_phase=True)
            self.plot_time_segment_data(segment_key, group_by_phase=False)
            
            # Plot group means for this segment
            self.plot_group_means(time_segment=segment_key)
        
        # Compare first hour across all phases
        self.plot_phase_comparison_for_segment('0_60')
        
        # Compare across segments
        if len(segment_keys) > 1:
            for metric in ['total_active_presses', 'total_inactive_presses', 'total_reinforcers']:
                self.plot_time_segment_comparison(segment_keys, metric)
        
        print("Enhanced analysis complete!")


# Example usage as a standalone script
if __name__ == "__main__":
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run enhanced MedPC data analysis')
    parser.add_argument('--data_dir', default='./data', help='Directory containing MedPC data files')
    parser.add_argument('--output_dir', default='./enhanced_analysis_output', help='Directory to save analysis outputs')
    parser.add_argument('--time_segments', type=str, nargs='+', default=['0_60', '0_30', '30_60'], 
                        help='Time segments to analyze (format: start_end in minutes)')
    parser.add_argument('--dates', type=str, nargs='+', help='List of dates to analyze (format: YYYY-MM-DD)')
    parser.add_argument('--subjects', type=int, nargs='+', help='List of subjects to analyze')
    parser.add_argument('--list_dates', action='store_true', help='List available experiment dates and exit')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = EnhancedMedPCAnalyzer(args.data_dir, args.output_dir)
    
    # List available dates if requested
    if args.list_dates:
        dates = analyzer.list_available_dates()
        print("Available experiment dates:")
        for date in dates:
            print(f"  {date}")
        exit(0)
    
    # Parse time segments
    time_segments = []
    for segment_str in args.time_segments:
        start_min, end_min = map(int, segment_str.split('_'))
        time_segments.append((start_min, end_min))
    
    # Run analysis
    try:
        analyzer.run_enhanced_analysis(
            time_segments=time_segments,
            selected_dates=args.dates,
            selected_subjects=args.subjects
        )
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()