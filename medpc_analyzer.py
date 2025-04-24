#!/usr/bin/env python
"""
MedPC Time-Based Analyzer

This module provides specialized tools for analyzing MedPC data with a focus on:
1. Separating data by experimental phases (SelfAdmin, EXT, REI)
2. Filtering and analyzing data by time segments (e.g., first 30 minutes of a session)
3. Creating summary statistics and visualizations for time-segmented data

The module is built to work with the existing MedPC parser and analyzer code.
"""

import os
import re
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Import the base parser - adjust import path as needed
from improved_medpc_parser import MedPCDataParser

class MedPCTimeAnalyzer:
    """
    Specialized analyzer for time-segmented MedPC data across experimental phases.
    """
    
    def __init__(self, data_dir="./data", output_dir="./time_analysis"):
        """
        Initialize the time-based analyzer.
        
        Parameters:
        -----------
        data_dir : str or Path
            Directory containing MedPC data files
        output_dir : str or Path
            Directory to save analysis outputs
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase-specific session lengths (in minutes)
        self.phase_session_lengths = {
            'SelfAdmin': 180,
            'EXT': 180,
            'REI': 60
        }
        
        # Create parser
        self.parser = MedPCDataParser(self.data_dir)
        
        # Data containers
        self.event_data = None  # Processed event data
        self.time_segment_data = {}  # Data organized by time segments
        self.summary_data = None  # Summary statistics
        
        # List of processed files
        self.processed_files = []
    
    def load_all_files(self):
        """
        Load and process all MedPC files in the data directory.
        """
        # Find all .txt files
        files = self.parser.find_files("*.txt")
        print(f"Found {len(files)} MedPC data files")
        
        # Parse all files
        if len(files) > 0:
            self.parser.parse_all_files()
            self.parser.create_dataframe()
            
            # Get time series data
            self.event_data = self.parser.create_time_series_dataframe()
            
            # Store the list of processed files
            self.processed_files = [f.name for f in files]
            
            print(f"Processed {len(self.processed_files)} files")
            print(f"Extracted {len(self.event_data)} events")
            
            return True
        else:
            print("No files found to process")
            return False
    
    def get_available_subjects(self):
        """
        Get a list of all available subjects in the processed data.
        
        Returns:
        --------
        list of subject IDs
        """
        if self.event_data is None:
            if not self.load_all_files():
                return []
        
        subjects = self.event_data.select(pl.col('subject')).unique().to_series().to_list()
        return subjects
    
    def get_available_phases(self):
        """
        Get a list of all available experimental phases in the processed data.
        
        Returns:
        --------
        list of phase names
        """
        if self.event_data is None:
            if not self.load_all_files():
                return []
        
        phases = self.event_data.select(pl.col('phase')).unique().to_series().to_list()
        return [p for p in phases if p is not None]
    
    def analyze_time_segment(self, start_min=0, end_min=30, bin_size=5, subjects=None, phases=None):
        """
        Analyze a specific time segment of experimental sessions.
        
        Parameters:
        -----------
        start_min : int
            Start time in minutes (inclusive)
        end_min : int
            End time in minutes (exclusive)
        bin_size : int
            Size of time bins in minutes for analysis
        subjects : list
            List of subjects to include (if None, all subjects are included)
        phases : list
            List of phases to include (if None, all phases are included)
            
        Returns:
        --------
        Polars DataFrame with time-segmented data
        """
        if self.event_data is None:
            if not self.load_all_files():
                return None
        
        # Convert time boundaries to seconds
        start_sec = start_min * 60
        end_sec = end_min * 60
        
        # Filter to the specified time segment
        segment_df = self.event_data.filter(
            (pl.col('time_seconds') >= start_sec) & 
            (pl.col('time_seconds') < end_sec)
        )
        
        # Apply subject filter if provided
        if subjects is not None:
            segment_df = segment_df.filter(pl.col('subject').is_in(subjects))
        
        # Apply phase filter if provided
        if phases is not None:
            segment_df = segment_df.filter(pl.col('phase').is_in(phases))
        
        # Check if the time segment exceeds phase-specific session length
        # and add a warning note if necessary
        phase_warnings = []
        for phase, length in self.phase_session_lengths.items():
            if end_min > length:
                if phases is None or phase in phases:
                    phase_data = segment_df.filter(pl.col('phase') == phase)
                    if len(phase_data) > 0:
                        phase_warnings.append(
                            f"Warning: Time segment {start_min}-{end_min}min exceeds {phase} "
                            f"session length ({length}min). Data beyond {length}min will be empty."
                        )
        
        if phase_warnings:
            for warning in phase_warnings:
                print(warning)
        
        # Create a unique key for this time segment
        segment_key = f"{start_min}_{end_min}"
        if subjects is not None:
            segment_key += f"_subj{'_'.join(map(str, subjects))}"
        if phases is not None:
            segment_key += f"_{'_'.join(phases)}"
        
        print(f"Analyzing time segment {start_min}-{end_min} minutes with {len(segment_df)} events")
        
        # Calculate binned metrics for this segment
        binned_data = []
        
        # Process each subject, phase and session
        filtered_subjects = segment_df.select('subject').unique().to_series().to_list()
        filtered_phases = segment_df.select('phase').unique().to_series().to_list()
        
        bin_size_sec = bin_size * 60
        num_bins = int(np.ceil((end_sec - start_sec) / bin_size_sec))
        
        for subject in filtered_subjects:
            for phase in filtered_phases:
                if phase is None:
                    continue
                
                # Filter data for this subject and phase
                subject_phase_df = segment_df.filter(
                    (pl.col('subject') == subject) & 
                    (pl.col('phase') == phase)
                )
                
                if len(subject_phase_df) == 0:
                    continue
                
                # Get sessions for this subject and phase
                sessions = subject_phase_df.select('filename').unique().to_series().to_list()
                
                for session in sessions:
                    # Filter data for this session
                    session_df = subject_phase_df.filter(pl.col('filename') == session)
                    
                    # Extract session date from filename
                    session_date = None
                    date_match = re.match(r'(\d{4}-\d{2}-\d{2})_', session)
                    if date_match:
                        session_date = date_match.group(1)
                    
                    # Calculate metrics for each time bin
                    bins = []
                    for bin_idx in range(num_bins):
                        bin_start_sec = start_sec + bin_idx * bin_size_sec
                        bin_end_sec = bin_start_sec + bin_size_sec
                        
                        # Filter events for this bin
                        bin_df = session_df.filter(
                            (pl.col('time_seconds') >= bin_start_sec) & 
                            (pl.col('time_seconds') < bin_end_sec)
                        )
                        
                        # Count events by type
                        active_count = len(bin_df.filter(pl.col('response_type') == 'active_lever'))
                        inactive_count = len(bin_df.filter(pl.col('response_type') == 'inactive_lever'))
                        reinforcer_count = len(bin_df.filter(pl.col('response_type') == 'reinforced'))
                        
                        # Add bin data
                        bins.append({
                            'bin_start_min': bin_start_sec / 60,
                            'bin_end_min': bin_end_sec / 60,
                            'active_lever_presses': active_count,
                            'inactive_lever_presses': inactive_count,
                            'reinforcers': reinforcer_count
                        })
                    
                    # Calculate totals for the entire segment
                    total_active = sum(b['active_lever_presses'] for b in bins)
                    total_inactive = sum(b['inactive_lever_presses'] for b in bins)
                    total_reinforcers = sum(b['reinforcers'] for b in bins)
                    
                    # Add segment summary
                    binned_data.append({
                        'subject': subject,
                        'phase': phase,
                        'session': session,
                        'session_date': session_date,
                        'time_segment': f"{start_min}-{end_min}min",
                        'total_active_presses': total_active,
                        'total_inactive_presses': total_inactive,
                        'total_reinforcers': total_reinforcers,
                        'bins': bins
                    })
        
        # Create DataFrame from binned data
        segment_summary_df = pl.DataFrame(binned_data)
        
        # Store the segment data
        self.time_segment_data[segment_key] = segment_summary_df
        
        # Save to CSV (excluding bins column which contains lists)
        if len(segment_summary_df) > 0:
            # Create a copy without the bins column
            csv_df = segment_summary_df.select([
                col for col in segment_summary_df.columns if col != 'bins'
            ])
            
            output_path = self.output_dir / f"time_segment_{segment_key}.csv"
            csv_df.write_csv(output_path)
            print(f"Saved time segment summary to {output_path}")
        
        return segment_summary_df
    
    def analyze_multiple_segments(self, segments=None, subjects=None, phases=None):
        """
        Analyze multiple time segments.
        
        Parameters:
        -----------
        segments : list
            List of (start_min, end_min) tuples
        subjects : list
            List of subjects to include
        phases : list
            List of phases to include
        """
        if segments is None:
            # Default segments based on phase lengths
            segments = [
                (0, 30),    # First 30 minutes
                (30, 60),   # Second 30 minutes
                (0, 60),    # First hour
                (0, 180)    # Full SelfAdmin/EXT session
            ]
        
        # Get and print available subjects and phases
        all_subjects = self.get_available_subjects()
        all_phases = self.get_available_phases()
        
        print(f"Available subjects: {all_subjects}")
        print(f"Available phases: {all_phases}")
        
        # Filter subjects and phases if provided
        if subjects is None:
            subjects = all_subjects
        else:
            subjects = [s for s in subjects if s in all_subjects]
        
        if phases is None:
            phases = all_phases
        else:
            phases = [p for p in phases if p in all_phases]
        
        print(f"Analyzing data for:")
        print(f"  Subjects: {subjects}")
        print(f"  Phases: {phases}")
        print(f"  Time segments: {segments}")
        
        # Analyze each segment
        for start_min, end_min in segments:
            self.analyze_time_segment(
                start_min=start_min,
                end_min=end_min,
                subjects=subjects,
                phases=phases
            )
    
    def summarize_segment_data(self, segment_key, by_phase=True, by_subject=False):
        """
        Summarize data for a time segment.
        
        Parameters:
        -----------
        segment_key : str
            Key for the time segment to summarize
        by_phase : bool
            Whether to group by phase (default: True)
        by_subject : bool
            Whether to group by subject (default: False)
            
        Returns:
        --------
        Polars DataFrame with summary data
        """
        if segment_key not in self.time_segment_data:
            print(f"Error: No data found for segment {segment_key}")
            return None
        
        segment_df = self.time_segment_data[segment_key]
        
        # Define group by columns
        group_cols = []
        if by_phase:
            group_cols.append('phase')
        if by_subject:
            group_cols.append('subject')
        
        # If no grouping, summarize all data
        if not group_cols:
            summary = segment_df.select([
                pl.col('total_active_presses').mean().alias('mean_active_presses'),
                pl.col('total_active_presses').std().alias('std_active_presses'),
                pl.col('total_inactive_presses').mean().alias('mean_inactive_presses'),
                pl.col('total_inactive_presses').std().alias('std_inactive_presses'),
                pl.col('total_reinforcers').mean().alias('mean_reinforcers'),
                pl.col('total_reinforcers').std().alias('std_reinforcers'),
                pl.col('total_active_presses').count().alias('sample_size')
            ])
        else:
            # Group and summarize
            grouped = segment_df.group_by(group_cols)
            
            summary = grouped.agg([
                pl.col('total_active_presses').mean().alias('mean_active_presses'),
                pl.col('total_active_presses').std().alias('std_active_presses'),
                pl.col('total_inactive_presses').mean().alias('mean_inactive_presses'),
                pl.col('total_inactive_presses').std().alias('std_inactive_presses'),
                pl.col('total_reinforcers').mean().alias('mean_reinforcers'),
                pl.col('total_reinforcers').std().alias('std_reinforcers'),
                pl.col('total_active_presses').count().alias('sample_size')
            ])
        
        # Add SEM columns
        summary = summary.with_columns([
            (pl.col('std_active_presses') / pl.col('sample_size').sqrt()).alias('sem_active_presses'),
            (pl.col('std_inactive_presses') / pl.col('sample_size').sqrt()).alias('sem_inactive_presses'),
            (pl.col('std_reinforcers') / pl.col('sample_size').sqrt()).alias('sem_reinforcers')
        ])
        
        # Save to CSV
        output_path = self.output_dir / f"summary_{segment_key}.csv"
        summary.write_csv(output_path)
        print(f"Saved summary to {output_path}")
        
        return summary
    
    def plot_time_segment_comparison(self, segment_key, metric='total_active_presses'):
        """
        Plot a comparison of phases for a specific time segment and metric.
        
        Parameters:
        -----------
        segment_key : str
            Key for the time segment to plot
        metric : str
            Metric to plot (default: 'total_active_presses')
        """
        if segment_key not in self.time_segment_data:
            print(f"Error: No data found for segment {segment_key}")
            return None
        
        segment_df = self.time_segment_data[segment_key]
        
        # Create a summary grouped by phase
        summary = self.summarize_segment_data(segment_key, by_phase=True, by_subject=False)
        
        if summary is None or len(summary) == 0:
            return None
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get phases and their positions
        phases = summary.select('phase').to_series().to_list()
        x_pos = np.arange(len(phases))
        
        # Get metrics
        means = summary.select(f'mean_{metric.replace("total_", "")}').to_series().to_list()
        sems = summary.select(f'sem_{metric.replace("total_", "")}').to_series().to_list()
        
        # Create bar plot
        bars = ax.bar(x_pos, means, yerr=sems, capsize=10, alpha=0.7)
        
        # Add jittered data points
        for i, phase in enumerate(phases):
            phase_data = segment_df.filter(pl.col('phase') == phase)
            values = phase_data.select(metric).to_series().to_list()
            
            # Add jitter
            jitter = np.random.normal(i, 0.1, size=len(values))
            ax.scatter(jitter, values, alpha=0.6, color='black', s=30)
        
        # Customize plot
        title_metric = metric.replace('total_', '').replace('_', ' ').title()
        
        # Extract time range from segment key
        time_range = segment_key.split('_')[:2]
        title_time = f"{time_range[0]}-{time_range[1]} min"
        
        ax.set_title(f"{title_metric} by Phase ({title_time})")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(phases)
        ax.set_ylabel(title_metric)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Save figure
        output_path = self.output_dir / f"phase_comparison_{segment_key}_{metric}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved phase comparison plot to {output_path}")
        
        return fig
    
    def plot_time_course(self, segment_key, subject=None, phase=None, bin_size=5):
        """
        Plot the time course of events within a segment.
        
        Parameters:
        -----------
        segment_key : str
            Key for the time segment to plot
        subject : int
            Subject to plot (if None, plots average across subjects)
        phase : str
            Phase to plot (if None, plots all phases)
        bin_size : int
            Size of time bins in minutes
            
        Returns:
        --------
        matplotlib Figure object
        """
        if segment_key not in self.time_segment_data:
            print(f"Error: No data found for segment {segment_key}")
            return None
        
        segment_df = self.time_segment_data[segment_key]
        
        # Filter by subject and phase if provided
        if subject is not None:
            segment_df = segment_df.filter(pl.col('subject') == subject)
        
        if phase is not None:
            segment_df = segment_df.filter(pl.col('phase') == phase)
        
        if len(segment_df) == 0:
            print("No data matching the specified filters")
            return None
        
        # Extract time range from segment_key
        time_range = segment_key.split('_')[:2]
        start_min, end_min = map(int, time_range)
        
        # Calculate number of bins
        num_bins = int(np.ceil((end_min - start_min) / bin_size))
        bin_edges = [start_min + i * bin_size for i in range(num_bins + 1)]
        bin_centers = [start_min + (i + 0.5) * bin_size for i in range(num_bins)]
        
        # Get phases for plotting
        if phase is None:
            phases = segment_df.select('phase').unique().to_series().to_list()
        else:
            phases = [phase]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot data for each phase
        for phase_name in phases:
            phase_data = segment_df.filter(pl.col('phase') == phase_name)
            
            if len(phase_data) == 0:
                continue
            
            # Extract bin data
            all_bins = []
            for row in phase_data.iter_rows(named=True):
                # Check if 'bins' is in the row
                if 'bins' in row and row['bins'] is not None:
                    all_bins.extend(row['bins'])
            
            if not all_bins:
                print(f"Warning: No bin data found for phase {phase_name}")
                continue
            
            # Convert to DataFrame
            try:
                bins_df = pl.DataFrame(all_bins)
            except Exception as e:
                print(f"Error creating DataFrame from bins: {e}")
                print(f"Example bin data: {all_bins[0] if all_bins else 'None'}")
                continue
            
            # Print column names for debugging
            print(f"Bin data columns: {bins_df.columns}")
            
            # Group by bin start time
            try:
                grouped = bins_df.group_by('bin_start_min').agg([
                    pl.col('active_lever_presses').mean().alias('active_mean'),
                    pl.col('active_lever_presses').std().alias('active_std'),
                    pl.col('inactive_lever_presses').mean().alias('inactive_mean'),
                    pl.col('inactive_lever_presses').std().alias('inactive_std'),
                    pl.col('reinforcers').mean().alias('reinforcers_mean'),
                    pl.col('reinforcers').std().alias('reinforcers_std'),
                    pl.col('active_lever_presses').count().alias('count')
                ])
                
                # Calculate SEM
                grouped = grouped.with_columns([
                    (pl.col('active_std') / pl.col('count').sqrt()).alias('active_sem'),
                    (pl.col('inactive_std') / pl.col('count').sqrt()).alias('inactive_sem'),
                    (pl.col('reinforcers_std') / pl.col('count').sqrt()).alias('reinforcers_sem')
                ])
                
                # First sort the DataFrame directly 
                grouped = grouped.sort('bin_start_min')
                
                # Then convert to lists for plotting
                bin_starts = grouped.select('bin_start_min').to_series().to_list()
                active_means = grouped.select('active_mean').to_series().to_list()
                active_sems = grouped.select('active_sem').to_series().to_list()
                inactive_means = grouped.select('inactive_mean').to_series().to_list()
                inactive_sems = grouped.select('inactive_sem').to_series().to_list()
                
                # Plot
                ax.errorbar(
                    bin_starts,
                    active_means,
                    yerr=active_sems,
                    fmt='o-',
                    label=f"{phase_name} - Active",
                    capsize=5
                )
                
                ax.errorbar(
                    bin_starts,
                    inactive_means,
                    yerr=inactive_sems,
                    fmt='s--',
                    label=f"{phase_name} - Inactive",
                    capsize=5
                )
                
            except Exception as e:
                print(f"Error processing bin data for phase {phase_name}: {e}")
                continue
        
        # Customize plot
        title = f"Lever Presses Over Time ({start_min}-{end_min} min)"
        if subject is not None:
            title += f" - Subject {subject}"
        
        ax.set_title(title)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Lever Presses (mean ± SEM)")
        ax.set_xticks(bin_edges)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Save figure
        output_name = f"time_course_{segment_key}"
        if subject is not None:
            output_name += f"_subj{subject}"
        if phase is not None:
            output_name += f"_{phase}"
            
        output_path = self.output_dir / f"{output_name}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved time course plot to {output_path}")
        
        return fig
        
    def plot_individual_subjects(self, segment_key, metric='total_active_presses', phase=None):
        """
        Plot individual subject data for a specific time segment.
        
        Parameters:
        -----------
        segment_key : str
            Key for the time segment to plot
        metric : str
            Metric to plot (default: 'total_active_presses')
        phase : str
            Phase to include (if None, plots all phases)
            
        Returns:
        --------
        matplotlib Figure object
        """
        if segment_key not in self.time_segment_data:
            print(f"Error: No data found for segment {segment_key}")
            return None
        
        segment_df = self.time_segment_data[segment_key]
        
        # Filter by phase if provided
        if phase is not None:
            segment_df = segment_df.filter(pl.col('phase') == phase)
        
        if len(segment_df) == 0:
            print("No data matching the specified filters")
            return None
        
        # Get unique subjects and phases
        subjects = segment_df.select('subject').unique().to_series().to_list()
        
        if phase is None:
            phases = segment_df.select('phase').unique().to_series().to_list()
        else:
            phases = [phase]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up positions for subjects
        x_pos = np.arange(len(subjects))
        bar_width = 0.8 / len(phases)
        
        # Plot data for each phase
        for i, phase_name in enumerate(phases):
            phase_data = segment_df.filter(pl.col('phase') == phase_name)
            
            if len(phase_data) == 0:
                continue
            
            # Calculate mean value for each subject
            subject_means = []
            
            for subject in subjects:
                subject_data = phase_data.filter(pl.col('subject') == subject)
                
                if len(subject_data) > 0:
                    mean_value = subject_data.select(metric).mean().item()
                    subject_means.append(mean_value)
                else:
                    subject_means.append(0)  # No data for this subject
            
            # Plot bars
            bar_pos = x_pos - 0.4 + (i + 0.5) * bar_width
            ax.bar(
                bar_pos, 
                subject_means, 
                width=bar_width, 
                label=phase_name,
                alpha=0.7
            )
            
            # Add individual data points
            for j, subject in enumerate(subjects):
                subject_data = phase_data.filter(pl.col('subject') == subject)
                values = subject_data.select(metric).to_series().to_list()
                
                if values:
                    jitter = np.random.normal(0, 0.05, size=len(values))
                    ax.scatter(
                        bar_pos[j] + jitter, 
                        values, 
                        color='black', 
                        alpha=0.6,
                        s=30
                    )
        
        # Customize plot
        title_metric = metric.replace('total_', '').replace('_', ' ').title()
        
        # Extract time range from segment key
        time_range = segment_key.split('_')[:2]
        title_time = f"{time_range[0]}-{time_range[1]} min"
        
        ax.set_title(f"{title_metric} by Subject ({title_time})")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"Subject {s}" for s in subjects])
        ax.set_ylabel(title_metric)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend()
        
        # Save figure
        output_name = f"individual_subjects_{segment_key}_{metric}"
        if phase is not None:
            output_name += f"_{phase}"
            
        output_path = self.output_dir / f"{output_name}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved individual subjects plot to {output_path}")
        
        return fig
    
    def plot_response_distribution(self, segment_key, phase=None):
        """
        Plot the distribution of active vs inactive lever presses.
        
        Parameters:
        -----------
        segment_key : str
            Key for the time segment to plot
        phase : str
            Phase to include (if None, creates separate plots for each phase)
            
        Returns:
        --------
        list of matplotlib Figure objects
        """
        if segment_key not in self.time_segment_data:
            print(f"Error: No data found for segment {segment_key}")
            return None
        
        segment_df = self.time_segment_data[segment_key]
        
        # Define phases to plot
        if phase is None:
            phases = segment_df.select('phase').unique().to_series().to_list()
        else:
            phases = [phase]
        
        # Extract time range from segment key
        time_range = segment_key.split('_')[:2]
        title_time = f"{time_range[0]}-{time_range[1]} min"
        
        figures = []
        
        # Create a plot for each phase
        for phase_name in phases:
            phase_data = segment_df.filter(pl.col('phase') == phase_name)
            
            if len(phase_data) == 0:
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Add a scatter plot of active vs inactive
            active = phase_data.select('total_active_presses').to_series().to_list()
            inactive = phase_data.select('total_inactive_presses').to_series().to_list()
            
            # Get subjects for coloring
            subjects = phase_data.select('subject').to_series().to_list()
            unique_subjects = sorted(set(subjects))
            
            # Create a colormap
            cmap = plt.cm.get_cmap('tab10', len(unique_subjects))
            subject_colors = {s: cmap(i) for i, s in enumerate(unique_subjects)}
            
            # Plot points
            for i, (x, y, s) in enumerate(zip(inactive, active, subjects)):
                ax.scatter(
                    x, y, 
                    color=subject_colors[s], 
                    s=100, 
                    alpha=0.7,
                    label=f"Subject {s}" if s not in unique_subjects[:i] else ""
                )
            
            # Add 1:1 line
            max_val = max(max(active), max(inactive)) if active and inactive else 10
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
            
            # Customize plot
            ax.set_title(f"Active vs Inactive Lever Presses - {phase_name} ({title_time})")
            ax.set_xlabel("Inactive Lever Presses")
            ax.set_ylabel("Active Lever Presses")
            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Only add legend entries for each subject once
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())
            
            # Save figure
            output_name = f"response_distribution_{segment_key}_{phase_name}"
            output_path = self.output_dir / f"{output_name}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            
            figures.append(fig)
            print(f"Saved response distribution plot to {output_path}")
            
            plt.close()
        
        return figures
    
    def create_summary_report(self, segment_keys=None):
        """
        Create a comprehensive summary report of the analysis.
        
        Parameters:
        -----------
        segment_keys : list
            List of segment keys to include in the report (if None, uses all analyzed segments)
            
        Returns:
        --------
        str
            Path to the saved report
        """
        if segment_keys is None:
            segment_keys = list(self.time_segment_data.keys())
        
        if not segment_keys:
            print("No data available for report")
            return None
        
        # Create report file
        report_path = self.output_dir / "summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("MedPC Time-Based Analysis Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Files processed
            f.write(f"Number of files processed: {len(self.processed_files)}\n")
            f.write("-" * 50 + "\n\n")
            
            # Subjects and phases
            subjects = self.get_available_subjects()
            phases = self.get_available_phases()
            
            f.write(f"Subjects: {', '.join(map(str, subjects))}\n")
            f.write(f"Phases: {', '.join(phases)}\n")
            f.write("-" * 50 + "\n\n")
            
            # Summary for each time segment
            for segment_key in segment_keys:
                if segment_key not in self.time_segment_data:
                    continue
                
                segment_df = self.time_segment_data[segment_key]
                
                # Extract time range
                time_parts = segment_key.split('_')[:2]
                time_range = f"{time_parts[0]}-{time_parts[1]}min"
                
                f.write(f"Time Segment: {time_range}\n")
                f.write("-" * 30 + "\n")
                
                # Get phases in this segment
                seg_phases = segment_df.select('phase').unique().to_series().to_list()
                
                for phase_name in seg_phases:
                    phase_data = segment_df.filter(pl.col('phase') == phase_name)
                    
                    f.write(f"\nPhase: {phase_name}\n")
                    
                    # Get subjects in this phase
                    phase_subjects = phase_data.select('subject').unique().to_series().to_list()
                    f.write(f"  Subjects: {', '.join(map(str, phase_subjects))}\n")
                    
                    # Calculate mean metrics
                    active_mean = phase_data.select('total_active_presses').mean().item()
                    active_sem = phase_data.select('total_active_presses').std().item() / np.sqrt(len(phase_data))
                    
                    inactive_mean = phase_data.select('total_inactive_presses').mean().item()
                    inactive_sem = phase_data.select('total_inactive_presses').std().item() / np.sqrt(len(phase_data))
                    
                    reinforcers_mean = phase_data.select('total_reinforcers').mean().item()
                    reinforcers_sem = phase_data.select('total_reinforcers').std().item() / np.sqrt(len(phase_data))
                    
                    f.write(f"  Active Lever Presses: {active_mean:.2f} ± {active_sem:.2f} SEM\n")
                    f.write(f"  Inactive Lever Presses: {inactive_mean:.2f} ± {inactive_sem:.2f} SEM\n")
                    f.write(f"  Reinforcers: {reinforcers_mean:.2f} ± {reinforcers_sem:.2f} SEM\n")
                    
                    # Calculate active:inactive ratio
                    if inactive_mean > 0:
                        ratio = active_mean / inactive_mean
                        f.write(f"  Active:Inactive Ratio: {ratio:.2f}\n")
                    else:
                        f.write(f"  Active:Inactive Ratio: N/A (no inactive presses)\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
            
            # Add a section for list of plots created
            f.write("Plots Created:\n")
            f.write("-" * 30 + "\n")
            
            plot_files = list(self.output_dir.glob("*.png"))
            for plot_file in plot_files:
                f.write(f"- {plot_file.name}\n")
        
        print(f"Created summary report: {report_path}")
        return report_path

    def plot_time_bin_across_sessions(self, subject, phase, time_bin=(0, 30), response_type='active_lever'):
        """
        Plot the mean lever presses in a specific time bin across all sessions for a subject.
        
        Parameters:
        -----------
        subject : int
            Subject ID to analyze
        phase : str
            Experimental phase to analyze (SelfAdmin, EXT, REI)
        time_bin : tuple
            Time bin to analyze in minutes (start, end)
        response_type : str
            Type of response to analyze ('active_lever', 'inactive_lever', 'head_entry')
            
        Returns:
        --------
        matplotlib Figure object
        """
        # Load data if not already loaded
        if self.event_data is None:
            if not self.load_all_files():
                return None
                
        # Calculate time bin in seconds
        start_sec = time_bin[0] * 60
        end_sec = time_bin[1] * 60
        
        # Get all data for this subject
        subject_all_data = self.event_data.filter(pl.col('subject') == subject)
        
        if len(subject_all_data) == 0:
            print(f"No data found for subject {subject}")
            return None
        
        # Print available phases for debugging
        available_phases = subject_all_data.select('phase').unique().to_series().to_list()
        
        print(f"Available phases for subject {subject}: {available_phases}")

        
        # We'll manually check each file to identify which ones match our requested phase
        session_files = subject_all_data.select('filename').unique().to_series().to_list()
        matching_sessions = []
        
        for session in session_files:
            # Get the original file data from the parser
            file_data = self.parser.parsed_data.get(session, {})
            if not file_data or 'header' not in file_data:
                continue
                
            # Check the MSN field for phase markers
            msn = file_data['header'].get('msn', '')
            phase_found = False
            
            # Check different formats of phase markers
            if phase == 'SelfAdmin' and 'SelfAdmin' in msn and 'EXT' not in msn and 'REI' not in msn:
                phase_found = True
            elif phase == 'EXT' and 'EXT' in msn:
                phase_found = True
            elif phase == 'REI' and 'REI' in msn:
                phase_found = True
                
            if phase_found:
                # Extract the date
                date_match = re.match(r'(\d{4}-\d{2}-\d{2})_', session)
                if date_match:
                    date = date_match.group(1)
                    matching_sessions.append((session, date))
                    print(f"Found {phase} session: {session} (Date: {date})")
        
        if not matching_sessions:
            print(f"No {phase} sessions found for subject {subject}")
            return None
        
        # Sort sessions by date
        matching_sessions.sort(key=lambda x: x[1])
        
        # Count responses in the time bin for each session
        session_counts = []
        
        for i, (session, date) in enumerate(matching_sessions):
            # Get data for this session
            session_data = subject_all_data.filter(pl.col('filename') == session)
            
            # Filter to time bin
            bin_data = session_data.filter(
                (pl.col('time_seconds') >= start_sec) & 
                (pl.col('time_seconds') < end_sec)
            )
            
            # Count responses of the specified type
            if response_type == 'head_entry':
                # For head entries, use response_type 'reinforced'
                count = len(bin_data.filter(pl.col('response_type') == 'reinforced'))
            else:
                # For lever presses, use the specified response type
                count = len(bin_data.filter(pl.col('response_type') == response_type))
            
            session_counts.append({
                'session': session,
                'date': date,
                'session_number': i + 1,  # Sequential session number
                'count': count
            })
        
        # Create DataFrame for plotting
        if not session_counts:
            print(f"No data found for the specified time window and response type")
            return None
            
        counts_df = pl.DataFrame(session_counts)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot response counts
        ax.plot(
            counts_df['session_number'].to_list(), 
            counts_df['count'].to_list(),
            'o-', 
            linewidth=2, 
            markersize=8
        )
        
        # Add data labels
        max_count = max(counts_df['count'].to_list()) if counts_df['count'].to_list() else 0
        for i, row in enumerate(counts_df.iter_rows(named=True)):
            ax.text(
                row['session_number'], 
                row['count'] + max_count * 0.05,  # Slightly above the point
                str(row['count']),
                ha='center'
            )
        
        # Customize plot
        response_label = response_type.replace('_', ' ').title()
        ax.set_title(f"Subject {subject}: {response_label} in {time_bin[0]}-{time_bin[1]} min window\nPhase: {phase}")
        ax.set_xlabel("Session Number")
        ax.set_ylabel(f"Number of {response_label}s")
        ax.set_xticks(counts_df['session_number'].to_list())
        ax.set_xticklabels(counts_df['date'].to_list(), rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        # Save figure
        output_name = f"time_bin_{time_bin[0]}_{time_bin[1]}_subject{subject}_{phase}_{response_type}"
        output_path = self.output_dir / f"{output_name}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        
        print(f"Saved time bin analysis plot to {output_path}")
        
        return fig

    def plot_time_bin_across_sessions_group(self, subjects, phase, time_bin=(0, 30), response_type='active_lever'):
        """
        Plot the mean lever presses in a specific time bin across all sessions for a group of subjects.
        
        Parameters:
        -----------
        subjects : list
            List of subject IDs to analyze
        phase : str
            Experimental phase to analyze (SelfAdmin, EXT, REI)
        time_bin : tuple
            Time bin to analyze in minutes (start, end)
        response_type : str
            Type of response to analyze ('active_lever', 'inactive_lever', 'head_entry')
            
        Returns:
        --------
        matplotlib Figure object
        """
        # Load data if not already loaded
        if self.event_data is None:
            if not self.load_all_files():
                return None
                
        # Calculate time bin in seconds
        start_sec = time_bin[0] * 60
        end_sec = time_bin[1] * 60
        
        # Check if we have valid subjects
        all_subjects = self.get_available_subjects()
        valid_subjects = [s for s in subjects if s in all_subjects]
        
        if not valid_subjects:
            print(f"No valid subjects found in data. Available subjects: {all_subjects}")
            return None
        
        print(f"Analyzing subjects: {valid_subjects}")
        
        # Find all sessions for each subject in the specified phase
        all_sessions = []
        
        for subject in valid_subjects:
            # Get all data for this subject
            subject_data = self.event_data.filter(pl.col('subject') == subject)
            
            if len(subject_data) == 0:
                print(f"No data found for subject {subject}")
                continue
            
            # Get all sessions for this subject
            session_files = subject_data.select('filename').unique().to_series().to_list()
            
            for session in session_files:
                # Get the original file data
                file_data = self.parser.parsed_data.get(session, {})
                if not file_data or 'header' not in file_data:
                    continue
                    
                # Check the MSN field for phase markers
                msn = file_data['header'].get('msn', '')
                phase_found = False
                
                # Check different formats of phase markers
                if phase == 'SelfAdmin' and 'SelfAdmin' in msn and 'EXT' not in msn and 'REI' not in msn:
                    phase_found = True
                elif phase == 'EXT' and 'EXT' in msn:
                    phase_found = True
                elif phase == 'REI' and 'REI' in msn:
                    phase_found = True
                    
                if phase_found:
                    # Extract the date
                    date_match = re.match(r'(\d{4}-\d{2}-\d{2})_', session)
                    if date_match:
                        date = date_match.group(1)
                        all_sessions.append({
                            'subject': subject,
                            'session': session,
                            'date': date
                        })
        
        if not all_sessions:
            print(f"No {phase} sessions found for any of the specified subjects")
            return None
        
        # Group sessions by date
        sessions_by_date = {}
        for session_info in all_sessions:
            date = session_info['date']
            if date not in sessions_by_date:
                sessions_by_date[date] = []
            sessions_by_date[date].append(session_info)
        
        # Sort dates
        sorted_dates = sorted(sessions_by_date.keys())
        
        # Count responses in the time bin for each session
        date_counts = []
        
        for i, date in enumerate(sorted_dates):
            sessions = sessions_by_date[date]
            subject_counts = []
            
            for session_info in sessions:
                subject = session_info['subject']
                session = session_info['session']
                
                # Get data for this session
                subject_data = self.event_data.filter(pl.col('subject') == subject)
                session_data = subject_data.filter(pl.col('filename') == session)
                
                # Filter to time bin
                bin_data = session_data.filter(
                    (pl.col('time_seconds') >= start_sec) & 
                    (pl.col('time_seconds') < end_sec)
                )
                
                # Count responses of the specified type
                if response_type == 'head_entry':
                    # For head entries, use response_type 'reinforced'
                    count = len(bin_data.filter(pl.col('response_type') == 'reinforced'))
                else:
                    # For lever presses, use the specified response type
                    count = len(bin_data.filter(pl.col('response_type') == response_type))
                
                subject_counts.append({
                    'subject': subject,
                    'count': count
                })
            
            # Calculate mean and SEM for this date
            counts = [info['count'] for info in subject_counts]
            mean_count = np.mean(counts) if counts else 0
            sem_count = np.std(counts) / np.sqrt(len(counts)) if len(counts) > 1 else 0
            
            date_counts.append({
                'date': date,
                'session_number': i + 1,  # Sequential session number
                'mean_count': mean_count,
                'sem_count': sem_count,
                'n_subjects': len(subject_counts),
                'subject_counts': subject_counts
            })
        
        # Create DataFrame for plotting
        if not date_counts:
            print(f"No data found for the specified time window and response type")
            return None
                
        counts_df = pl.DataFrame([
            {
                'date': item['date'],
                'session_number': item['session_number'],
                'mean_count': item['mean_count'],
                'sem_count': item['sem_count'],
                'n_subjects': item['n_subjects']
            }
            for item in date_counts
        ])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot mean response counts with error bars
        ax.errorbar(
            counts_df['session_number'].to_list(), 
            counts_df['mean_count'].to_list(),
            yerr=counts_df['sem_count'].to_list(),
            fmt='o-', 
            linewidth=2, 
            markersize=8,
            capsize=5
        )
        
        # Add data labels
        for i, row in enumerate(counts_df.iter_rows(named=True)):
            ax.text(
                row['session_number'], 
                row['mean_count'] + max(counts_df['mean_count'].to_list()) * 0.05,  # Slightly above the point
                f"{row['mean_count']:.1f}",
                ha='center'
            )
        
        # Customize plot
        response_label = response_type.replace('_', ' ').title()
        subject_range = f"Subjects {min(valid_subjects)}-{max(valid_subjects)}"
        ax.set_title(f"{subject_range}: Mean {response_label} in {time_bin[0]}-{time_bin[1]} min window\nPhase: {phase} (n={len(valid_subjects)})")
        ax.set_xlabel("Session Number")
        ax.set_ylabel(f"Mean {response_label}s (± SEM)")
        ax.set_xticks(counts_df['session_number'].to_list())
        ax.set_xticklabels(counts_df['date'].to_list(), rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        # Save figure
        subject_str = f"subjects{min(valid_subjects)}-{max(valid_subjects)}"
        output_name = f"time_bin_{time_bin[0]}_{time_bin[1]}_{subject_str}_{phase}_{response_type}"
        output_path = self.output_dir / f"{output_name}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        
        print(f"Saved group analysis plot to {output_path}")
        
        return fig

    def run_comprehensive_analysis(self, segments=None, subjects=None, phases=None):
        """
        Run a comprehensive analysis with multiple time segments and plots.
        
        Parameters:
        -----------
        segments : list
            List of (start_min, end_min) tuples
        subjects : list
            List of subjects to include
        phases : list
            List of phases to include
            
        Returns:
        --------
        str
            Path to the summary report
        """
        # Define default time segments if not provided
        if segments is None:
            segments = [
                (0, 30),   # First 30 minutes
                (30, 60),  # Second 30 minutes
                (0, 60),   # First hour (common to all phases)
            ]
            
            # Add full session length segments
            for phase_name, length in self.phase_session_lengths.items():
                segments.append((0, length))
        
        # Analyze all segments
        self.analyze_multiple_segments(segments, subjects, phases)
        
        # Create plots for each segment
        for segment_key in self.time_segment_data.keys():
            # Phase comparison
            self.plot_time_segment_comparison(segment_key, 'total_active_presses')
            self.plot_time_segment_comparison(segment_key, 'total_inactive_presses')
            self.plot_time_segment_comparison(segment_key, 'total_reinforcers')
            
            # Individual subjects
            self.plot_individual_subjects(segment_key, 'total_active_presses')
            
            # Time course
            self.plot_time_course(segment_key)
            
            # Response distribution
            self.plot_response_distribution(segment_key)
            
            # Create segment summaries
            self.summarize_segment_data(segment_key, by_phase=True, by_subject=False)
            self.summarize_segment_data(segment_key, by_phase=True, by_subject=True)
        
        # Create summary report
        report_path = self.create_summary_report()
        
        print("Comprehensive analysis complete!")
        return report_path