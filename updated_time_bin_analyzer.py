#!/usr/bin/env python
"""
Enhanced MedPC Time Bin Analysis

This script analyzes lever press behavior in specific time bins across multiple sessions
for subjects with both numerical (83) and text-based (T1, T2, etc.) IDs.

Updated to work with the EnhancedMedPCDataParser.
"""

import argparse
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from enhanced_medpc_parser import EnhancedMedPCDataParser

class EnhancedMedPCTimeAnalyzer:
    """
    Enhanced analyzer that works with the new parser supporting text-based subject IDs.
    """
    
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize parser
        self.parser = EnhancedMedPCDataParser(data_dir)
        self.event_data = None
        
    def load_all_files(self):
        """Load and process all MedPC files."""
        # Find files
        files = self.parser.find_files('*.txt')
        print(f"Found {len(files)} MedPC data files")
        
        if len(files) > 0:
            # Parse all files
            self.parser.parse_all_files()
            
            # Create dataframes
            df = self.parser.create_dataframe()
            
            # Get time series data
            self.event_data = self.parser.create_time_series_dataframe()
            
            print(f"Processed {len(df)} sessions")
            print(f"Extracted {len(self.event_data)} events")
            
            return True
        else:
            print("No files found to process")
            return False
    
    def get_available_subjects(self):
        """Get list of available subjects (original IDs)."""
        if self.event_data is None:
            return []
        
        subjects = self.event_data.select('subject_original').unique().to_series().to_list()
        return [s for s in subjects if s is not None]
    
    def get_available_phases(self):
        """Get list of available experimental phases."""
        if self.event_data is None:
            return []
        
        phases = self.event_data.select('phase').unique().to_series().to_list()
        return [p for p in phases if p is not None]
    
    def _get_phase_sessions(self, subject_original, phase):
        """Get all sessions for a subject in a specific phase."""
        if self.event_data is None:
            return []
        
        # Filter data for this subject and phase
        subject_data = self.event_data.filter(
            (pl.col('subject_original') == subject_original) & 
            (pl.col('phase') == phase)
        )
        
        if len(subject_data) == 0:
            return []
        
        # Get unique sessions
        sessions = subject_data.select('filename').unique().to_series().to_list()
        
        # Extract dates and sort
        session_dates = []
        for session in sessions:
            date_match = re.match(r'(\d{4}-\d{2}-\d{2})_', session)
            if date_match:
                date = date_match.group(1)
                session_dates.append((session, date))
        
        # Sort by date
        session_dates.sort(key=lambda x: x[1])
        return session_dates
    
    def plot_time_bin_across_sessions(self, subject, phase, time_bin=(0, 30), response_type='active_lever'):
        """
        Plot response counts in a time bin across sessions for a single subject.
        """
        # Get sessions for this subject and phase
        sessions = self._get_phase_sessions(subject, phase)
        
        if not sessions:
            print(f"No {phase} sessions found for subject {subject}")
            return None
        
        print(f"Found {len(sessions)} {phase} sessions for subject {subject}")
        
        # Calculate time window in seconds
        start_sec = time_bin[0] * 60
        end_sec = time_bin[1] * 60
        
        # Count responses in each session
        session_counts = []
        
        for i, (session, date) in enumerate(sessions):
            # Get data for this session in the time window
            session_data = self.event_data.filter(
                (pl.col('subject_original') == subject) & 
                (pl.col('phase') == phase) & 
                (pl.col('filename') == session) &
                (pl.col('time_seconds') >= start_sec) & 
                (pl.col('time_seconds') < end_sec)
            )
            
            # Count responses of the specified type
            if response_type == 'head_entry':
                count = len(session_data.filter(pl.col('response_type') == 'head_entry'))
            else:
                count = len(session_data.filter(pl.col('response_type') == response_type))
            
            session_counts.append({
                'session': session,
                'date': date,
                'session_number': i + 1,
                'count': count
            })
        
        if not session_counts:
            print(f"No data found for the specified parameters")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        session_numbers = [s['session_number'] for s in session_counts]
        counts = [s['count'] for s in session_counts]
        dates = [s['date'] for s in session_counts]
        
        # Plot line
        ax.plot(session_numbers, counts, 'o-', linewidth=2, markersize=8)
        
        # Add value labels
        max_count = max(counts) if counts else 0
        for i, count in enumerate(counts):
            ax.text(session_numbers[i], count + max_count * 0.05, str(count), ha='center')
        
        # Customize plot
        response_label = response_type.replace('_', ' ').title()
        ax.set_title(f"Subject {subject}: {response_label} in {time_bin[0]}-{time_bin[1]} min window\nPhase: {phase}")
        ax.set_xlabel("Session Number")
        ax.set_ylabel(f"Number of {response_label}s")
        ax.set_xticks(session_numbers)
        ax.set_xticklabels(dates, rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0)
        
        # Save figure
        output_name = f"time_bin_{time_bin[0]}_{time_bin[1]}_subject{subject}_{phase}_{response_type}"
        output_path = self.output_dir / f"{output_name}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved plot: {output_path}")
        return fig
    
    def plot_time_bin_across_sessions_group(self, subjects, phase, time_bin=(0, 30), response_type='active_lever'):
        """
        Plot mean response counts across sessions for a group of subjects.
        """
        # Calculate time window in seconds
        start_sec = time_bin[0] * 60
        end_sec = time_bin[1] * 60
        
        # Get all sessions for all subjects
        all_sessions = []
        for subject in subjects:
            sessions = self._get_phase_sessions(subject, phase)
            for session, date in sessions:
                all_sessions.append({
                    'subject': subject,
                    'session': session,
                    'date': date
                })
        
        if not all_sessions:
            print(f"No {phase} sessions found for any of the specified subjects")
            return None
        
        # Group by date
        sessions_by_date = {}
        for session_info in all_sessions:
            date = session_info['date']
            if date not in sessions_by_date:
                sessions_by_date[date] = []
            sessions_by_date[date].append(session_info)
        
        # Calculate means for each date
        date_data = []
        for i, date in enumerate(sorted(sessions_by_date.keys())):
            sessions = sessions_by_date[date]
            subject_counts = []
            
            for session_info in sessions:
                subject = session_info['subject']
                session = session_info['session']
                
                # Get data for this session in the time window
                session_data = self.event_data.filter(
                    (pl.col('subject_original') == subject) & 
                    (pl.col('phase') == phase) & 
                    (pl.col('filename') == session) &
                    (pl.col('time_seconds') >= start_sec) & 
                    (pl.col('time_seconds') < end_sec)
                )
                
                # Count responses
                if response_type == 'head_entry':
                    count = len(session_data.filter(pl.col('response_type') == 'head_entry'))
                else:
                    count = len(session_data.filter(pl.col('response_type') == response_type))
                
                subject_counts.append(count)
            
            # Calculate statistics
            mean_count = np.mean(subject_counts) if subject_counts else 0
            sem_count = np.std(subject_counts) / np.sqrt(len(subject_counts)) if len(subject_counts) > 1 else 0
            
            date_data.append({
                'date': date,
                'session_number': i + 1,
                'mean_count': mean_count,
                'sem_count': sem_count,
                'n_subjects': len(subject_counts)
            })
        
        if not date_data:
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        session_numbers = [d['session_number'] for d in date_data]
        means = [d['mean_count'] for d in date_data]
        sems = [d['sem_count'] for d in date_data]
        dates = [d['date'] for d in date_data]
        
        # Plot with error bars
        ax.errorbar(session_numbers, means, yerr=sems, fmt='o-', linewidth=2, markersize=8, capsize=5)
        
        # Add value labels
        for i, mean in enumerate(means):
            ax.text(session_numbers[i], mean + max(means) * 0.05, f"{mean:.1f}", ha='center')
        
        # Customize plot
        response_label = response_type.replace('_', ' ').title()
        subject_list = ", ".join(subjects)
        ax.set_title(f"Subjects {subject_list}: Mean {response_label} in {time_bin[0]}-{time_bin[1]} min window\nPhase: {phase} (n={len(subjects)})")
        ax.set_xlabel("Session Number")
        ax.set_ylabel(f"Mean {response_label}s (± SEM)")
        ax.set_xticks(session_numbers)
        ax.set_xticklabels(dates, rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0)
        
        # Save figure
        subject_str = "_".join(subjects)
        output_name = f"time_bin_{time_bin[0]}_{time_bin[1]}_group_{subject_str}_{phase}_{response_type}"
        output_path = self.output_dir / f"{output_name}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved group plot: {output_path}")
        return fig
    
    def plot_discrimination_index_across_sessions(self, subject, phase, time_bin=(0, 30)):
        """
        Plot discrimination index across sessions for a subject.
        """
        sessions = self._get_phase_sessions(subject, phase)
        
        if not sessions:
            print(f"No {phase} sessions found for subject {subject}")
            return None
        
        # Calculate time window in seconds
        start_sec = time_bin[0] * 60
        end_sec = time_bin[1] * 60
        
        # Calculate discrimination index for each session
        session_data = []
        
        for i, (session, date) in enumerate(sessions):
            # Get data for this session in the time window
            window_data = self.event_data.filter(
                (pl.col('subject_original') == subject) & 
                (pl.col('phase') == phase) & 
                (pl.col('filename') == session) &
                (pl.col('time_seconds') >= start_sec) & 
                (pl.col('time_seconds') < end_sec)
            )
            
            # Count lever presses
            active_count = len(window_data.filter(pl.col('response_type') == 'active_lever'))
            inactive_count = len(window_data.filter(pl.col('response_type') == 'inactive_lever'))
            
            # Calculate discrimination index
            total_presses = active_count + inactive_count
            if total_presses > 0:
                di = (active_count - inactive_count) / total_presses
            else:
                di = 0.0
            
            session_data.append({
                'session': session,
                'date': date,
                'session_number': i + 1,
                'active_count': active_count,
                'inactive_count': inactive_count,
                'discrimination_index': di
            })
        
        if not session_data:
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        session_numbers = [s['session_number'] for s in session_data]
        dis = [s['discrimination_index'] for s in session_data]
        dates = [s['date'] for s in session_data]
        
        # Plot line
        ax.plot(session_numbers, dis, 'o-', linewidth=2, markersize=8, color='purple')
        
        # Add value labels
        for i, di in enumerate(dis):
            ax.text(session_numbers[i], di + 0.05, f"{di:.2f}", ha='center')
        
        # Customize plot
        ax.set_title(f"Subject {subject}: Discrimination Index in {time_bin[0]}-{time_bin[1]} min window\nPhase: {phase}")
        ax.set_xlabel("Session Number")
        ax.set_ylabel("Discrimination Index\n(active-inactive)/(active+inactive)")
        ax.set_xticks(session_numbers)
        ax.set_xticklabels(dates, rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_ylim(-1.1, 1.1)
        
        # Save figure
        output_name = f"discrimination_index_{time_bin[0]}_{time_bin[1]}_subject{subject}_{phase}"
        output_path = self.output_dir / f"{output_name}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved discrimination plot: {output_path}")
        return fig
    
    def plot_discrimination_index_across_sessions_group(self, subjects, phase, time_bin=(0, 30)):
        """
        Plot mean discrimination index across sessions for a group of subjects.
        """
        # Calculate time window in seconds
        start_sec = time_bin[0] * 60
        end_sec = time_bin[1] * 60
        
        # Get all sessions for all subjects
        all_sessions = []
        for subject in subjects:
            sessions = self._get_phase_sessions(subject, phase)
            for session, date in sessions:
                all_sessions.append({
                    'subject': subject,
                    'session': session,
                    'date': date
                })
        
        if not all_sessions:
            print(f"No {phase} sessions found for any of the specified subjects")
            return None
        
        # Group by date
        sessions_by_date = {}
        for session_info in all_sessions:
            date = session_info['date']
            if date not in sessions_by_date:
                sessions_by_date[date] = []
            sessions_by_date[date].append(session_info)
        
        # Calculate discrimination indices for each date
        date_data = []
        for i, date in enumerate(sorted(sessions_by_date.keys())):
            sessions = sessions_by_date[date]
            subject_dis = []
            
            for session_info in sessions:
                subject = session_info['subject']
                session = session_info['session']
                
                # Get data for this session in the time window
                window_data = self.event_data.filter(
                    (pl.col('subject_original') == subject) & 
                    (pl.col('phase') == phase) & 
                    (pl.col('filename') == session) &
                    (pl.col('time_seconds') >= start_sec) & 
                    (pl.col('time_seconds') < end_sec)
                )
                
                # Count lever presses
                active_count = len(window_data.filter(pl.col('response_type') == 'active_lever'))
                inactive_count = len(window_data.filter(pl.col('response_type') == 'inactive_lever'))
                
                # Calculate discrimination index
                total_presses = active_count + inactive_count
                if total_presses > 0:
                    di = (active_count - inactive_count) / total_presses
                else:
                    di = 0.0
                
                subject_dis.append(di)
            
            # Calculate statistics
            mean_di = np.mean(subject_dis) if subject_dis else 0
            sem_di = np.std(subject_dis) / np.sqrt(len(subject_dis)) if len(subject_dis) > 1 else 0
            
            date_data.append({
                'date': date,
                'session_number': i + 1,
                'mean_di': mean_di,
                'sem_di': sem_di,
                'n_subjects': len(subject_dis)
            })
        
        if not date_data:
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        session_numbers = [d['session_number'] for d in date_data]
        means = [d['mean_di'] for d in date_data]
        sems = [d['sem_di'] for d in date_data]
        dates = [d['date'] for d in date_data]
        
        # Plot with error bars
        ax.errorbar(session_numbers, means, yerr=sems, fmt='o-', linewidth=2, markersize=8, capsize=5, color='purple')
        
        # Add value labels
        for i, mean in enumerate(means):
            ax.text(session_numbers[i], mean + 0.05, f"{mean:.2f}", ha='center')
        
        # Customize plot
        subject_list = ", ".join(subjects)
        ax.set_title(f"Subjects {subject_list}: Mean Discrimination Index in {time_bin[0]}-{time_bin[1]} min window\nPhase: {phase} (n={len(subjects)})")
        ax.set_xlabel("Session Number")
        ax.set_ylabel("Discrimination Index ± SEM")
        ax.set_xticks(session_numbers)
        ax.set_xticklabels(dates, rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax.set_ylim(-1.1, 1.1)
        
        # Save figure
        subject_str = "_".join(subjects)
        output_name = f"discrimination_index_{time_bin[0]}_{time_bin[1]}_group_{subject_str}_{phase}"
        output_path = self.output_dir / f"{output_name}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Saved group discrimination plot: {output_path}")
        return fig
    
    def export_for_igor(self, subjects, phase, time_bins):
        """
        Export data in Igor Pro-friendly format.
        """
        for time_bin in time_bins:
            start_min, end_min = time_bin
            start_sec = start_min * 60
            end_sec = end_min * 60
            
            # Individual subject data
            for subject in subjects:
                sessions = self._get_phase_sessions(subject, phase)
                subject_data = []
                
                for i, (session, date) in enumerate(sessions):
                    # Get data for this session in the time window
                    window_data = self.event_data.filter(
                        (pl.col('subject_original') == subject) & 
                        (pl.col('phase') == phase) & 
                        (pl.col('filename') == session) &
                        (pl.col('time_seconds') >= start_sec) & 
                        (pl.col('time_seconds') < end_sec)
                    )
                    
                    # Count responses
                    active_count = len(window_data.filter(pl.col('response_type') == 'active_lever'))
                    inactive_count = len(window_data.filter(pl.col('response_type') == 'inactive_lever'))
                    head_entry_count = len(window_data.filter(pl.col('response_type') == 'head_entry'))
                    
                    # Calculate discrimination index
                    total_presses = active_count + inactive_count
                    di = (active_count - inactive_count) / total_presses if total_presses > 0 else 0.0
                    
                    subject_data.append({
                        'subject': subject,
                        'session_number': i + 1,
                        'date': date,
                        'time_window': f"{start_min}-{end_min}min",
                        'active_lever': active_count,
                        'inactive_lever': inactive_count,
                        'head_entries': head_entry_count,
                        'discrimination_index': di
                    })
                
                # Save individual subject data
                if subject_data:
                    df = pl.DataFrame(subject_data)
                    output_path = self.output_dir / f"igor_subject{subject}_{phase}_{start_min}-{end_min}min.csv"
                    df.write_csv(output_path)
                    print(f"Exported Igor data for subject {subject}: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze MedPC data in specific time bins across sessions')
    
    # List phases option first
    parser.add_argument('--list_phases', action='store_true', 
                      help='List available phases for each subject and exit')
    
    # Required arguments
    parser.add_argument('--subjects', type=str, nargs='+', 
                       help='Subject ID(s) to analyze (e.g., T1 T2 T3 or 83 84)')
    parser.add_argument('--phase', type=str, choices=['SelfAdmin', 'EXT', 'REI'], 
                       help='Experimental phase to analyze')
    
    # Optional arguments
    parser.add_argument('--data_dir', default='./data', 
                       help='Directory containing MedPC data files')
    parser.add_argument('--output_dir', default='./time_bin_analysis', 
                       help='Directory to save analysis outputs')
    parser.add_argument('--start_min', type=int, default=0, 
                       help='Start time in minutes (inclusive)')
    parser.add_argument('--end_min', type=int, default=30, 
                       help='End time in minutes (exclusive)')
    parser.add_argument('--response_type', type=str, default='active_lever', 
                       choices=['active_lever', 'inactive_lever', 'head_entry'],
                       help='Type of response to analyze')
    parser.add_argument('--all_time_bins', action='store_true', 
                       help='Generate plots for multiple time bins')
    parser.add_argument('--all_response_types', action='store_true',
                       help='Generate plots for all response types')
    parser.add_argument('--discrimination_index', action='store_true', 
                       help='Generate discrimination index plots')
    parser.add_argument('--export_for_igor', action='store_true',
                       help='Export data in Igor Pro format')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = EnhancedMedPCTimeAnalyzer(args.data_dir, args.output_dir)
    
    # Load data
    if not analyzer.load_all_files():
        print("Failed to load data files")
        return
    
    # List phases if requested
    if args.list_phases:
        subjects = analyzer.get_available_subjects()
        print("\nAvailable subjects and phases:")
        for subject in subjects:
            phases = analyzer.event_data.filter(pl.col('subject_original') == subject).select('phase').unique().to_series().to_list()
            phases = [p for p in phases if p is not None]
            print(f"Subject {subject}: {', '.join(phases)}")
        return
    
    # Validate required arguments
    if not args.subjects or not args.phase:
        parser.error("--subjects and --phase are required unless --list_phases is specified")
    
    # Determine time bins
    if args.all_time_bins:
        time_bins = [(0, 30), (30, 60), (0, 60)]
    else:
        time_bins = [(args.start_min, args.end_min)]
    
    # Determine response types
    if args.all_response_types:
        response_types = ['active_lever', 'inactive_lever', 'head_entry']
    else:
        response_types = [args.response_type]
    
    # Run analysis
    if len(args.subjects) == 1:
        # Single subject analysis
        subject = args.subjects[0]
        
        for time_bin in time_bins:
            for response_type in response_types:
                analyzer.plot_time_bin_across_sessions(
                    subject=subject,
                    phase=args.phase,
                    time_bin=time_bin,
                    response_type=response_type
                )
            
            if args.discrimination_index:
                analyzer.plot_discrimination_index_across_sessions(
                    subject=subject,
                    phase=args.phase,
                    time_bin=time_bin
                )
    else:
        # Group analysis
        for time_bin in time_bins:
            for response_type in response_types:
                analyzer.plot_time_bin_across_sessions_group(
                    subjects=args.subjects,
                    phase=args.phase,
                    time_bin=time_bin,
                    response_type=response_type
                )
            
            if args.discrimination_index:
                analyzer.plot_discrimination_index_across_sessions_group(
                    subjects=args.subjects,
                    phase=args.phase,
                    time_bin=time_bin
                )
    
    # Export for Igor if requested
    if args.export_for_igor:
        print("Exporting data for Igor Pro...")
        analyzer.export_for_igor(
            subjects=args.subjects,
            phase=args.phase,
            time_bins=time_bins
        )
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()