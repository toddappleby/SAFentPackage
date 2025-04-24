#!/usr/bin/env python
"""
MedPC Time Bin Analysis

This script analyzes lever press behavior in specific time bins across multiple sessions
for a subject or group of subjects in a particular experimental phase.
"""

import argparse
import polars as pl
from pathlib import Path
from medpc_analyzer import MedPCTimeAnalyzer

def main():
    """Run MedPC time bin analysis from command line."""
    
    # Create command-line parser
    parser = argparse.ArgumentParser(description='Analyze MedPC data in specific time bins across sessions')
    
    # Add the list_phases option BEFORE required arguments
    parser.add_argument('--list_phases', action='store_true', 
                      help='List available phases for each subject and exit')
    
    # Required arguments (only if not listing phases)
    parser.add_argument('--subjects', type=int, nargs='+', 
                       help='Subject ID(s) to analyze')
    parser.add_argument('--phase', type=str, choices=['SelfAdmin', 'EXT', 'REI'], 
                       help='Experimental phase to analyze (SelfAdmin, EXT, REI)')
    
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
                       help='Generate plots for multiple time bins (0-30, 30-60, 0-60 min)')
    parser.add_argument('--all_response_types', action='store_true',
                       help='Generate plots for all response types')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = MedPCTimeAnalyzer(args.data_dir, args.output_dir)
    
    # Load data
    analyzer.load_all_files()
    
    # List phases if requested
    if args.list_phases:
        # List available phases for each subject
        subjects = analyzer.get_available_subjects()
        print("\nAvailable phases by subject:")
        
        for subject in subjects:
            subject_data = analyzer.event_data.filter(pl.col('subject') == subject)
            phases = subject_data.select('phase').unique().to_series().to_list()
            phases = [p for p in phases if p]  # Remove None values
            
            # Count sessions per phase
            phase_counts = {}
            for phase in phases:
                phase_data = subject_data.filter(pl.col('phase') == phase)
                sessions = phase_data.select('filename').unique().to_series().to_list()
                phase_counts[phase] = len(sessions)
            
            print(f"Subject {subject}: {', '.join([f'{p} ({phase_counts[p]} sessions)' for p in phases])}")
        
        return
    
    # Require subjects and phase if not listing phases
    if args.subjects is None or args.phase is None:
        parser.error("--subjects and --phase are required unless --list_phases is specified")
    
    # Check if multiple subjects are specified
    if len(args.subjects) > 1:
        # Run group analysis
        if args.all_time_bins:
            # Generate plots for multiple time bins
            time_bins = [(0, 30), (30, 60), (0, 60)]
            
            for start, end in time_bins:
                if args.all_response_types:
                    # Generate plots for all response types
                    for response_type in ['active_lever', 'inactive_lever', 'head_entry']:
                        analyzer.plot_time_bin_across_sessions_group(
                            subjects=args.subjects,
                            phase=args.phase,
                            time_bin=(start, end),
                            response_type=response_type
                        )
                else:
                    # Generate plot for specified response type
                    analyzer.plot_time_bin_across_sessions_group(
                        subjects=args.subjects,
                        phase=args.phase,
                        time_bin=(start, end),
                        response_type=args.response_type
                    )
        else:
            # Generate plot for specific time bin
            if args.all_response_types:
                # Generate plots for all response types
                for response_type in ['active_lever', 'inactive_lever', 'head_entry']:
                    analyzer.plot_time_bin_across_sessions_group(
                        subjects=args.subjects,
                        phase=args.phase,
                        time_bin=(args.start_min, args.end_min),
                        response_type=response_type
                    )
            else:
                # Generate plot for specified response type and time bin
                analyzer.plot_time_bin_across_sessions_group(
                    subjects=args.subjects,
                    phase=args.phase,
                    time_bin=(args.start_min, args.end_min),
                    response_type=args.response_type
                )
    else:
        # Single subject analysis
        subject = args.subjects[0]  # Get the single subject from the list
        if args.all_time_bins:
            # Generate plots for multiple time bins
            time_bins = [(0, 30), (30, 60), (0, 60)]
            
            for start, end in time_bins:
                if args.all_response_types:
                    # Generate plots for all response types
                    for response_type in ['active_lever', 'inactive_lever', 'head_entry']:
                        analyzer.plot_time_bin_across_sessions(
                            subject=subject,
                            phase=args.phase,
                            time_bin=(start, end),
                            response_type=response_type
                        )
                else:
                    # Generate plot for specified response type
                    analyzer.plot_time_bin_across_sessions(
                        subject=subject,
                        phase=args.phase,
                        time_bin=(start, end),
                        response_type=args.response_type
                    )
        else:
            # Generate plot for specific time bin
            if args.all_response_types:
                # Generate plots for all response types
                for response_type in ['active_lever', 'inactive_lever', 'head_entry']:
                    analyzer.plot_time_bin_across_sessions(
                        subject=subject,
                        phase=args.phase,
                        time_bin=(args.start_min, args.end_min),
                        response_type=response_type
                    )
            else:
                # Generate plot for specified response type and time bin
                analyzer.plot_time_bin_across_sessions(
                    subject=subject,
                    phase=args.phase,
                    time_bin=(args.start_min, args.end_min),
                    response_type=args.response_type
                )
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()