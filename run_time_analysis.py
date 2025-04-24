#!/usr/bin/env python
"""
Run MedPC Time-Based Analysis

This script provides an easy-to-use command-line interface for the MedPC time-based analyzer.
It allows for analysis of MedPC data files with a focus on time segments and experimental phases.
"""

import os
import argparse
from pathlib import Path
from medpc_analyzer import MedPCTimeAnalyzer

def main():
    """Run the MedPC time-based analysis."""
    
    # Create command-line parser
    parser = argparse.ArgumentParser(description='Run MedPC Time-Based Analysis')
    
    # Data and output directories
    parser.add_argument('--data_dir', default='./data', help='Directory containing MedPC data files')
    parser.add_argument('--output_dir', default='./time_analysis', help='Directory to save analysis outputs')
    
    # Filter options
    parser.add_argument('--subjects', type=int, nargs='+', help='List of subjects to analyze')
    parser.add_argument('--phases', type=str, nargs='+', choices=['SelfAdmin', 'EXT', 'REI'], 
                        help='Phases to analyze (SelfAdmin, EXT, REI)')
    
    # Time segment options
    parser.add_argument('--segments', type=str, nargs='+', 
                        help='Time segments to analyze (format: start-end, e.g., 0-30)')
    parser.add_argument('--first_hour', action='store_true', 
                        help='Analyze first hour (0-60 minutes)')
    parser.add_argument('--first_30min', action='store_true', 
                        help='Analyze first 30 minutes (0-30 minutes)')
    parser.add_argument('--full_sessions', action='store_true', 
                        help='Analyze full session length for each phase')
    
    # Analysis type
    parser.add_argument('--comprehensive', action='store_true',
                        help='Run comprehensive analysis with all plots and summaries')
    parser.add_argument('--list_only', action='store_true',
                        help='Only list available subjects and phases')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Prepare paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    print(f"MedPC Time-Based Analysis")
    print(f"========================")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Initialize analyzer
    analyzer = MedPCTimeAnalyzer(data_dir, output_dir)
    
    # Load data
    analyzer.load_all_files()
    
    # If list_only, just show available subjects and phases
    if args.list_only:
        subjects = analyzer.get_available_subjects()
        phases = analyzer.get_available_phases()
        
        print("\nAvailable Subjects:")
        print(f"  {', '.join(map(str, subjects))}")
        
        print("\nAvailable Phases:")
        print(f"  {', '.join(phases)}")
        return
    
    # Determine time segments to analyze
    segments = []
    
    # Add segments from command-line arguments
    if args.segments:
        for segment in args.segments:
            start, end = map(int, segment.split('-'))
            segments.append((start, end))
    
    # Add preset segments
    if args.first_hour:
        segments.append((0, 60))
    
    if args.first_30min:
        segments.append((0, 30))
    
    # Add full session lengths if requested
    if args.full_sessions:
        for phase, length in analyzer.phase_session_lengths.items():
            segments.append((0, length))
    
    # If no segments specified, use a default set
    if not segments:
        segments = [(0, 30), (0, 60)]
    
    # Remove duplicates
    segments = list(set(segments))
    
    print("\nAnalyzing Time Segments:")
    for start, end in segments:
        print(f"  {start}-{end} minutes")
    
    # Run selected analysis
    if args.comprehensive:
        # Run comprehensive analysis
        analyzer.run_comprehensive_analysis(
            segments=segments,
            subjects=args.subjects,
            phases=args.phases
        )
    else:
        # Run standard analysis
        analyzer.analyze_multiple_segments(
            segments=segments,
            subjects=args.subjects,
            phases=args.phases
        )
        
        # Create basic plots for each segment
        for start, end in segments:
            segment_key = f"{start}_{end}"
            
            # Check if data exists for this segment
            if segment_key in analyzer.time_segment_data:
                print(f"\nCreating plots for segment {start}-{end} minutes...")
                
                # Create basic plots
                analyzer.plot_time_segment_comparison(segment_key, 'total_active_presses')
                analyzer.plot_time_course(segment_key)
                analyzer.summarize_segment_data(segment_key)
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()