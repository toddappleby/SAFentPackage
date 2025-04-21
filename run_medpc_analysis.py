#!/usr/bin/env python
"""
Run MedPC Data Analysis

This script processes MedPC data files and generates analysis outputs.
It combines the improved MedPC parser and analyzer to provide a complete workflow.

Usage:
    python run_medpc_analysis.py --data_dir=./my_data --output_dir=./my_results
    python run_medpc_analysis.py --list_dates
    python run_medpc_analysis.py --dates 2025-04-14
    python run_medpc_analysis.py --dates 2025-04-14 --subjects 83 84
"""

import os
import argparse
from pathlib import Path
from improved_medpc_parser import MedPCDataParser
from analyze_medpc_with_date_selection import MedPCAnalyzer

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process and analyze MedPC data files')
    parser.add_argument('--data_dir', default='./data', help='Directory containing MedPC data files')
    parser.add_argument('--output_dir', default='./analysis_output', help='Directory to save analysis outputs')
    parser.add_argument('--bin_sizes', type=int, nargs='+', default=[5, 10, 30], help='Time bin sizes in minutes')
    parser.add_argument('--metadata_file', default=None, help='Path to experimental metadata CSV file (optional)')
    parser.add_argument('--parse_only', action='store_true', help='Only parse files, skip analysis')
    parser.add_argument('--file_pattern', default='*.txt', help='File pattern to match (default: *.txt)')
    
    # Add new arguments for date and subject selection
    parser.add_argument('--dates', type=str, nargs='+', help='List of dates to analyze (format: YYYY-MM-DD)')
    parser.add_argument('--subjects', type=int, nargs='+', help='List of subjects to analyze')
    parser.add_argument('--list_dates', action='store_true', help='List available experiment dates and exit')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing MedPC data from: {data_dir}")
    print(f"Saving results to: {output_dir}")
    
    # Create an analyzer instance for basic functionality
    analyzer = MedPCAnalyzer(data_dir, output_dir)
    
    # List available dates if requested
    if args.list_dates:
        dates = analyzer.list_available_dates()
        print("Available experiment dates:")
        for date in dates:
            print(f"  {date}")
        return 0
    
    # Step 1: Parse MedPC files
    if args.parse_only:
        parser = MedPCDataParser(data_dir)
        
        try:
            # Find data files
            files = parser.find_files(args.file_pattern)
            print(f"Found {len(files)} MedPC data files")
            
            # Parse all files
            parser.parse_all_files()
            
            # Create dataframes
            df = parser.create_dataframe()
            print(f"Created DataFrame with {len(df)} rows")
            
            # Save processed data
            processed_data_dir = output_dir / 'processed_data'
            parser.save_data(processed_data_dir)
            print(f"Saved processed data to {processed_data_dir}")
            
            # If metadata file is provided, merge with it
            if args.metadata_file:
                metadata_path = Path(args.metadata_file)
                if metadata_path.exists():
                    print(f"Merging with metadata from: {metadata_path}")
                    merged_df = parser.merge_with_experiment_metadata(metadata_path)
                    merged_df.write_csv(processed_data_dir / 'merged_data.csv')
                    print(f"Saved merged data to {processed_data_dir / 'merged_data.csv'}")
                else:
                    print(f"Warning: Metadata file not found: {metadata_path}")
            
            return 0
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Step 2: Run analysis
    try:
        print("\nRunning analysis...")
        analyzer.run_analysis(
            bin_sizes=args.bin_sizes,
            selected_dates=args.dates,
            selected_subjects=args.subjects
        )
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\nAll done!")
    return 0

if __name__ == "__main__":
    exit(main())