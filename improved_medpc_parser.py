import os
import re
import polars as pl
import numpy as np
from pathlib import Path
import glob

class MedPCDataParser:
    def __init__(self, base_dir=None):
        """
        Initialize the MedPC data parser using Polars.
        
        Parameters:
        -----------
        base_dir : str or Path
            Base directory containing MedPC data files
        """
        self.base_dir = Path(base_dir) if base_dir else None
        self.data_files = []
        self.parsed_data = {}
        self.combined_df = None
    
    def find_files(self, pattern="*.txt"):
        """
        Find all MedPC data files matching the pattern in the base directory.
        
        Parameters:
        -----------
        pattern : str
            File pattern to match
            
        Returns:
        --------
        list of Path objects
        """
        if not self.base_dir:
            raise ValueError("Base directory not set")
        
        self.data_files = list(self.base_dir.glob(pattern))
        return self.data_files
    
    def parse_file(self, file_path):
        """
        Parse a single MedPC data file.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to the MedPC data file
            
        Returns:
        --------
        dict containing parsed data
        """
        file_path = Path(file_path)
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract header information
        header_data = {}
        
        # Extract file name
        header_data['filename'] = file_path.name
        
        # Extract date information
        start_date_match = re.search(r'Start Date: (\d{2}/\d{2}/\d{2})', content)
        if start_date_match:
            header_data['start_date'] = start_date_match.group(1)
        
        end_date_match = re.search(r'End Date: (\d{2}/\d{2}/\d{2})', content)
        if end_date_match:
            header_data['end_date'] = end_date_match.group(1)
        
        # Extract subject information
        subject_match = re.search(r'Subject: (\d+)', content)
        if subject_match:
            header_data['subject'] = int(subject_match.group(1))
        
        # Extract experiment type
        experiment_match = re.search(r'Experiment: (\w+)', content)
        if experiment_match:
            header_data['experiment'] = experiment_match.group(1)
        
        # Extract group information
        group_match = re.search(r'Group: (\w+)', content)
        if group_match:
            header_data['group'] = group_match.group(1)
        
        # Extract box number
        box_match = re.search(r'Box: (\d+)', content)
        if box_match:
            header_data['box'] = int(box_match.group(1))
        
        # Extract start and end times
        start_time_match = re.search(r'Start Time: (\d{2}:\d{2}:\d{2})', content)
        if start_time_match:
            header_data['start_time'] = start_time_match.group(1)
        
        end_time_match = re.search(r'End Time: (\d{2}:\d{2}:\d{2})', content)
        if end_time_match:
            header_data['end_time'] = end_time_match.group(1)
        
        # Extract MSN (program name)
        msn_match = re.search(r'MSN: (.+)', content)
        if msn_match:
            header_data['msn'] = msn_match.group(1)
            # Extract experimental phase from MSN
            if 'SelfAdmin' in header_data['msn']:
                header_data['phase'] = 'SelfAdmin'
            elif 'EXT' in header_data['msn']:
                header_data['phase'] = 'EXT'
            elif 'REI' in header_data['msn']:
                header_data['phase'] = 'REI'
        
        # Extract data arrays
        arrays = {}
        
        # Extract single-letter variables (F-Z)
        for letter in 'FGHIJKLMNOPQRSTUVWXYZ':
            pattern = rf'{letter}:\s+([\d.]+)'
            match = re.search(pattern, content)
            if match:
                arrays[letter] = float(match.group(1))
        
        # Extract multi-dimensional arrays (A-E and T)
        for letter in 'ABCDET':
            pattern = rf'{letter}:([\s\d.:]+?)(?=[A-Z]:|$)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                array_data = []
                array_text = match.group(1).strip()
                rows = array_text.split('\n')
                for row in rows:
                    if row.strip():
                        values = re.findall(r'(\d+):\s+([\d.\s]+)', row)
                        if values:
                            indices, data_points = values[0]
                            data_points = [float(dp) for dp in data_points.split() if dp.strip()]
                            array_data.extend(data_points)
                arrays[letter] = array_data
        
        # Combine all data
        data = {
            'header': header_data,
            'arrays': arrays
        }
        
        return data
    
    def parse_all_files(self):
        """
        Parse all found MedPC data files.
        
        Returns:
        --------
        dict containing parsed data for all files
        """
        if not self.data_files:
            raise ValueError("No data files found. Call find_files() first.")
        
        for file_path in self.data_files:
            try:
                parsed_data = self.parse_file(file_path)
                self.parsed_data[file_path.name] = parsed_data
                print(f"Successfully parsed {file_path.name}")
            except Exception as e:
                print(f"Error parsing {file_path.name}: {e}")
        
        return self.parsed_data
    
    def create_dataframe(self):
        """
        Create a Polars DataFrame from the parsed data.
        
        Returns:
        --------
        Polars DataFrame
        """
        if not self.parsed_data:
            raise ValueError("No parsed data available. Call parse_all_files() first.")
        
        rows = []
        
        for filename, data in self.parsed_data.items():
            header = data['header']
            arrays = data['arrays']
            
            # Create row for basic information
            row = {
                'filename': filename,
                'subject': header.get('subject'),
                'experiment': header.get('experiment'),
                'phase': header.get('phase'),
                'group': header.get('group'),
                'box': header.get('box'),
                'start_date': header.get('start_date'),
                'end_date': header.get('end_date'),
                'start_time': header.get('start_time'),
                'end_time': header.get('end_time'),
                'msn': header.get('msn'),
            }
            
            # Add single-letter variables
            for letter in 'FGHIJKLMNOPQRSTUVWXYZ':
                if letter in arrays and letter != 'T':  # Skip T as it's handled separately
                    row[f'{letter}_value'] = arrays[letter]
            
            # Process timestamp array (T) and response array (E) together
            if 'T' in arrays and 'E' in arrays:
                # Ensure t_array and e_array are lists
                t_array = arrays['T']
                e_array = arrays['E']
                
                if not isinstance(t_array, list):
                    t_array = [t_array]
                if not isinstance(e_array, list):
                    e_array = [e_array]
                
                # Make sure arrays are the same length (use the shorter one)
                min_length = min(len(t_array), len(e_array))
                t_array = t_array[:min_length]
                e_array = e_array[:min_length]
                
                # Store these arrays as lists in the row (Polars can handle lists in columns)
                row['timestamps'] = t_array
                row['responses'] = e_array
                
                # Calculate additional metrics
                # Active lever presses (typically response code 2 in E array)
                active_presses = sum(1 for resp in e_array if resp == 2)
                row['active_lever_presses'] = active_presses
                
                # Inactive lever presses (typically response code 1 in E array)
                inactive_presses = sum(1 for resp in e_array if resp == 1)
                row['inactive_lever_presses'] = inactive_presses
                
                # Reinforcers (G value typically represents this)
                row['reinforcers'] = arrays.get('G', 0)
                
                # Calculate lever presses in time bins (e.g., 30-minute bins)
                bin_size = 30 * 60  # 30 minutes in seconds
                max_time = max(t_array) if t_array else 0
                num_bins = int(np.ceil(max_time / bin_size))
                
                active_bins = [0] * num_bins
                inactive_bins = [0] * num_bins
                
                for i, (time, resp) in enumerate(zip(t_array, e_array)):
                    bin_idx = int(time // bin_size)
                    if bin_idx < num_bins:
                        if resp == 2:  # Active lever
                            active_bins[bin_idx] += 1
                        elif resp == 1:  # Inactive lever
                            inactive_bins[bin_idx] += 1
                
                row['active_30min_bins'] = active_bins
                row['inactive_30min_bins'] = inactive_bins
            
            rows.append(row)
        
        # Create Polars DataFrame
        df = pl.DataFrame(rows)
        self.combined_df = df
        return df
    
    def get_lever_presses_by_time(self, time_bin_minutes=30):
        """
        Get lever presses organized by time bins.
        
        Parameters:
        -----------
        time_bin_minutes : int
            Size of time bins in minutes
            
        Returns:
        --------
        Polars DataFrame with lever presses by time bins
        """
        if self.combined_df is None:
            raise ValueError("No combined DataFrame available. Call create_dataframe() first.")
        
        results = []
        
        # Convert Polars DataFrame to dictionaries for easier processing
        for row in self.combined_df.iter_rows(named=True):
            subject = row['subject']
            phase = row['phase']
            group = row['group']
            
            # Get timestamps and responses
            timestamps = row.get('timestamps', [])
            responses = row.get('responses', [])
            
            # Ensure they are lists
            if not isinstance(timestamps, list):
                timestamps = [timestamps]
            if not isinstance(responses, list):
                responses = [responses]
            
            # Make sure arrays are the same length
            min_length = min(len(timestamps), len(responses))
            timestamps = timestamps[:min_length]
            responses = responses[:min_length]
            
            # Calculate bins
            bin_size = time_bin_minutes * 60  # convert to seconds
            max_time = max(timestamps) if timestamps else 0
            num_bins = int(np.ceil(max_time / bin_size))
            
            active_bins = [0] * num_bins
            inactive_bins = [0] * num_bins
            
            for time, resp in zip(timestamps, responses):
                bin_idx = int(time // bin_size)
                if bin_idx < num_bins:
                    if resp == 2:  # Active lever
                        active_bins[bin_idx] += 1
                    elif resp == 1:  # Inactive lever
                        inactive_bins[bin_idx] += 1
            
            # Create a row for each time bin
            for bin_idx in range(max(len(active_bins), len(inactive_bins))):
                active_count = active_bins[bin_idx] if bin_idx < len(active_bins) else 0
                inactive_count = inactive_bins[bin_idx] if bin_idx < len(inactive_bins) else 0
                
                bin_start_min = bin_idx * time_bin_minutes
                bin_end_min = (bin_idx + 1) * time_bin_minutes
                
                results.append({
                    'subject': subject,
                    'phase': phase,
                    'group': group,
                    'bin_start_min': bin_start_min,
                    'bin_end_min': bin_end_min,
                    'active_lever_presses': active_count,
                    'inactive_lever_presses': inactive_count,
                    'session_time': f"{bin_start_min}-{bin_end_min} min"
                })
        
        return pl.DataFrame(results)
    
    def create_time_series_dataframe(self):
        """
        Create a long-format time series DataFrame with individual lever presses.
        
        Returns:
        --------
        Polars DataFrame with time series data
        """
        if self.combined_df is None:
            raise ValueError("No combined DataFrame available. Call create_dataframe() first.")
        
        results = []
        
        # Convert Polars DataFrame to dictionaries for easier processing
        for row in self.combined_df.iter_rows(named=True):
            subject = row['subject']
            phase = row['phase']
            group = row['group']
            
            timestamps = row.get('timestamps', [])
            responses = row.get('responses', [])
            
            # Ensure they are lists
            if not isinstance(timestamps, list):
                timestamps = [timestamps]
            if not isinstance(responses, list):
                responses = [responses]
            
            # Make sure arrays are the same length
            min_length = min(len(timestamps), len(responses))
            timestamps = timestamps[:min_length]
            responses = responses[:min_length]
            
            for time, resp in zip(timestamps, responses):
                # Determine response type
                if resp == 1:
                    response_type = 'inactive_lever'
                elif resp == 2:
                    response_type = 'active_lever'
                elif resp == 6:
                    response_type = 'reinforced'
                else:
                    response_type = f'other_{int(resp)}'
                
                results.append({
                    'subject': subject,
                    'phase': phase,
                    'group': group,
                    'time_seconds': time,
                    'time_minutes': time / 60,
                    'response_code': resp,
                    'response_type': response_type
                })
        
        return pl.DataFrame(results)
    
    def save_data(self, output_path):
        """
        Save the processed data to CSV files using Polars.
        
        Parameters:
        -----------
        output_path : str or Path
            Directory to save output files
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main dataframe
        if self.combined_df is not None:
            # Save basic info (exclude array columns)
            array_cols = ['timestamps', 'responses', 'active_30min_bins', 'inactive_30min_bins']
            basic_cols = [col for col in self.combined_df.columns if col not in array_cols]
            
            basic_df = self.combined_df.select(basic_cols)
            basic_df.write_csv(output_path / 'summary_data.csv')
            
            # Save time-binned data for different bin sizes
            for bin_size in [5, 10, 30]:
                time_binned_df = self.get_lever_presses_by_time(bin_size)
                time_binned_df.write_csv(output_path / f'time_binned_data_{bin_size}min.csv')
            
            # Save time series data
            time_series_df = self.create_time_series_dataframe()
            time_series_df.write_csv(output_path / 'time_series_data.csv')
            
            print(f"Data saved to {output_path}")
    
    def merge_with_experiment_metadata(self, metadata_file):
        """
        Merge the processed data with experiment metadata from a CSV file.
        
        Parameters:
        -----------
        metadata_file : str or Path
            Path to the CSV file containing experiment metadata
            
        Returns:
        --------
        Polars DataFrame with merged data
        """
        if self.combined_df is None:
            raise ValueError("No combined DataFrame available. Call create_dataframe() first.")
        
        # Read metadata file
        metadata_df = pl.read_csv(metadata_file)
        
        # Merge based on subject ID
        merged_df = self.combined_df.join(
            metadata_df,
            left_on="subject",
            right_on="subject_id",
            how="left"
        )
        
        return merged_df
    
    def analyze_by_phase(self):
        """
        Analyze data by experimental phase (SelfAdmin, EXT, REI).
        
        Returns:
        --------
        Dict of Polars DataFrames for each phase
        """
        if self.combined_df is None:
            raise ValueError("No combined DataFrame available. Call create_dataframe() first.")
        
        # Group data by phase
        phases = {}
        
        # Get unique phases
        unique_phases = self.combined_df.select('phase').unique().to_series().to_list()
        
        for phase in unique_phases:
            if phase is None:
                continue
                
            # Filter data for this phase
            phase_df = self.combined_df.filter(pl.col('phase') == phase)
            phases[phase] = phase_df
        
        return phases
    
    def generate_summary_statistics(self):
        """
        Generate summary statistics for lever presses and reinforcers.
        
        Returns:
        --------
        Polars DataFrame with summary statistics
        """
        if self.combined_df is None:
            raise ValueError("No combined DataFrame available. Call create_dataframe() first.")
        
        # Group by subject, phase, and group
        grouped = self.combined_df.group_by(['subject', 'phase', 'group'])
        
        # Calculate summary statistics
        summary = grouped.agg([
            pl.col('active_lever_presses').mean().alias('mean_active_presses'),
            pl.col('active_lever_presses').std().alias('std_active_presses'),
            pl.col('inactive_lever_presses').mean().alias('mean_inactive_presses'),
            pl.col('inactive_lever_presses').std().alias('std_inactive_presses'),
            pl.col('reinforcers').mean().alias('mean_reinforcers'),
            pl.col('reinforcers').std().alias('std_reinforcers')
        ])
        
        return summary

# Main execution block
if __name__ == "__main__":
    import argparse
    
    # Create command-line argument parser
    arg_parser = argparse.ArgumentParser(description='Parse MedPC data files')
    arg_parser.add_argument('--data_dir', default='./data', help='Directory containing MedPC data files')
    arg_parser.add_argument('--output_dir', default='./processed_data', help='Directory to save processed data')
    arg_parser.add_argument('--bin_size', type=int, default=30, help='Time bin size in minutes')
    args = arg_parser.parse_args()
    
    # Initialize parser
    parser = MedPCDataParser()
    
    # Set the directory where your MedPC files are located
    parser.base_dir = Path(args.data_dir)
    
    # Find all txt files
    files = parser.find_files('*.txt')
    print(f"Found {len(files)} MedPC data files")
    
    # Parse all files
    parser.parse_all_files()
    print("All files parsed successfully")
    
    try:
        # Create a DataFrame from the parsed data
        df = parser.create_dataframe()
        print(f"Created DataFrame with {len(df)} rows")
        
        # Get data organized in time bins
        time_binned_df = parser.get_lever_presses_by_time(args.bin_size)
        print(f"Created time-binned DataFrame with {len(time_binned_df)} rows")
        
        # Create a time series DataFrame
        time_series_df = parser.create_time_series_dataframe()
        print(f"Created time series DataFrame with {len(time_series_df)} rows")
        
        # Generate summary statistics
        summary_stats = parser.generate_summary_statistics()
        print(f"Generated summary statistics")
        
        # Save the processed data
        parser.save_data(args.output_dir)
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        import traceback
        traceback.print_exc()