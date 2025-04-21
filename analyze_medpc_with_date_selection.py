import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from improved_medpc_parser import MedPCDataParser

class MedPCAnalyzer:
    def __init__(self, data_dir="./data", output_dir="./analysis_output"):
        """
        Initialize the MedPC data analyzer.
        
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
        
        # Initialize the parser
        self.parser = MedPCDataParser(self.data_dir)
        
        # Data containers
        self.summary_df = None
        self.binned_df = {}
        self.time_series_df = None
        self.phase_data = {}
        
        # Available dates
        self.available_dates = []
    
    def load_and_process_data(self, bin_sizes=[5, 10, 30], selected_dates=None, selected_subjects=None):
        """
        Load and process MedPC data files.
        
        Parameters:
        -----------
        bin_sizes : list
            List of time bin sizes in minutes
        selected_dates : list
            List of dates to include in the analysis (format: "YYYY-MM-DD")
            If None, all dates will be included
        selected_subjects : list
            List of subjects to include in the analysis
            If None, all subjects will be included
        """
        # Find and parse data files
        files = self.parser.find_files()
        print(f"Found {len(files)} MedPC data files")
        
        # Parse all files
        self.parser.parse_all_files()
        
        # Create summary dataframe
        self.summary_df = self.parser.create_dataframe()
        
        # Extract available dates from filenames
        dates = []
        for name in self.summary_df.select('filename').to_series():
            # Extract date from filename (format: YYYY-MM-DD_HHhMMm_Subject XX.txt)
            try:
                date_str = name.split('_')[0]
                dates.append(date_str)
            except:
                pass
        
        self.available_dates = sorted(list(set(dates)))
        print(f"Available dates: {', '.join(self.available_dates)}")
        
        # Filter by selected dates if provided
        if selected_dates is not None:
            # Create a mask for rows that match any of the selected dates
            date_filter = pl.lit(False)
            for date in selected_dates:
                date_filter = date_filter | pl.col('filename').str.contains(date)
            
            filtered_df = self.summary_df.filter(date_filter)
            
            if len(filtered_df) > 0:
                self.summary_df = filtered_df
                print(f"Filtered data to {len(selected_dates)} dates with {len(self.summary_df)} sessions")
            else:
                print("No data found for selected dates. Using all data.")
        
        # Filter by selected subjects if provided
        if selected_subjects is not None:
            filtered_df = self.summary_df.filter(pl.col('subject').is_in(selected_subjects))
            if len(filtered_df) > 0:
                self.summary_df = filtered_df
                print(f"Filtered data to {len(selected_subjects)} subjects with {len(self.summary_df)} sessions")
            else:
                print("No data found for selected subjects. Using all data.")
        
        print(f"Using {len(self.summary_df)} sessions for analysis")
        
        # Generate binned data for different bin sizes
        self.binned_df = {}
        for bin_size in bin_sizes:
            df = self.parser.get_lever_presses_by_time(bin_size)
            
            # Filter by date and subject if needed
            if selected_dates is not None:
                # Build a regex pattern that matches any of the selected dates
                pattern = "|".join(selected_dates)
                
                # Filter rows where session_time contains any of the dates
                df = df.filter(pl.col('session_time').str.contains(pattern))
            
            if selected_subjects is not None:
                df = df.filter(pl.col('subject').is_in(selected_subjects))
            
            self.binned_df[bin_size] = df
            print(f"Created {bin_size}-min binned DataFrame with {len(df)} rows")
        
        # Generate time series data
        self.time_series_df = self.parser.create_time_series_dataframe()
        
        # Filter by date and subject if needed
        if selected_dates is not None or selected_subjects is not None:
            # Create filters for dates and subjects
            date_filter = pl.lit(True)
            subject_filter = pl.lit(True)
            
            if selected_dates is not None:
                # Create a regex pattern for all selected dates
                pattern = "|".join(selected_dates)
                date_filter = pl.col('filename').str.contains(pattern)
            
            if selected_subjects is not None:
                subject_filter = pl.col('subject').is_in(selected_subjects)
            
            # Apply filters
            self.time_series_df = self.time_series_df.filter(date_filter & subject_filter)
        
        print(f"Created time series DataFrame with {len(self.time_series_df)} rows")
        
        # Group data by experimental phase
        self.phase_data = {}
        phases = self.summary_df.select('phase').unique().to_series().to_list()
        for phase in phases:
            if phase is None:
                continue
            phase_df = self.summary_df.filter(pl.col('phase') == phase)
            self.phase_data[phase] = phase_df
            print(f"Phase {phase}: {len(phase_df)} sessions")
    
    def list_available_dates(self):
        """
        List all available experiment dates.
        
        Returns:
        --------
        list of dates (str)
        """
        if not self.available_dates:
            # Parse files if not already done
            if self.summary_df is None:
                files = self.parser.find_files()
                self.parser.parse_all_files()
                self.summary_df = self.parser.create_dataframe()
            
            # Extract dates from filenames
            dates = []
            for name in self.summary_df.select('filename').to_series():
                try:
                    date_str = name.split('_')[0]
                    dates.append(date_str)
                except:
                    pass
            
            self.available_dates = sorted(list(set(dates)))
        
        return self.available_dates
    
    def save_processed_data(self, selected_dates=None):
        """
        Save processed data to CSV files.
        
        Parameters:
        -----------
        selected_dates : list
            List of dates to include in the saved data (format: "YYYY-MM-DD")
        """
        # Create output directory with date suffix if using selected dates
        output_dir = self.output_dir
        if selected_dates:
            date_suffix = "_".join(selected_dates)
            output_dir = self.output_dir / f"selected_dates_{date_suffix}"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a parser instance with filtered data
        filtered_parser = MedPCDataParser()
        filtered_parser.parsed_data = self.parser.parsed_data
        filtered_parser.combined_df = self.summary_df
        
        # Save via the parser
        filtered_parser.save_data(output_dir)
        print(f"Data saved to {output_dir}")
    
    def plot_lever_presses_by_phase(self, bin_size=30, selected_dates=None):
        """
        Plot active and inactive lever presses by experimental phase.
        
        Parameters:
        -----------
        bin_size : int
            Time bin size in minutes
        selected_dates : list
            List of dates to include in the plot (format: "YYYY-MM-DD")
        """
        if bin_size not in self.binned_df:
            raise ValueError(f"No data available for {bin_size}-min bins")
        
        binned_data = self.binned_df[bin_size]
        
        # Create suffix for filename based on selected dates
        date_suffix = ""
        if selected_dates:
            date_suffix = "_" + "_".join(selected_dates)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        phases = ['SelfAdmin', 'EXT', 'REI']
        
        for i, phase in enumerate(phases):
            # Filter data for this phase
            phase_data = binned_data.filter(pl.col('phase') == phase)
            
            if len(phase_data) == 0:
                axes[i].set_title(f"{phase} - No data")
                continue
            
            # Group by time bin and calculate mean and SEM
            grouped = phase_data.group_by(['bin_start_min', 'bin_end_min']).sort(['bin_start_min'])
            
            mean_data = grouped.agg([
                pl.col('active_lever_presses').mean().alias('active_mean'),
                pl.col('inactive_lever_presses').mean().alias('inactive_mean'),
                pl.col('active_lever_presses').std().alias('active_std'),
                pl.col('inactive_lever_presses').std().alias('inactive_std'),
                pl.col('active_lever_presses').count().alias('count')
            ])
            
            # Convert to lists for plotting
            bin_centers = mean_data.select(pl.col('bin_start_min') + bin_size/2).to_series().to_list()
            active_means = mean_data.select('active_mean').to_series().to_list()
            inactive_means = mean_data.select('inactive_mean').to_series().to_list()
            
            # Calculate SEM
            active_stds = mean_data.select('active_std').to_series().to_list()
            inactive_stds = mean_data.select('inactive_std').to_series().to_list()
            counts = mean_data.select('count').to_series().to_list()
            
            active_sems = [std / np.sqrt(count) if count > 1 else 0 for std, count in zip(active_stds, counts)]
            inactive_sems = [std / np.sqrt(count) if count > 1 else 0 for std, count in zip(inactive_stds, counts)]
            
            # Plot active lever presses
            axes[i].errorbar(
                bin_centers,
                active_means,
                yerr=active_sems,
                fmt='o-',
                label='Active Lever',
                color='blue'
            )
            
            # Plot inactive lever presses
            axes[i].errorbar(
                bin_centers,
                inactive_means,
                yerr=inactive_sems,
                fmt='s--',
                label='Inactive Lever',
                color='red'
            )
            
            # Customize plot
            title = f"{phase}"
            if selected_dates:
                title += f" ({', '.join(selected_dates)})"
            
            axes[i].set_title(title)
            axes[i].set_xlabel('Time (minutes)')
            
            if i == 0:
                axes[i].set_ylabel('Lever Presses (mean ± SEM)')
                axes[i].legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.output_dir / f'lever_presses_by_phase_{bin_size}min{date_suffix}.png', dpi=300)
        plt.close()
        
        print(f"Saved lever presses by phase plot ({bin_size}-min bins)")
    
    def plot_response_rate_over_time(self, bin_size=5, selected_dates=None):
        """
        Plot response rate over time for each experimental phase.
        
        Parameters:
        -----------
        bin_size : int
            Time bin size in minutes
        selected_dates : list
            List of dates to include in the plot (format: "YYYY-MM-DD")
        """
        if bin_size not in self.binned_df:
            raise ValueError(f"No data available for {bin_size}-min bins")
        
        binned_data = self.binned_df[bin_size]
        
        # Create suffix for filename based on selected dates
        date_suffix = ""
        if selected_dates:
            date_suffix = "_" + "_".join(selected_dates)
        
        # Calculate response rate (responses per minute)
        binned_data = binned_data.with_columns([
            (pl.col('active_lever_presses') / bin_size).alias('active_rate'),
            (pl.col('inactive_lever_presses') / bin_size).alias('inactive_rate')
        ])
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        phases = ['SelfAdmin', 'EXT', 'REI']
        
        for i, phase in enumerate(phases):
            # Filter data for this phase
            phase_data = binned_data.filter(pl.col('phase') == phase)
            
            if len(phase_data) == 0:
                axes[i].set_title(f"{phase} - No data")
                continue
            
            # Group by time bin and calculate mean and SEM
            grouped = phase_data.group_by(['bin_start_min', 'bin_end_min']).sort(['bin_start_min'])
            
            mean_data = grouped.agg([
                pl.col('active_rate').mean().alias('active_mean'),
                pl.col('inactive_rate').mean().alias('inactive_mean'),
                pl.col('active_rate').std().alias('active_std'),
                pl.col('inactive_rate').std().alias('inactive_std'),
                pl.col('active_rate').count().alias('count')
            ])
            
            # Convert to lists for plotting
            bin_centers = mean_data.select(pl.col('bin_start_min') + bin_size/2).to_series().to_list()
            active_means = mean_data.select('active_mean').to_series().to_list()
            inactive_means = mean_data.select('inactive_mean').to_series().to_list()
            
            # Calculate SEM
            active_stds = mean_data.select('active_std').to_series().to_list()
            inactive_stds = mean_data.select('inactive_std').to_series().to_list()
            counts = mean_data.select('count').to_series().to_list()
            
            active_sems = [std / np.sqrt(count) if count > 1 else 0 for std, count in zip(active_stds, counts)]
            inactive_sems = [std / np.sqrt(count) if count > 1 else 0 for std, count in zip(inactive_stds, counts)]
            
            # Plot active lever presses
            axes[i].errorbar(
                bin_centers,
                active_means,
                yerr=active_sems,
                fmt='o-',
                label='Active Lever',
                color='blue'
            )
            
            # Plot inactive lever presses
            axes[i].errorbar(
                bin_centers,
                inactive_means,
                yerr=inactive_sems,
                fmt='s--',
                label='Inactive Lever',
                color='red'
            )
            
            # Customize plot
            title = f"{phase}"
            if selected_dates:
                title += f" ({', '.join(selected_dates)})"
            
            axes[i].set_title(title)
            axes[i].set_xlabel('Time (minutes)')
            
            if i == 0:
                axes[i].set_ylabel('Response Rate (per minute)')
                axes[i].legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.output_dir / f'response_rate_by_phase_{bin_size}min{date_suffix}.png', dpi=300)
        plt.close()
        
        print(f"Saved response rate by phase plot ({bin_size}-min bins)")
    
    def plot_time_series_data(self, selected_dates=None, selected_subjects=None):
        """
        Plot time series data for active and inactive lever presses.
        
        Parameters:
        -----------
        selected_dates : list
            List of dates to include in the plot (format: "YYYY-MM-DD")
        selected_subjects : list
            List of subjects to plot. If None, all subjects will be plotted.
        """
        if self.time_series_df is None:
            raise ValueError("No time series data available")
        
        # Create suffix for filename based on selected dates
        date_suffix = ""
        if selected_dates:
            date_suffix = "_" + "_".join(selected_dates)
        
        # Get unique subjects and phases
        subjects = self.time_series_df.select('subject').unique().to_series().to_list()
        
        # Filter by selected subjects if provided
        if selected_subjects:
            subjects = [s for s in subjects if s in selected_subjects]
        
        phases = self.time_series_df.select('phase').unique().to_series().to_list()
        
        for subject in subjects:
            # Filter for this subject
            subject_data = self.time_series_df.filter(pl.col('subject') == subject)
            
            # Create figure
            n_phases = len([p for p in phases if p is not None])
            fig, axes = plt.subplots(n_phases, 1, figsize=(12, 4 * n_phases), sharex=True)
            if n_phases == 1:
                axes = [axes]
            
            plot_idx = 0
            for phase in phases:
                if phase is None:
                    continue
                    
                # Filter data for this phase
                phase_data = subject_data.filter(pl.col('phase') == phase)
                
                if len(phase_data) == 0:
                    axes[plot_idx].set_title(f"Subject {subject}, {phase} - No data")
                    plot_idx += 1
                    continue
                
                # Plot active lever presses
                active_data = phase_data.filter(pl.col('response_type') == 'active_lever')
                if len(active_data) > 0:
                    active_times = active_data.select('time_minutes').to_series().to_list()
                    axes[plot_idx].plot(active_times, 
                                        np.ones(len(active_times)) * 1.0, 
                                        '|', markersize=10, color='blue', label='Active Lever')
                
                # Plot inactive lever presses
                inactive_data = phase_data.filter(pl.col('response_type') == 'inactive_lever')
                if len(inactive_data) > 0:
                    inactive_times = inactive_data.select('time_minutes').to_series().to_list()
                    axes[plot_idx].plot(inactive_times, 
                                        np.ones(len(inactive_times)) * 0.8, 
                                        '|', markersize=10, color='red', label='Inactive Lever')
                
                # Plot reinforcers
                reinforced_data = phase_data.filter(pl.col('response_type') == 'reinforced')
                if len(reinforced_data) > 0:
                    reinforced_times = reinforced_data.select('time_minutes').to_series().to_list()
                    axes[plot_idx].plot(reinforced_times, 
                                        np.ones(len(reinforced_times)) * 0.6, 
                                        'v', markersize=8, color='green', label='Reinforcers')
                
                # Customize plot
                title = f"Subject {subject}, {phase}"
                if selected_dates:
                    title += f" ({', '.join(selected_dates)})"
                
                axes[plot_idx].set_title(title)
                axes[plot_idx].set_yticks([0.6, 0.8, 1.0])
                axes[plot_idx].set_yticklabels(['Reinforcers', 'Inactive', 'Active'])
                axes[plot_idx].set_ylim(0.4, 1.2)
                
                if plot_idx == 0:
                    axes[plot_idx].legend(loc='upper right')
                
                if plot_idx == n_phases - 1:
                    axes[plot_idx].set_xlabel('Time (minutes)')
                
                plot_idx += 1
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(self.output_dir / f'time_series_subject_{subject}{date_suffix}.png', dpi=300)
            plt.close()
        
        print(f"Saved time series plots for {len(subjects)} subjects")
    
    def plot_cumulative_responses(self, selected_dates=None, selected_subjects=None):
        """
        Plot cumulative active and inactive lever presses over time.
        
        Parameters:
        -----------
        selected_dates : list
            List of dates to include in the plot (format: "YYYY-MM-DD")
        selected_subjects : list
            List of subjects to plot. If None, all subjects will be plotted.
        """
        if self.time_series_df is None:
            raise ValueError("No time series data available")
        
        # Create suffix for filename based on selected dates
        date_suffix = ""
        if selected_dates:
            date_suffix = "_" + "_".join(selected_dates)
        
        # Get unique subjects and phases
        subjects = self.time_series_df.select('subject').unique().to_series().to_list()
        
        # Filter by selected subjects if provided
        if selected_subjects:
            subjects = [s for s in subjects if s in selected_subjects]
        
        phases = self.time_series_df.select('phase').unique().to_series().to_list()
        
        for subject in subjects:
            # Filter for this subject
            subject_data = self.time_series_df.filter(pl.col('subject') == subject)
            
            # Create figure
            n_phases = len([p for p in phases if p is not None])
            fig, axes = plt.subplots(n_phases, 1, figsize=(12, 4 * n_phases), sharex=True)
            if n_phases == 1:
                axes = [axes]
            
            plot_idx = 0
            for phase in phases:
                if phase is None:
                    continue
                    
                # Filter data for this phase
                phase_data = subject_data.filter(pl.col('phase') == phase)
                
                if len(phase_data) == 0:
                    axes[plot_idx].set_title(f"Subject {subject}, {phase} - No data")
                    plot_idx += 1
                    continue
                
                # Calculate cumulative responses for each type
                response_types = {
                    'active_lever': {'color': 'blue', 'style': '-', 'label': 'Active Lever'},
                    'inactive_lever': {'color': 'red', 'style': '--', 'label': 'Inactive Lever'},
                    'reinforced': {'color': 'green', 'style': ':', 'label': 'Reinforcers'}
                }
                
                for resp_type, style in response_types.items():
                    # Filter for this response type
                    resp_data = phase_data.filter(pl.col('response_type') == resp_type)
                    
                    if len(resp_data) > 0:
                        # Sort by time
                        resp_data = resp_data.sort('time_minutes')
                        
                        # Get times
                        times = resp_data.select('time_minutes').to_series().to_list()
                        
                        # Calculate cumulative counts
                        cumulative = list(range(1, len(times) + 1))
                        
                        # Plot
                        axes[plot_idx].step(times, cumulative, 
                                            style['style'], color=style['color'], 
                                            label=style['label'], where='post')
                
                # Customize plot
                title = f"Subject {subject}, {phase} - Cumulative Responses"
                if selected_dates:
                    title += f" ({', '.join(selected_dates)})"
                
                axes[plot_idx].set_title(title)
                
                if plot_idx == 0:
                    axes[plot_idx].legend(loc='upper left')
                
                if plot_idx == n_phases - 1:
                    axes[plot_idx].set_xlabel('Time (minutes)')
                
                axes[plot_idx].set_ylabel('Cumulative Responses')
                axes[plot_idx].grid(True, linestyle='--', alpha=0.7)
                
                plot_idx += 1
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(self.output_dir / f'cumulative_responses_subject_{subject}{date_suffix}.png', dpi=300)
            plt.close()
        
        print(f"Saved cumulative response plots for {len(subjects)} subjects")
    
    def plot_summary_by_phase(self, selected_dates=None):
        """
        Plot summary statistics by experimental phase.
        
        Parameters:
        -----------
        selected_dates : list
            List of dates to include in the plot (format: "YYYY-MM-DD")
        """
        if self.summary_df is None:
            raise ValueError("No summary data available")
        
        # Create suffix for filename based on selected dates
        date_suffix = ""
        if selected_dates:
            date_suffix = "_" + "_".join(selected_dates)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot metrics by phase
        metrics = [
            ('active_lever_presses', 'Active Lever Presses'),
            ('inactive_lever_presses', 'Inactive Lever Presses'),
            ('reinforcers', 'Reinforcers')
        ]
        
        for i, (metric, label) in enumerate(metrics):
            # Get data for each phase
            phases = self.summary_df.select('phase').unique().to_series().to_list()
            phases = [p for p in phases if p is not None]
            
            box_data = []
            for phase in phases:
                # Get values for this phase and metric
                phase_data = self.summary_df.filter(pl.col('phase') == phase)
                values = phase_data.select(metric).to_series().to_list()
                box_data.append(values)
            
            # Create box plot
            box = axes[i].boxplot(box_data, labels=phases, patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightgreen', 'coral']
            for patch, color in zip(box['boxes'], colors[:len(phases)]):
                patch.set_facecolor(color)
            
            # Add individual data points (jittered)
            for j, values in enumerate(box_data):
                # Add jitter
                x = np.random.normal(j + 1, 0.04, size=len(values))
                axes[i].scatter(x, values, alpha=0.6, color='black', s=20)
            
            # Customize plot
            title = label
            if selected_dates:
                title += f" ({', '.join(selected_dates)})"
            
            axes[i].set_title(title)
            axes[i].set_xlabel('Experimental Phase')
            axes[i].set_ylabel('Count')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.output_dir / f'summary_by_phase{date_suffix}.png', dpi=300)
        plt.close()
        
        print("Saved summary statistics by phase plot")
    
    def compare_active_vs_inactive(self, selected_dates=None):
        """
        Compare active vs inactive lever presses for each subject and phase.
        
        Parameters:
        -----------
        selected_dates : list
            List of dates to include in the plot (format: "YYYY-MM-DD")
        """
        if self.summary_df is None:
            raise ValueError("No summary data available")
        
        # Create suffix for filename based on selected dates
        date_suffix = ""
        if selected_dates:
            date_suffix = "_" + "_".join(selected_dates)
        
        # Create a copy of summary_df to avoid modifying the original
        summary_df = self.summary_df.clone()
        
        # Calculate active-inactive ratio (handle division by zero)
        summary_df = summary_df.with_columns([
            pl.when(pl.col('inactive_lever_presses') == 0)
            .then(pl.col('active_lever_presses').cast(pl.Float64))
            .otherwise(pl.col('active_lever_presses') / pl.col('inactive_lever_presses'))
            .alias('active_inactive_ratio')
        ])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get data for box plot
        phases = summary_df.select('phase').unique().to_series().to_list()
        phases = [p for p in phases if p is not None]
        
        box_data = []
        for phase in phases:
            # Get ratios for this phase
            phase_data = summary_df.filter(pl.col('phase') == phase)
            ratios = phase_data.select('active_inactive_ratio').to_series().to_list()
            box_data.append(ratios)
        
        # Create box plot
        box = ax.boxplot(box_data, labels=phases, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'coral']
        for patch, color in zip(box['boxes'], colors[:len(phases)]):
            patch.set_facecolor(color)
        
        # Add individual data points (jittered)
        for i, ratios in enumerate(box_data):
            # Add jitter
            x = np.random.normal(i + 1, 0.04, size=len(ratios))
            ax.scatter(x, ratios, alpha=0.6, color='black', s=20)
        
        # Add horizontal line at ratio = 1 (equal active and inactive)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7)

        # Define title
        title = 'Active vs Inactive Lever Press Ratio'
        if selected_dates:
            title += f" ({', '.join(selected_dates)})"
        
        ax.set_title(title)
        ax.set_xlabel('Experimental Phase')
        ax.set_ylabel('Active/Inactive Ratio')
        ax.set_yscale('log')  # Log scale for better visualization
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(self.output_dir / f'active_vs_inactive_ratio{date_suffix}.png', dpi=300)
        plt.close()
        
        print("Saved active vs inactive ratio plot")
    
    def run_analysis(self, bin_sizes=[5, 10, 30], selected_dates=None, selected_subjects=None):
        """
        Run the full analysis workflow.
        
        Parameters:
        -----------
        bin_sizes : list
            List of time bin sizes in minutes
        selected_dates : list
            List of dates to include in the analysis (format: "YYYY-MM-DD")
        selected_subjects : list
            List of subjects to analyze. If None, all subjects will be analyzed.
        """
        # Load and process data
        self.load_and_process_data(bin_sizes, selected_dates, selected_subjects)
        
        # Save processed data
        self.save_processed_data(selected_dates)
        
        # Generate plots
        for bin_size in bin_sizes:
            self.plot_lever_presses_by_phase(bin_size, selected_dates)
            self.plot_response_rate_over_time(bin_size, selected_dates)
        
        self.plot_time_series_data(selected_dates, selected_subjects)
        self.plot_cumulative_responses(selected_dates, selected_subjects)
        self.plot_summary_by_phase(selected_dates)
        self.compare_active_vs_inactive(selected_dates)
        
        print("Analysis complete!")


# Example usage in a Jupyter notebook:
# 
# ```python
# from analyze_medpc_with_date_selection import MedPCAnalyzer
# 
# # Initialize analyzer
# analyzer = MedPCAnalyzer('./data', './analysis_output')
# 
# # List available dates
# dates = analyzer.list_available_dates()
# print(f"Available dates: {dates}")
# 
# # Run analysis for specific dates
# selected_dates = ['2025-04-14']  # Single date
# analyzer.run_analysis(selected_dates=selected_dates)
# 
# # Run analysis for specific subjects on specific dates
# selected_subjects = [73, 74, 75]
# analyzer.run_analysis(selected_dates=selected_dates, selected_subjects=selected_subjects)
# ```

# Main execution block
if __name__ == "__main__":
    import argparse
    
    # Create command-line argument parser
    arg_parser = argparse.ArgumentParser(description='Analyze MedPC data with date selection')
    arg_parser.add_argument('--data_dir', default='./data', help='Directory containing MedPC data files')
    arg_parser.add_argument('--output_dir', default='./analysis_output', help='Directory to save analysis outputs')
    arg_parser.add_argument('--bin_sizes', type=int, nargs='+', default=[5, 10, 30], help='Time bin sizes in minutes')
    arg_parser.add_argument('--dates', type=str, nargs='+', help='List of dates to analyze (format: YYYY-MM-DD)')
    arg_parser.add_argument('--subjects', type=int, nargs='+', help='List of subjects to analyze')
    arg_parser.add_argument('--list_dates', action='store_true', help='List available experiment dates and exit')
    args = arg_parser.parse_args()
    
    # Create analyzer
    analyzer = MedPCAnalyzer(args.data_dir, args.output_dir)
    
    # List available dates if requested
    if args.list_dates:
        dates = analyzer.list_available_dates()
        print("Available experiment dates:")
        for date in dates:
            print(f"  {date}")
        exit(0)
    
    # Run analysis
    try:
        analyzer.run_analysis(args.bin_sizes, args.dates, args.subjects)
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()