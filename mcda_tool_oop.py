import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os
import shutil

class DataGenerator:
    """
    Handles modular generation of synthetic multi-criteria datasets.
    Supports Uniform, Normal, and Exponential distributions per criterion.
    """
    def __init__(self, num_configurations: int = 200, seed: int = 42):
        self.num_configurations = num_configurations
        self.seed = seed
        np.random.seed(self.seed)

    def generate(self, criteria_config: list) -> pd.DataFrame:
        """
        Generates synthetic data based on a list of criterion configurations.
        criteria_config example:
        [
            {
                'name': 'Storage_Capacity_MWh',
                'dist': 'uniform',
                'params': {'low': 10, 'high': 150}
            },
            ...
        ]
        """
        data = {}
        for config in criteria_config:
            name = config['name']
            dist = config['dist']
            params = config['params']
            
            if dist == 'uniform':
                data[name] = np.random.uniform(low=params['low'], high=params['high'], size=self.num_configurations)
            elif dist == 'normal':
                vals = np.random.normal(loc=params['loc'], scale=params['scale'], size=self.num_configurations)
                if 'clip_low' in params or 'clip_high' in params:
                    vals = np.clip(vals, params.get('clip_low', -np.inf), params.get('clip_high', np.inf))
                data[name] = vals
            elif dist == 'exponential':
                vals = np.random.exponential(scale=params['scale'], size=self.num_configurations)
                if 'shift' in params:
                    vals += params['shift']
                data[name] = vals
            else:
                raise ValueError(f"Unsupported distribution type: {dist}")
                
        df = pd.DataFrame(data)
        df.insert(0, 'Config_ID', [f'Config_{i}' for i in range(1, self.num_configurations + 1)])
        return df


class MCDAAnalyzer:
    """
    Encapsulates multi-criteria evaluation algorithms including normalization,
    weighted scoring, Pareto efficiency, and cluster-based profiling.
    """
    def __init__(self, data_df: pd.DataFrame, criteria_meta: dict):
        """
        criteria_meta example:
        {
            'Storage_Capacity_MWh': 'maximize',
            'Transmission_Losses_Percent': 'minimize',
            ...
        }
        """
        self.raw_df = data_df.copy()
        self.criteria_meta = criteria_meta
        self.criteria_cols = list(criteria_meta.keys())
        self.norm_df = pd.DataFrame()
        self.norm_df['Config_ID'] = self.raw_df['Config_ID']
        
        self._normalize()

    def _normalize(self):
        """
        Normalizes all criteria to [0, 1] interval.
        Correctly handles minimization (lower is better -> 1) and maximization.
        """
        for col, direction in self.criteria_meta.items():
            min_val = self.raw_df[col].min()
            max_val = self.raw_df[col].max()
            diff = max_val - min_val if max_val != min_val else 1.0
            
            if direction == 'maximize':
                self.norm_df[f'n_{col}'] = (self.raw_df[col] - min_val) / diff
            elif direction == 'minimize':
                self.norm_df[f'n_{col}'] = (max_val - self.raw_df[col]) / diff
            else:
                raise ValueError(f"Invalid direction '{direction}' for column {col}")
                
        self.norm_cols = [f'n_{col}' for col in self.criteria_cols]

    def calculate_scores(self, weights: np.ndarray) -> pd.Series:
        """Calculates linear weighted sum scores for all configurations."""
        assert len(weights) == len(self.norm_cols), "Weights length must match criteria count."
        assert np.isclose(np.sum(weights), 1.0), "Sum of weights must equal 1."
        return self.norm_df[self.norm_cols].dot(weights)

    def find_pareto_frontier(self) -> np.ndarray:
        """
        Computes Pareto optimal frontier (non-dominated configurations).
        Operates on normalized matrix where higher value is always better.
        """
        costs = self.norm_df[self.norm_cols].values
        num_configs = costs.shape[0]
        is_efficient = np.ones(num_configs, dtype=bool)
        
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Keep configurations that are better in at least one objective or equal
                is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1) | \
                                             np.all(costs[is_efficient] == c, axis=1)
                is_efficient[i] = True
        return is_efficient

    def run_clustering(self, n_clusters: int = 3, seed: int = 42) -> tuple:
        """Clusters configurations using KMeans based on their multi-criteria profiles."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(self.norm_df[self.norm_cols].values)
        return labels, kmeans


class DecisionProfiler:
    """
    Analyzes, profiles, and evaluates the relationships and stability of
    configurations within the multi-criteria decision space.
    """
    def __init__(self, raw_df: pd.DataFrame, norm_df: pd.DataFrame, criteria_cols: list, norm_cols: list):
        self.raw_df = raw_df
        self.norm_df = norm_df
        self.criteria_cols = criteria_cols
        self.norm_cols = norm_cols

    def get_cluster_profiles(self) -> pd.DataFrame:
        """Computes cluster averages for both raw and normalized metrics."""
        raw_profiles = self.raw_df.groupby('Cluster')[self.criteria_cols].mean()
        norm_profiles = self.norm_df.groupby('Cluster')[self.norm_cols].mean()
        return raw_profiles, norm_profiles

    def get_pareto_cooccurrence(self) -> pd.DataFrame:
        """Measures indicator co-occurrence/correlations along the Pareto frontier."""
        pareto_raw = self.raw_df[self.raw_df['Is_Pareto'] == True]
        return pareto_raw[self.criteria_cols].corr()


class GlobalSensitivitySimulator:
    """
    Performs global sensitivity analysis (Monte Carlo weight spaces via Dirichlet)
    and evaluates alternative decision stability across different weight priorities.
    """
    def __init__(self, analyzer: MCDAAnalyzer, criteria_cols: list):
        self.analyzer = analyzer
        self.criteria_cols = criteria_cols
        self.norm_cols = analyzer.norm_cols

    def run_simulation(self, n_simulations: int = 1000, target_col_idx: int = 2) -> pd.DataFrame:
        """
        Simulates randomized weight vectors using Dirichlet distribution.
        Tracks winning configurations and categorizes target criteria weight.
        """
        results = []
        norm_matrix = self.analyzer.norm_df[self.norm_cols].values
        config_ids = self.analyzer.raw_df['Config_ID'].values
        
        for _ in range(n_simulations):
            weights = np.random.dirichlet(np.ones(len(self.norm_cols)))
            scores = np.dot(norm_matrix, weights)
            best_idx = np.argmax(scores)
            
            target_w = weights[target_col_idx]
            
            # Classification of weight importance (Task 3.3)
            if target_w <= 0.33:
                importance = 'Ниска важност (0-33%)'
            elif target_w <= 0.66:
                importance = 'Средна важност (33-66%)'
            else:
                importance = 'Висока важност (>66%)'
                
            results.append({
                'Weights': weights,
                'Target_Weight': target_w,
                'Importance': importance,
                'Best_Config': config_ids[best_idx],
                'Max_Score': scores[best_idx]
            })
            
        return pd.DataFrame(results)


class MCDAVisualizer:
    """
    Generates academic, publication-quality visualizations for MCDA.
    """
    def __init__(self, raw_df: pd.DataFrame, norm_df: pd.DataFrame, criteria_cols: list, norm_cols: list):
        self.raw_df = raw_df
        self.norm_df = norm_df
        self.criteria_cols = criteria_cols
        self.norm_cols = norm_cols
        
        sns.set_theme(style='whitegrid', palette='colorblind', font='DejaVu Sans')
        self.palette = sns.color_palette('colorblind', 3)
        self.cluster_names_map = {
            0: 'Клъстер 0: Икономични / Високи загуби',
            1: 'Клъстер 1: Екологични / Среден капацитет',
            2: 'Клъстер 2: Високопроизводителни / Балансирани'
        }

    def generate_dashboard(self, save_path: str = '/workspace/scratch/pareto-cluster-dashboard-v2.png'):
        """Creates and saves the 4-panel multi-criteria analysis dashboard."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
        
        fig.suptitle('Аналитичен модел на пространството от решения в многокритериална система за оценка',
                     fontsize=18, fontweight='bold', y=0.96)
        
        # Prepare Plot Data
        plot_df = self.raw_df.copy()
        plot_df['Cluster_Name'] = plot_df['Cluster'].map(self.cluster_names_map)
        
        # Panel A: Capacity vs CAPEX
        ax1 = fig.add_subplot(gs[0, 0])
        sns.scatterplot(
            data=plot_df[plot_df['Is_Pareto'] == False],
            x='CAPEX_Mln_Euro', y='Storage_Capacity_MWh',
            hue='Cluster_Name', palette='colorblind', alpha=0.4, s=60, ax=ax1, legend=True
        )
        sns.scatterplot(
            data=plot_df[plot_df['Is_Pareto'] == True],
            x='CAPEX_Mln_Euro', y='Storage_Capacity_MWh',
            hue='Cluster_Name', palette='colorblind', alpha=1.0, s=120, marker='o', edgecolor='black', linewidth=2.0,
            ax=ax1, legend=False
        )
        # Plot Pareto boundary approximation line
        pareto_pts = plot_df[plot_df['Is_Pareto'] == True].sort_values('CAPEX_Mln_Euro')
        ax1.plot(pareto_pts['CAPEX_Mln_Euro'], pareto_pts['Storage_Capacity_MWh'], color='black', linestyle='--', alpha=0.6)
        ax1.set_title('А. Търговски компромис: Капацитет срещу Инвестиции (CAPEX)', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Инвестиционни разходи (CAPEX, Млн. Евро) [По-малко = По-добре]')
        ax1.set_ylabel('Капацитет на съхранение (MWh) [Повече = По-добре]')
        
        # Panel B: Transmission Losses vs Carbon Footprint
        ax2 = fig.add_subplot(gs[0, 1])
        sns.scatterplot(
            data=plot_df[plot_df['Is_Pareto'] == False],
            x='Transmission_Losses_Percent', y='Carbon_Footprint_gCO2_kWh',
            hue='Cluster_Name', palette='colorblind', alpha=0.4, s=60, ax=ax2, legend=False
        )
        sns.scatterplot(
            data=plot_df[plot_df['Is_Pareto'] == True],
            x='Transmission_Losses_Percent', y='Carbon_Footprint_gCO2_kWh',
            hue='Cluster_Name', palette='colorblind', alpha=1.0, s=120, marker='o', edgecolor='black', linewidth=2.0,
            ax=ax2, legend=False
        )
        ax2.set_title('Б. Екологичен/Мрежови компромис: Загуби срещу Въглероден отпечатък', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Загуби при пренос (%) [По-малко = По-добре]')
        ax2.set_ylabel('Въглероден отпечатък (gCO2/kWh) [По-малко = По-добре]')
        
        # Panel C: Normalized Cluster Profiles
        ax3 = fig.add_subplot(gs[1, 0])
        norm_plot_df = self.norm_df.copy()
        norm_plot_df['Cluster_Name'] = norm_plot_df['Cluster'].map(self.cluster_names_map)
        profile_data = norm_plot_df.groupby('Cluster_Name')[self.norm_cols].mean().reset_index()
        profile_melted = pd.melt(profile_data, id_vars='Cluster_Name', value_vars=self.norm_cols,
                                 var_name='Indicator', value_name='Normalized_Value')
        indicator_labels = {
            'n_Storage_Capacity_MWh': 'Капацитет',
            'n_Transmission_Losses_Percent': 'Ефективност на преноса',
            'n_CAPEX_Mln_Euro': 'Бюджетна ефективност',
            'n_Carbon_Footprint_gCO2_kWh': 'Климатична съвместимост'
        }
        profile_melted['Indicator'] = profile_melted['Indicator'].map(indicator_labels)
        sns.barplot(data=profile_melted, x='Indicator', y='Normalized_Value', hue='Cluster_Name', palette='colorblind', ax=ax3)
        ax3.set_title('В. Сравнителен профил на клъстерите (Нормализиран)', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Ключови аналитични индикатори')
        ax3.set_ylabel('Относителна стойност (1.0 е оптимум)')
        ax3.set_ylim(0, 1.1)
        ax3.legend().remove()
        
        # Panel D: Donut chart of Pareto configurations
        ax4 = fig.add_subplot(gs[1, 1])
        pareto_counts = plot_df[plot_df['Is_Pareto'] == True]['Cluster_Name'].value_counts()
        for name in self.cluster_names_map.values():
            if name not in pareto_counts:
                pareto_counts[name] = 0
        pareto_counts = pareto_counts.reindex(self.cluster_names_map.values())
        
        wedges, texts, autotexts = ax4.pie(
            pareto_counts, labels=[name.split(': ')[0] for name in pareto_counts.index],
            autopct='%1.0f%%', startangle=90, colors=self.palette,
            wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2.0), pctdistance=0.75
        )
        plt.setp(texts, fontsize=10, fontweight='bold')
        plt.setp(autotexts, fontsize=11, fontweight='bold')
        ax4.set_title(f'Г. Разпределение на Парето-оптималните решения (Общо: {plot_df["Is_Pareto"].sum()})',
                      fontsize=13, fontweight='bold')
        
        # Global despine & legends
        sns.despine(fig=fig)
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=11, frameon=True)
        ax1.legend().remove()
        
        fig.text(0.1, 0.01, 'Източник: Симулиран експериментален набор от 200 конфигурации (seed=42). Черният кант маркира Парето-оптималност.',
                 fontsize=9, color='gray', style='italic')
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.94])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"OOP Dashboard saved successfully to {save_path}!")


# --- PIPELINE DEMO & VERIFICATION ---
if __name__ == '__main__':
    # Configuration Setup
    criteria_config = [
        {'name': 'Storage_Capacity_MWh', 'dist': 'uniform', 'params': {'low': 10, 'high': 150}},
        {'name': 'Transmission_Losses_Percent', 'dist': 'normal', 'params': {'loc': 8.0, 'scale': 2.0, 'clip_low': 1.0, 'clip_high': 20.0}},
        {'name': 'CAPEX_Mln_Euro', 'dist': 'exponential', 'params': {'scale': 5.0, 'shift': 1.5}},
        {'name': 'Carbon_Footprint_gCO2_kWh', 'dist': 'uniform', 'params': {'low': 15, 'high': 400}}
    ]
    
    criteria_meta = {
        'Storage_Capacity_MWh': 'maximize',
        'Transmission_Losses_Percent': 'minimize',
        'CAPEX_Mln_Euro': 'minimize',
        'Carbon_Footprint_gCO2_kWh': 'minimize'
    }
    
    # 1. Generation
    generator = DataGenerator(num_configurations=200, seed=42)
    df_raw = generator.generate(criteria_config)
    
    # 2. Evaluation
    analyzer = MCDAAnalyzer(df_raw, criteria_meta)
    
    # Pareto & Clustering
    df_raw['Is_Pareto'] = analyzer.find_pareto_frontier()
    analyzer.raw_df['Is_Pareto'] = df_raw['Is_Pareto']
    analyzer.norm_df['Is_Pareto'] = df_raw['Is_Pareto']
    
    cluster_labels, kmeans_model = analyzer.run_clustering(n_clusters=3, seed=42)
    df_raw['Cluster'] = cluster_labels
    analyzer.raw_df['Cluster'] = cluster_labels
    analyzer.norm_df['Cluster'] = cluster_labels
    
    # 3. Global Simulation (using CAPEX index = 2 as target weight)
    simulator = GlobalSensitivitySimulator(analyzer, list(criteria_meta.keys()))
    df_sim = simulator.run_simulation(n_simulations=1000, target_col_idx=2)
    
    print("\n--- СТАБИЛНОСТ НА РЕШЕНИЯТА ПРИ ПРОМЯНА НА ТЕГЛОТО ЗА CAPEX ---")
    print(df_sim.groupby('Importance')['Best_Config'].value_counts().groupby(level=0).head(2))
    
    # 4. Visualization Dashboard
    visualizer = MCDAVisualizer(
        raw_df=df_raw,
        norm_df=analyzer.norm_df,
        criteria_cols=list(criteria_meta.keys()),
        norm_cols=analyzer.norm_cols
    )
    visualizer.generate_dashboard('/workspace/scratch/pareto-cluster-dashboard-v2.png')
