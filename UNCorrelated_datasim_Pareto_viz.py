import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os

# 1. Възпроизвеждане на симулираните данни (задаване на seed=42)
np.random.seed(42)
num_configurations = 200
storage_capacity = np.random.uniform(low=10, high=150, size=num_configurations)
transmission_losses = np.clip(np.random.normal(loc=8.0, scale=2.0, size=num_configurations), 1.0, 20.0)
capex = np.random.exponential(scale=5.0, size=num_configurations) + 1.5
carbon_footprint = np.random.uniform(low=15, high=400, size=num_configurations)

df_energy = pd.DataFrame({
    'Config_ID': [f'Config_{i}' for i in range(1, num_configurations + 1)],
    'Storage_Capacity_MWh': storage_capacity,
    'Transmission_Losses_Percent': transmission_losses,
    'CAPEX_Mln_Euro': capex,
    'Carbon_Footprint_gCO2_kWh': carbon_footprint
})

# 2. Нормализиране на индикаторите в интервала [0, 1] (съгласно Задача 1 и 2)
df_norm = pd.DataFrame()
df_norm['Config_ID'] = df_energy['Config_ID']
df_norm['n_Capacity'] = (df_energy['Storage_Capacity_MWh'] - df_energy['Storage_Capacity_MWh'].min()) /                         (df_energy['Storage_Capacity_MWh'].max() - df_energy['Storage_Capacity_MWh'].min())

df_norm['n_Losses'] = (df_energy['Transmission_Losses_Percent'].max() - df_energy['Transmission_Losses_Percent']) /                       (df_energy['Transmission_Losses_Percent'].max() - df_energy['Transmission_Losses_Percent'].min())

df_norm['n_CAPEX'] = (df_energy['CAPEX_Mln_Euro'].max() - df_energy['CAPEX_Mln_Euro']) /                      (df_energy['CAPEX_Mln_Euro'].max() - df_energy['CAPEX_Mln_Euro'].min())

df_norm['n_Carbon'] = (df_energy['Carbon_Footprint_gCO2_kWh'].max() - df_energy['Carbon_Footprint_gCO2_kWh']) /                       (df_energy['Carbon_Footprint_gCO2_kWh'].max() - df_energy['Carbon_Footprint_gCO2_kWh'].min())

criteria_cols = ['n_Capacity', 'n_Losses', 'n_CAPEX', 'n_Carbon']
data_matrix = df_norm[criteria_cols].values

# 3. Алгоритъм за намиране на Парето-оптималните решения (non-dominated)
def get_pareto_frontier(costs):
    # ФИКС: Използва се само първото измерение на формата (брой редове), за да се получи 1D масив
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Тъй като критериите са нормализирани така, че 1.0 е НАЙ-ДОБРОТО (по-голямото е по-добро),
            # решение доминира друго, ако има по-големи или равни стойности
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1) |                                          np.all(costs[is_efficient] == c, axis=1)
            is_efficient[i] = True
    return is_efficient

df_energy['Is_Pareto'] = get_pareto_frontier(data_matrix)
df_norm['Is_Pareto'] = df_energy['Is_Pareto']

# 4. K-Means клъстеризация (Разделяне на 3 класа съгласно Задача 5)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_energy['Cluster'] = kmeans.fit_predict(data_matrix)
df_norm['Cluster'] = df_energy['Cluster']

# Наименования на клъстерите спрямо анализирания профил
cluster_names = {
    0: 'Клъстер 0: Ниска производителност / Висок отпечатък',
    1: 'Клъстер 1: Екологични / Среден капацитет',
    2: 'Клъстер 2: Високопроизводителни / Ниски разходи'
}
df_energy['Cluster_Name'] = df_energy['Cluster'].map(cluster_names)
df_norm['Cluster_Name'] = df_norm['Cluster'].map(cluster_names)

# Настройка на стила на визуализацията
sns.set_theme(style='whitegrid', palette='colorblind', font='DejaVu Sans')
palette = sns.color_palette('colorblind', 3)
cluster_colors = {
    'Клъстер 0: Ниска производителност / Висок отпечатък': palette[0],
    'Клъстер 1: Екологични / Среден капацитет': palette[2],  # Използване на валиден индекс в палитрата
    'Клъстер 2: Високопроизводителни / Ниски разходи': palette[1]
}

# Инициализиране на 2х2 пространство за фигурата
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
fig.suptitle('Аналитичен модел на пространството от решения: Клъстери 1 и 2 доминират 91% от Парето фронта',
             fontsize=18, fontweight='bold', y=0.96)

# --- ПАНЕЛ А: Капацитет срещу CAPEX (Търговски компромис) ---
ax1 = fig.add_subplot(gs[0, 0])  # ФИКС: Индексиране на решетката (Горе вляво)
sns.scatterplot(
    data=df_energy[df_energy['Is_Pareto'] == False],
    x='CAPEX_Mln_Euro', y='Storage_Capacity_MWh',
    hue='Cluster_Name', palette=cluster_colors, alpha=0.4, s=60, ax=ax1, legend=True
)
sns.scatterplot(
    data=df_energy[df_energy['Is_Pareto'] == True],
    x='CAPEX_Mln_Euro', y='Storage_Capacity_MWh',
    hue='Cluster_Name', palette=cluster_colors, alpha=1.0, s=120, marker='o', 
    edgecolor='black', linewidth=2.0, ax=ax1, legend=False
)
ax1.set_title('А. Търговски компромис: Капацитет срещу CAPEX', fontsize=13, fontweight='bold')
ax1.set_xlabel('Инвестиционни разходи (CAPEX, Млн. Евро) -> По-ниското е по-добро')
ax1.set_ylabel('Капацитет на съхранение (MWh) -> По-високото е по-добро')

pareto_points = df_energy[df_energy['Is_Pareto'] == True].sort_values('CAPEX_Mln_Euro')
ax1.plot(pareto_points['CAPEX_Mln_Euro'], pareto_points['Storage_Capacity_MWh'], 
         color='black', linestyle='--', alpha=0.5)

# --- ПАНЕЛ Б: Загуби срещу Емисии (Екологичен компромис) ---
ax2 = fig.add_subplot(gs[0, 1])  # ФИКС: Индексиране на решетката (Горе вдясно)
sns.scatterplot(
    data=df_energy[df_energy['Is_Pareto'] == False],
    x='Transmission_Losses_Percent', y='Carbon_Footprint_gCO2_kWh',
    hue='Cluster_Name', palette=cluster_colors, alpha=0.4, s=60, ax=ax2, legend=False
)
sns.scatterplot(
    data=df_energy[df_energy['Is_Pareto'] == True],
    x='Transmission_Losses_Percent', y='Carbon_Footprint_gCO2_kWh',
    hue='Cluster_Name', palette=cluster_colors, alpha=1.0, s=120, marker='o', 
    edgecolor='black', linewidth=2.0, ax=ax2, legend=False
)
ax2.set_title('Б. Екологичен/Мрежови компромис: Загуби срещу Емисии', fontsize=13, fontweight='bold')
ax2.set_xlabel('Загуби при пренос (%) -> По-ниското е по-добро')
ax2.set_ylabel('Въглероден отпечатък (gCO2/kWh) -> По-ниското е по-добро')

# --- ПАНЕЛ В: Сравнителен профил на клъстерите (Нормализиран) ---
ax3 = fig.add_subplot(gs[1, 0])  # ФИКС: Индексиране на решетката (Долу вляво)
profile_data = df_norm.groupby('Cluster_Name')[criteria_cols].mean().reset_index()
profile_melted = pd.melt(profile_data, id_vars='Cluster_Name', value_vars=criteria_cols,
                         var_name='Indicator', value_name='Normalized_Value')
indicator_map = {
    'n_Capacity': 'Капацитет',
    'n_Losses': 'Спестени Загуби',
    'n_CAPEX': 'Спестен CAPEX',
    'n_Carbon': 'Спестени Емисии'
}
profile_melted['Indicator'] = profile_melted['Indicator'].map(indicator_map)
sns.barplot(data=profile_melted, x='Indicator', y='Normalized_Value', hue='Cluster_Name', 
            palette=cluster_colors, ax=ax3)
ax3.set_title('В. Сравнителен профил на клъстерите (Нормализиран)', fontsize=13, fontweight='bold')
ax3.set_xlabel('Нормализирани критерии (1.0 е оптимално представяне)')
ax3.set_ylabel('Средна стойност в клъстера')
ax3.set_ylim(0, 1.05)
ax3.legend().remove()

# --- ПАНЕЛ Г: Разпределение на Парето-оптималните решения ---
ax4 = fig.add_subplot(gs[1, 1])  # ФИКС: Индексиране на решетката (Долу вдясно)
pareto_counts = df_energy[df_energy['Is_Pareto'] == True]['Cluster_Name'].value_counts()
for name in cluster_colors.keys():
    if name not in pareto_counts:
        pareto_counts[name] = 0
pareto_counts = pareto_counts.reindex(cluster_colors.keys())

# Изчертаване на Donut Chart
wedges, texts, autotexts = ax4.pie(
    pareto_counts, labels=['\n'.join(name.split(': ')) for name in pareto_counts.index],
    autopct='%1.0f%%', startangle=90, colors=[cluster_colors[name] for name in pareto_counts.index],
    wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2.0), pctdistance=0.75
)
plt.setp(texts, fontsize=10, fontweight='bold')
plt.setp(autotexts, fontsize=11, fontweight='bold')
ax4.set_title(f'Г. Разпределение на Парето-оптималните решения (Общо: {df_energy["Is_Pareto"].sum()})', 
             fontsize=13, fontweight='bold')

# Финални фини настройки
sns.despine(fig=fig)
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles[:3], labels[:3], loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, 
           fontsize=11, frameon=True)
ax1.legend().remove()

fig.text(0.1, 0.01, 'Източник: Симулирани некорелирани данни. Маркираните с черен кант точки са Парето-оптимални.',
         fontsize=9, color='gray', style='italic')
plt.tight_layout(rect=[0, 0.05, 1, 0.94])
plt.savefig('generated/pareto_cluster_dashboard.png', dpi=150, bbox_inches='tight')
