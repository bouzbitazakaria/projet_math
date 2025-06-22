"""
Module d'Analyse Exploratoire des Données (EDA) pour les données immobilières
Auteur: Projet Math IA
Date: Juin 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import List, Dict, Tuple, Optional
import os

# Configuration des visualisations
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Configuration pour les graphiques en français
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

class RealEstateEDA:
    """
    Classe pour l'analyse exploratoire des données immobilières
    """
    
    def __init__(self, df: pd.DataFrame, output_dir: str = 'visualizations'):
        """
        Initialise l'analyseur EDA
        
        Args:
            df (pd.DataFrame): DataFrame contenant les données immobilières
            output_dir (str): Répertoire de sortie pour les visualisations
        """
        self.df = df.copy()
        self.output_dir = output_dir
        self.ensure_output_dir()
        
        # Configuration des couleurs
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'info': '#7209B7'
        }
    
    def ensure_output_dir(self):
        """Crée le répertoire de sortie s'il n'existe pas"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def generate_summary_stats(self) -> Dict:
        """
        Génère des statistiques descriptives
        
        Returns:
            Dict: Dictionnaire contenant les statistiques
        """
        summary = {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_summary': self.df.describe().to_dict(),
            'categorical_summary': {}
        }
        
        # Statistiques pour les variables catégoriques
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_values': self.df[col].nunique(),
                'most_frequent': self.df[col].mode().iloc[0] if not self.df[col].empty else None,
                'value_counts': self.df[col].value_counts().head().to_dict()
            }
        
        return summary
    
    def plot_price_distribution(self, save: bool = True) -> None:
        """
        Visualise la distribution des prix
        
        Args:
            save (bool): Sauvegarder le graphique
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Distribution des Prix Immobiliers', fontsize=16, fontweight='bold')
        
        # Histogramme des prix
        axes[0, 0].hist(self.df['prix_dh'], bins=50, color=self.colors['primary'], alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Histogramme des Prix')
        axes[0, 0].set_xlabel('Prix (DH)')
        axes[0, 0].set_ylabel('Fréquence')
        axes[0, 0].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        # Box plot des prix
        axes[0, 1].boxplot(self.df['prix_dh'], patch_artist=True,
                          boxprops=dict(facecolor=self.colors['secondary'], alpha=0.7))
        axes[0, 1].set_title('Box Plot des Prix')
        axes[0, 1].set_ylabel('Prix (DH)')
        axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Histogramme log des prix
        prix_log = np.log10(self.df['prix_dh'])
        axes[1, 0].hist(prix_log, bins=50, color=self.colors['accent'], alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Distribution des Prix (échelle log)')
        axes[1, 0].set_xlabel('Log10(Prix)')
        axes[1, 0].set_ylabel('Fréquence')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(self.df['prix_dh'], dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Distribution normale)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/price_distribution.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(self, save: bool = True) -> None:
        """
        Visualise la matrice de corrélation
        
        Args:
            save (bool): Sauvegarder le graphique
        """
        # Sélection des variables numériques
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        
        # Création de la heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        plt.title('Matrice de Corrélation des Variables Numériques', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_price_by_location(self, save: bool = True) -> None:
        """
        Analyse des prix par localisation
        
        Args:
            save (bool): Sauvegarder le graphique
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        fig.suptitle('Analyse des Prix par Localisation', fontsize=16, fontweight='bold')
        
        # Box plot par ville
        sns.boxplot(data=self.df, x='localisation', y='prix_dh', ax=axes[0])
        axes[0].set_title('Distribution des Prix par Ville')
        axes[0].set_xlabel('Localisation')
        axes[0].set_ylabel('Prix (DH)')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Prix moyen par ville
        prix_moyen = self.df.groupby('localisation')['prix_dh'].mean().sort_values(ascending=False)
        bars = axes[1].bar(range(len(prix_moyen)), prix_moyen.values, 
                          color=self.colors['primary'], alpha=0.7)
        axes[1].set_title('Prix Moyen par Ville')
        axes[1].set_xlabel('Localisation')
        axes[1].set_ylabel('Prix Moyen (DH)')
        axes[1].set_xticks(range(len(prix_moyen)))
        axes[1].set_xticklabels(prix_moyen.index, rotation=45)
        axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Ajout des valeurs sur les barres
        for bar, value in zip(bars, prix_moyen.values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                        f'{value:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/price_by_location.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_price_vs_surface(self, save: bool = True) -> None:
        """
        Analyse de la relation prix-surface
        
        Args:
            save (bool): Sauvegarder le graphique
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Relation Prix - Surface', fontsize=16, fontweight='bold')
        
        # Scatter plot prix vs surface
        axes[0, 0].scatter(self.df['surface_m2'], self.df['prix_dh'], 
                          alpha=0.6, color=self.colors['primary'])
        axes[0, 0].set_title('Prix vs Surface')
        axes[0, 0].set_xlabel('Surface (m²)')
        axes[0, 0].set_ylabel('Prix (DH)')
        axes[0, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Régression linéaire
        z = np.polyfit(self.df['surface_m2'], self.df['prix_dh'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(self.df['surface_m2'], p(self.df['surface_m2']), 
                       "r--", alpha=0.8, linewidth=2)
        
        # Corrélation
        correlation = self.df['surface_m2'].corr(self.df['prix_dh'])
        axes[0, 0].text(0.05, 0.95, f'Corrélation: {correlation:.3f}', 
                       transform=axes[0, 0].transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Prix au m² par type de bien
        if 'type_bien' in self.df.columns:
            sns.boxplot(data=self.df, x='type_bien', y='prix_par_m2', ax=axes[0, 1])
            axes[0, 1].set_title('Prix au m² par Type de Bien')
            axes[0, 1].set_xlabel('Type de Bien')
            axes[0, 1].set_ylabel('Prix au m² (DH)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Distribution de la surface
        axes[1, 0].hist(self.df['surface_m2'], bins=30, color=self.colors['accent'], 
                       alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Distribution des Surfaces')
        axes[1, 0].set_xlabel('Surface (m²)')
        axes[1, 0].set_ylabel('Fréquence')
        
        # Heatmap prix vs surface vs chambres
        if 'nombre_chambres' in self.df.columns:
            pivot_data = self.df.groupby(['nombre_chambres', pd.cut(self.df['surface_m2'], bins=5)])['prix_dh'].mean().unstack()
            sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[1, 1])
            axes[1, 1].set_title('Prix Moyen (Chambres vs Surface)')
            axes[1, 1].set_xlabel('Classes de Surface')
            axes[1, 1].set_ylabel('Nombre de Chambres')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/price_vs_surface.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_property_types_analysis(self, save: bool = True) -> None:
        """
        Analyse par type de bien
        
        Args:
            save (bool): Sauvegarder le graphique
        """
        if 'type_bien' not in self.df.columns:
            print("Colonne 'type_bien' non trouvée")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analyse par Type de Bien', fontsize=16, fontweight='bold')
        
        # Distribution des types de biens
        type_counts = self.df['type_bien'].value_counts()
        wedges, texts, autotexts = axes[0, 0].pie(type_counts.values, labels=type_counts.index, 
                                                 autopct='%1.1f%%', startangle=90,
                                                 colors=sns.color_palette("husl", len(type_counts)))
        axes[0, 0].set_title('Répartition par Type de Bien')
        
        # Prix moyen par type
        prix_par_type = self.df.groupby('type_bien')['prix_dh'].mean().sort_values(ascending=True)
        bars = axes[0, 1].barh(range(len(prix_par_type)), prix_par_type.values,
                              color=self.colors['secondary'], alpha=0.7)
        axes[0, 1].set_title('Prix Moyen par Type de Bien')
        axes[0, 1].set_xlabel('Prix Moyen (DH)')
        axes[0, 1].set_yticks(range(len(prix_par_type)))
        axes[0, 1].set_yticklabels(prix_par_type.index)
        axes[0, 1].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        
        # Surface moyenne par type
        surface_par_type = self.df.groupby('type_bien')['surface_m2'].mean().sort_values(ascending=True)
        axes[1, 0].bar(range(len(surface_par_type)), surface_par_type.values,
                      color=self.colors['accent'], alpha=0.7)
        axes[1, 0].set_title('Surface Moyenne par Type de Bien')
        axes[1, 0].set_ylabel('Surface Moyenne (m²)')
        axes[1, 0].set_xticks(range(len(surface_par_type)))
        axes[1, 0].set_xticklabels(surface_par_type.index, rotation=45)
        
        # Box plot prix par type
        sns.boxplot(data=self.df, x='type_bien', y='prix_dh', ax=axes[1, 1])
        axes[1, 1].set_title('Distribution des Prix par Type')
        axes[1, 1].set_xlabel('Type de Bien')
        axes[1, 1].set_ylabel('Prix (DH)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/property_types_analysis.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_rooms_analysis(self, save: bool = True) -> None:
        """
        Analyse par nombre de chambres
        
        Args:
            save (bool): Sauvegarder le graphique
        """
        if 'nombre_chambres' not in self.df.columns:
            print("Colonne 'nombre_chambres' non trouvée")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analyse par Nombre de Chambres', fontsize=16, fontweight='bold')
        
        # Distribution du nombre de chambres
        room_counts = self.df['nombre_chambres'].value_counts().sort_index()
        axes[0, 0].bar(room_counts.index, room_counts.values, 
                      color=self.colors['primary'], alpha=0.7)
        axes[0, 0].set_title('Distribution du Nombre de Chambres')
        axes[0, 0].set_xlabel('Nombre de Chambres')
        axes[0, 0].set_ylabel('Nombre de Propriétés')
        
        # Prix moyen par nombre de chambres
        prix_par_chambres = self.df.groupby('nombre_chambres')['prix_dh'].mean()
        axes[0, 1].plot(prix_par_chambres.index, prix_par_chambres.values, 
                       marker='o', linewidth=2, markersize=8, color=self.colors['secondary'])
        axes[0, 1].set_title('Prix Moyen par Nombre de Chambres')
        axes[0, 1].set_xlabel('Nombre de Chambres')
        axes[0, 1].set_ylabel('Prix Moyen (DH)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Surface moyenne par nombre de chambres
        surface_par_chambres = self.df.groupby('nombre_chambres')['surface_m2'].mean()
        axes[1, 0].plot(surface_par_chambres.index, surface_par_chambres.values,
                       marker='s', linewidth=2, markersize=8, color=self.colors['accent'])
        axes[1, 0].set_title('Surface Moyenne par Nombre de Chambres')
        axes[1, 0].set_xlabel('Nombre de Chambres')
        axes[1, 0].set_ylabel('Surface Moyenne (m²)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot prix par nombre de chambres
        sns.boxplot(data=self.df, x='nombre_chambres', y='prix_dh', ax=axes[1, 1])
        axes[1, 1].set_title('Distribution des Prix par Nombre de Chambres')
        axes[1, 1].set_xlabel('Nombre de Chambres')
        axes[1, 1].set_ylabel('Prix (DH)')
        axes[1, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'{self.output_dir}/rooms_analysis.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self) -> None:
        """
        Crée un tableau de bord interactif avec Plotly
        """
        # Création du dashboard avec subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Distribution des Prix', 'Prix par Localisation',
                          'Prix vs Surface', 'Distribution par Type',
                          'Corrélations', 'Évolution Temporelle'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Histogramme des prix
        fig.add_trace(
            go.Histogram(x=self.df['prix_dh'], name='Prix', nbinsx=30,
                        marker_color=self.colors['primary']),
            row=1, col=1
        )
        
        # 2. Box plot prix par localisation
        for i, location in enumerate(self.df['localisation'].unique()):
            location_data = self.df[self.df['localisation'] == location]['prix_dh']
            fig.add_trace(
                go.Box(y=location_data, name=location, boxpoints='outliers'),
                row=1, col=2
            )
        
        # 3. Scatter plot prix vs surface
        fig.add_trace(
            go.Scatter(x=self.df['surface_m2'], y=self.df['prix_dh'],
                      mode='markers', name='Propriétés',
                      marker=dict(color=self.colors['accent'], opacity=0.6)),
            row=2, col=1
        )
        
        # 4. Pie chart par type de bien
        if 'type_bien' in self.df.columns:
            type_counts = self.df['type_bien'].value_counts()
            fig.add_trace(
                go.Pie(labels=type_counts.index, values=type_counts.values,
                      name="Types de Biens"),
                row=2, col=2
            )
        
        # Mise à jour de la disposition
        fig.update_layout(
            height=1200,
            title_text="Tableau de Bord - Analyse Immobilière",
            title_x=0.5,
            showlegend=False
        )
        
        # Sauvegarde en HTML
        fig.write_html(f'{self.output_dir}/interactive_dashboard.html')
        print(f"Tableau de bord interactif sauvegardé dans {self.output_dir}/interactive_dashboard.html")
    
    def generate_eda_report(self) -> Dict:
        """
        Génère un rapport complet d'EDA
        
        Returns:
            Dict: Rapport d'analyse
        """
        report = {
            'dataset_info': {
                'nombre_proprietes': len(self.df),
                'nombre_variables': len(self.df.columns),
                'variables': list(self.df.columns),
                'periode_analyse': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'statistiques_prix': {
                'prix_moyen': self.df['prix_dh'].mean(),
                'prix_median': self.df['prix_dh'].median(),
                'prix_min': self.df['prix_dh'].min(),
                'prix_max': self.df['prix_dh'].max(),
                'ecart_type': self.df['prix_dh'].std()
            },
            'correlations_principales': {},
            'insights_cles': []
        }
        
        # Corrélations avec le prix
        if len(self.df.select_dtypes(include=[np.number]).columns) > 1:
            correlations = self.df.corr()['prix_dh'].abs().sort_values(ascending=False)
            report['correlations_principales'] = correlations.to_dict()
        
        # Insights clés
        insights = []
        
        # Prix par ville
        if 'localisation' in self.df.columns:
            ville_plus_chere = self.df.groupby('localisation')['prix_dh'].mean().idxmax()
            prix_plus_cher = self.df.groupby('localisation')['prix_dh'].mean().max()
            insights.append(f"La ville la plus chère est {ville_plus_chere} avec un prix moyen de {prix_plus_cher:,.0f} DH")
        
        # Surface moyenne
        if 'surface_m2' in self.df.columns:
            surface_moyenne = self.df['surface_m2'].mean()
            insights.append(f"La surface moyenne des propriétés est de {surface_moyenne:.1f} m²")
        
        # Type le plus fréquent
        if 'type_bien' in self.df.columns:
            type_frequent = self.df['type_bien'].mode().iloc[0]
            pourcentage = (self.df['type_bien'] == type_frequent).mean() * 100
            insights.append(f"Le type de bien le plus fréquent est '{type_frequent}' ({pourcentage:.1f}%)")
        
        report['insights_cles'] = insights
        
        return report
    
    def run_complete_eda(self) -> None:
        """
        Exécute l'analyse EDA complète
        """
        print("🏠 Début de l'Analyse Exploratoire des Données Immobilières")
        print("=" * 60)
        
        # Statistiques descriptives
        print("📊 Génération des statistiques descriptives...")
        summary = self.generate_summary_stats()
        print(f"Dataset: {summary['shape'][0]} lignes, {summary['shape'][1]} colonnes")
        
        # Visualisations
        print("\n📈 Génération des visualisations...")
        
        print("   - Distribution des prix...")
        self.plot_price_distribution()
        
        print("   - Matrice de corrélation...")
        self.plot_correlation_matrix()
        
        print("   - Analyse par localisation...")
        self.plot_price_by_location()
        
        print("   - Relation prix-surface...")
        self.plot_price_vs_surface()
        
        print("   - Analyse par type de bien...")
        self.plot_property_types_analysis()
        
        print("   - Analyse par nombre de chambres...")
        self.plot_rooms_analysis()
        
        print("   - Tableau de bord interactif...")
        self.create_interactive_dashboard()
        
        # Rapport final
        print("\n📋 Génération du rapport...")
        report = self.generate_eda_report()
        
        print("\n🎯 Insights Clés:")
        for insight in report['insights_cles']:
            print(f"   • {insight}")
        
        print(f"\n✅ Analyse terminée! Visualisations sauvegardées dans '{self.output_dir}/'")
        
        return report

if __name__ == "__main__":
    # Exemple d'utilisation
    try:
        # Chargement des données
        df = pd.read_csv('data/cleaned_properties.csv')
        print(f"Données chargées: {len(df)} propriétés")
        
        # Initialisation de l'analyseur EDA
        eda = RealEstateEDA(df)
        
        # Exécution de l'analyse complète
        report = eda.run_complete_eda()
        
    except FileNotFoundError:
        print("❌ Fichier de données non trouvé.")
        print("Exécutez d'abord:")
        print("1. python src/data_scraper.py")
        print("2. python src/data_pipeline.py")
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse EDA: {e}")
