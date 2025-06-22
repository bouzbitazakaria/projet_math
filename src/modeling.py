"""
Module de modélisation pour la prédiction des prix immobiliers
Auteur: Projet Math IA
Date: Juin 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RealEstatePricePredictor:
    """
    Classe pour la prédiction des prix immobiliers
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialise le prédicteur
        
        Args:
            random_state (int): Graine pour la reproductibilité
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.scaler = None
        self.results = {}
        
        # Initialisation des modèles
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialise les différents modèles à tester"""
        self.models = {
            'Linear_Regression': LinearRegression(),
            'Ridge_Regression': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso_Regression': Lasso(alpha=1.0, random_state=self.random_state),
            'Random_Forest': RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient_Boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.random_state
            )
        }
    
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'prix_dh', 
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prépare les données pour l'entraînement
        
        Args:
            df (pd.DataFrame): DataFrame avec les données
            target_column (str): Nom de la colonne cible
            test_size (float): Proportion des données de test
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        # Importation du pipeline de données
        from data_pipeline import DataCleaner
        
        # Nettoyage et préparation des données
        cleaner = DataCleaner()
        X, y = cleaner.prepare_for_modeling(df, target_column)
        
        # Sauvegarde des noms de features
        self.feature_names = cleaner.get_feature_names()
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Données préparées:")
        print(f"  - Features: {X.shape[1]}")
        print(f"  - Entraînement: {X_train.shape[0]} échantillons")
        print(f"  - Test: {X_test.shape[0]} échantillons")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Entraîne tous les modèles et évalue leurs performances
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement  
            X_test: Features de test
            y_test: Target de test
            
        Returns:
            Dict: Résultats des modèles
        """
        results = {}
        
        print("🚀 Entraînement des modèles...")
        print("=" * 50)
        
        for name, model in self.models.items():
            print(f"📈 Entraînement: {name}")
            
            # Entraînement
            model.fit(X_train, y_train)
            
            # Prédictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Métriques
            train_metrics = self._calculate_metrics(y_train, y_train_pred)
            test_metrics = self._calculate_metrics(y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='neg_mean_squared_error', n_jobs=-1)
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results[name] = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_rmse': cv_rmse,
                'y_pred': y_test_pred
            }
            
            print(f"   R²: {test_metrics['r2']:.4f}")
            print(f"   RMSE: {test_metrics['rmse']:,.0f}")
            print(f"   CV RMSE: {cv_rmse:,.0f}")
            print()
        
        self.results = results
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calcule les métriques de performance
        
        Args:
            y_true: Valeurs réelles
            y_pred: Valeurs prédites
            
        Returns:
            Dict: Métriques calculées
        """
        return {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Optimise les hyperparamètres des meilleurs modèles
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            
        Returns:
            Dict: Modèles optimisés
        """
        print("🔧 Optimisation des hyperparamètres...")
        
        # Grilles de paramètres
        param_grids = {
            'Ridge_Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Lasso_Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Random_Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient_Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        optimized_models = {}
        
        for name, model in self.models.items():
            if name in param_grids:
                print(f"   Optimisation: {name}")
                
                grid_search = GridSearchCV(
                    model, param_grids[name],
                    cv=5, scoring='neg_mean_squared_error',
                    n_jobs=-1, verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                optimized_models[name] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_score': -grid_search.best_score_
                }
                
                print(f"      Meilleurs paramètres: {grid_search.best_params_}")
                print(f"      Score CV: {np.sqrt(-grid_search.best_score_):,.0f}")
        
        return optimized_models
    
    def select_best_model(self) -> str:
        """
        Sélectionne le meilleur modèle basé sur les métriques
        
        Returns:
            str: Nom du meilleur modèle
        """
        if not self.results:
            raise ValueError("Aucun modèle entraîné. Exécutez train_models() d'abord.")
        
        # Sélection basée sur le R² de test
        best_r2 = -np.inf
        best_name = None
        
        for name, result in self.results.items():
            r2 = result['test_metrics']['r2']
            if r2 > best_r2:
                best_r2 = r2
                best_name = name
        
        self.best_model = self.results[best_name]['model']
        self.best_model_name = best_name
        
        print(f"🏆 Meilleur modèle: {best_name}")
        print(f"   R²: {best_r2:.4f}")
        
        return best_name
    
    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        Analyse l'importance des features
        
        Returns:
            pd.DataFrame: Importance des features
        """
        if self.best_model is None:
            raise ValueError("Aucun meilleur modèle sélectionné.")
        
        importances = None
        
        # Récupération de l'importance selon le type de modèle
        if hasattr(self.best_model, 'feature_importances_'):
            # Random Forest, Gradient Boosting
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            # Régression linéaire, Ridge, Lasso
            importances = np.abs(self.best_model.coef_)
        
        if importances is not None and self.feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': self.feature_names[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return pd.DataFrame()
    
    def plot_model_comparison(self, save: bool = True) -> None:
        """
        Visualise la comparaison des modèles
        
        Args:
            save (bool): Sauvegarder le graphique
        """
        if not self.results:
            print("Aucun résultat à afficher")
            return
        
        # Préparation des données pour la visualisation
        model_names = list(self.results.keys())
        r2_scores = [self.results[name]['test_metrics']['r2'] for name in model_names]
        rmse_scores = [self.results[name]['test_metrics']['rmse'] for name in model_names]
        mae_scores = [self.results[name]['test_metrics']['mae'] for name in model_names]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparaison des Modèles de Prédiction', fontsize=16, fontweight='bold')
        
        # R² Score
        bars1 = axes[0, 0].bar(model_names, r2_scores, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Score R² (plus élevé = meilleur)')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_ylim(0, 1)
        
        # Ajout des valeurs sur les barres
        for bar, score in zip(bars1, r2_scores):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        # RMSE
        bars2 = axes[0, 1].bar(model_names, rmse_scores, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('RMSE (plus bas = meilleur)')
        axes[0, 1].set_ylabel('RMSE (DH)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # MAE
        bars3 = axes[1, 0].bar(model_names, mae_scores, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('MAE (plus bas = meilleur)')
        axes[1, 0].set_ylabel('MAE (DH)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # Graphique radar des performances
        angles = np.linspace(0, 2 * np.pi, len(model_names), endpoint=False).tolist()
        angles += angles[:1]  # Fermer le cercle
        
        axes[1, 1].remove()  # Supprimer le subplot standard
        ax_radar = fig.add_subplot(2, 2, 4, projection='polar')
        
        # Normalisation des scores pour le radar
        r2_norm = [(score - min(r2_scores)) / (max(r2_scores) - min(r2_scores)) for score in r2_scores]
        r2_norm += r2_norm[:1]
        
        ax_radar.plot(angles, r2_norm, 'o-', linewidth=2, label='R² Normalisé')
        ax_radar.fill(angles, r2_norm, alpha=0.25)
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(model_names)
        ax_radar.set_title('Performance Relative des Modèles')
        ax_radar.grid(True)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_predictions_vs_actual(self, X_test: np.ndarray, y_test: np.ndarray, save: bool = True) -> None:
        """
        Visualise les prédictions vs valeurs réelles
        
        Args:
            X_test: Features de test
            y_test: Valeurs réelles de test
            save (bool): Sauvegarder le graphique
        """
        if self.best_model is None:
            print("Aucun modèle sélectionné")
            return
        
        y_pred = self.best_model.predict(X_test)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Analyse des Prédictions - {self.best_model_name}', fontsize=16, fontweight='bold')
        
        # Scatter plot prédictions vs réelles
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue')
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[0, 0].set_xlabel('Prix Réel (DH)')
        axes[0, 0].set_ylabel('Prix Prédit (DH)')
        axes[0, 0].set_title('Prédictions vs Valeurs Réelles')
        axes[0, 0].ticklabel_format(style='scientific', scilimits=(0,0))
        
        # Calcul du R²
        r2 = r2_score(y_test, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Résidus
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Prix Prédit (DH)')
        axes[0, 1].set_ylabel('Résidus (DH)')
        axes[0, 1].set_title('Graphique des Résidus')
        axes[0, 1].ticklabel_format(style='scientific', scilimits=(0,0))
        
        # Distribution des résidus
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_xlabel('Résidus (DH)')
        axes[1, 0].set_ylabel('Fréquence')
        axes[1, 0].set_title('Distribution des Résidus')
        axes[1, 0].axvline(x=0, color='red', linestyle='--')
        
        # Q-Q plot des résidus
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot des Résidus')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('visualizations/predictions_analysis.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, save: bool = True) -> None:
        """
        Visualise l'importance des features
        
        Args:
            save (bool): Sauvegarder le graphique
        """
        importance_df = self.analyze_feature_importance()
        
        if importance_df.empty:
            print("Importance des features non disponible pour ce modèle")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Top 15 features les plus importantes
        top_features = importance_df.head(15)
        
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       color='steelblue', alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Importance des Features - {self.best_model_name}', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Ajout des valeurs
        for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
            plt.text(bar.get_width() + value * 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', va='center', ha='left')
        
        plt.tight_layout()
        
        if save:
            plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filename: str = 'models/best_price_predictor.pkl') -> None:
        """
        Sauvegarde le meilleur modèle
        
        Args:
            filename (str): Nom du fichier de sauvegarde
        """
        if self.best_model is None:
            raise ValueError("Aucun modèle à sauvegarder")
        
        # Création du répertoire si nécessaire
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Sauvegarde du modèle et des métadonnées
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'metrics': self.results[self.best_model_name]['test_metrics']
        }
        
        joblib.dump(model_data, filename)
        print(f"Modèle sauvegardé dans {filename}")
    
    def load_model(self, filename: str) -> None:
        """
        Charge un modèle sauvegardé
        
        Args:
            filename (str): Nom du fichier à charger
        """
        model_data = joblib.load(filename)
        
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.feature_names = model_data['feature_names']
        
        print(f"Modèle {self.best_model_name} chargé depuis {filename}")
    
    def predict_price(self, property_data: Dict) -> float:
        """
        Prédit le prix d'une propriété
        
        Args:
            property_data (Dict): Caractéristiques de la propriété
            
        Returns:
            float: Prix prédit en DH
        """
        if self.best_model is None:
            raise ValueError("Aucun modèle entraîné")
        
        # Conversion en DataFrame pour le préprocessing
        df_input = pd.DataFrame([property_data])
        
        # Préprocessing (utilisation du même pipeline)
        from data_pipeline import DataCleaner
        cleaner = DataCleaner()
        X_processed, _ = cleaner.prepare_for_modeling(df_input, target_column=None)
        
        # Prédiction
        predicted_price = self.best_model.predict(X_processed)[0]
        
        return predicted_price
    
    def generate_model_report(self) -> Dict:
        """
        Génère un rapport complet de modélisation
        
        Returns:
            Dict: Rapport de modélisation
        """
        if not self.results:
            return {"error": "Aucun modèle entraîné"}
        
        # Métriques de tous les modèles
        model_metrics = {}
        for name, result in self.results.items():
            model_metrics[name] = result['test_metrics']
        
        # Importance des features
        feature_importance = self.analyze_feature_importance()
        
        report = {
            'meilleur_modele': self.best_model_name,
            'metriques_tous_modeles': model_metrics,
            'metriques_meilleur_modele': self.results[self.best_model_name]['test_metrics'] if self.best_model_name else None,
            'importance_features': feature_importance.to_dict('records') if not feature_importance.empty else [],
            'recommandations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """
        Génère des recommandations d'amélioration
        
        Returns:
            List[str]: Liste de recommandations
        """
        recommendations = []
        
        if not self.results or not self.best_model_name:
            return recommendations
        
        best_r2 = self.results[self.best_model_name]['test_metrics']['r2']
        
        if best_r2 < 0.7:
            recommendations.append("Le R² est faible (<0.7). Considérez collecter plus de données ou ajouter des features.")
        
        if best_r2 < 0.5:
            recommendations.append("Performance très faible. Vérifiez la qualité des données et les features utilisées.")
        
        # Vérification de l'overfitting
        train_r2 = self.results[self.best_model_name]['train_metrics']['r2']
        if train_r2 - best_r2 > 0.1:
            recommendations.append("Possible overfitting détecté. Considérez la régularisation ou plus de données.")
        
        recommendations.append("Testez des transformations des variables (log, polynomial).")
        recommendations.append("Explorez d'autres algorithmes (XGBoost, Neural Networks).")
        
        return recommendations

if __name__ == "__main__":
    # Exemple d'utilisation
    try:
        # Chargement des données
        df = pd.read_csv('data/cleaned_properties.csv')
        print(f"Données chargées: {len(df)} propriétés")
        
        # Initialisation du prédicteur
        predictor = RealEstatePricePredictor()
        
        # Préparation des données
        X_train, X_test, y_train, y_test = predictor.prepare_data(df)
        
        # Entraînement des modèles
        results = predictor.train_models(X_train, y_train, X_test, y_test)
        
        # Sélection du meilleur modèle
        best_model = predictor.select_best_model()
        
        # Visualisations
        predictor.plot_model_comparison()
        predictor.plot_predictions_vs_actual(X_test, y_test)
        predictor.plot_feature_importance()
        
        # Sauvegarde du modèle
        predictor.save_model()
        
        # Génération du rapport
        report = predictor.generate_model_report()
        print("\n📊 Rapport de modélisation généré")
        
        # Exemple de prédiction
        example_property = {
            'surface_m2': 100,
            'nombre_chambres': 3,
            'localisation': 'Casablanca',
            'type_bien': 'Appartement',
            'annee_construction': 2015
        }
        
        # predicted_price = predictor.predict_price(example_property)
        # print(f"\n🏠 Prédiction pour l'exemple: {predicted_price:,.0f} DH")
        
    except FileNotFoundError:
        print("❌ Fichier de données non trouvé.")
        print("Exécutez d'abord les scripts de collecte et nettoyage des données.")
    except Exception as e:
        print(f"❌ Erreur lors de la modélisation: {e}")
