"""
Pipeline de nettoyage et préparation des données immobilières
Auteur: Projet Math IA
Date: Juin 2025
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from typing import List, Tuple, Optional
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Transformateur personnalisé pour supprimer les outliers using IQR
    """
    
    def __init__(self, columns: List[str] = None, factor: float = 1.5):
        """
        Initialise le suppresseur d'outliers
        
        Args:
            columns (List[str]): Colonnes à traiter
            factor (float): Facteur IQR (1.5 standard)
        """
        self.columns = columns
        self.factor = factor
        self.bounds_ = {}
    
    def fit(self, X, y=None):
        """
        Calcule les bornes pour chaque colonne
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in self.columns:
            if col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.factor * IQR
                upper_bound = Q3 + self.factor * IQR
                self.bounds_[col] = (lower_bound, upper_bound)
        
        return self
    
    def transform(self, X):
        """
        Supprime les outliers
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                mask = (X[col] >= lower) & (X[col] <= upper)
                X = X[mask]
        
        return X

class DataValidator(BaseEstimator, TransformerMixin):
    """
    Transformateur pour valider et nettoyer les données
    """
    
    def __init__(self):
        self.validation_rules = {
            'prix_dh': {'min': 50000, 'max': 50000000},  # Prix entre 50k et 50M DH
            'surface_m2': {'min': 15, 'max': 1000},       # Surface entre 15 et 1000 m²
            'nombre_chambres': {'min': 0, 'max': 10},     # 0 à 10 chambres
            'annee_construction': {'min': 1950, 'max': 2025}  # Années réalistes
        }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Applique les règles de validation
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        initial_count = len(X)
        
        for col, rules in self.validation_rules.items():
            if col in X.columns:
                # Suppression des valeurs hors limites
                mask = (X[col] >= rules['min']) & (X[col] <= rules['max'])
                X = X[mask]
        
        # Suppression des lignes avec des valeurs négatives
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ['prix_dh', 'surface_m2']:  # Ces colonnes ne peuvent pas être négatives
                X = X[X[col] > 0]
        
        final_count = len(X)
        removed = initial_count - final_count
        
        if removed > 0:
            logger.info(f"Validation: {removed} lignes supprimées ({removed/initial_count:.1%})")
        
        return X

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformateur pour créer de nouvelles features
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Crée de nouvelles features
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Prix au m²
        if 'prix_dh' in X.columns and 'surface_m2' in X.columns:
            X['prix_par_m2'] = X['prix_dh'] / X['surface_m2']
        
        # Surface par chambre
        if 'surface_m2' in X.columns and 'nombre_chambres' in X.columns:
            X['surface_par_chambre'] = X['surface_m2'] / (X['nombre_chambres'] + 1)  # +1 pour éviter division par 0
        
        # Âge du bien
        if 'annee_construction' in X.columns:
            X['age_bien'] = 2025 - X['annee_construction']
        
        # Catégories de surface
        if 'surface_m2' in X.columns:
            X['categorie_surface'] = pd.cut(
                X['surface_m2'],
                bins=[0, 50, 100, 150, 300, float('inf')],
                labels=['Très petit', 'Petit', 'Moyen', 'Grand', 'Très grand']
            )
        
        # Catégories de prix
        if 'prix_dh' in X.columns:
            X['categorie_prix'] = pd.cut(
                X['prix_dh'],
                bins=[0, 500000, 1000000, 2000000, 5000000, float('inf')],
                labels=['Économique', 'Abordable', 'Moyen', 'Élevé', 'Luxe']
            )
        
        return X

def create_preprocessing_pipeline() -> Pipeline:
    """
    Crée la pipeline complète de prétraitement
    
    Returns:
        Pipeline: Pipeline de prétraitement
    """
    
    # Étape 1: Validation et nettoyage initial
    validation_step = DataValidator()
    
    # Étape 2: Création de nouvelles features
    feature_engineering_step = FeatureEngineer()
    
    # Étape 3: Suppression des outliers
    outlier_removal_step = OutlierRemover(
        columns=['prix_dh', 'surface_m2', 'prix_par_m2'],
        factor=1.5
    )
    
    # Pipeline principale
    preprocessing_pipeline = Pipeline([
        ('validation', validation_step),
        ('feature_engineering', feature_engineering_step),
        ('outlier_removal', outlier_removal_step)
    ])
    
    return preprocessing_pipeline

def create_modeling_pipeline() -> ColumnTransformer:
    """
    Crée la pipeline pour la préparation des données pour le machine learning
    
    Returns:
        ColumnTransformer: Transformateur pour les features
    """
    
    # Définition des colonnes par type
    numeric_features = ['surface_m2', 'nombre_chambres', 'age_bien', 'surface_par_chambre']
    categorical_features = ['localisation', 'type_bien']
    
    # Transformateurs pour chaque type de features
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combinaison des transformateurs
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    return preprocessor

class DataCleaner:
    """
    Classe principale pour le nettoyage des données
    """
    
    def __init__(self):
        self.preprocessing_pipeline = create_preprocessing_pipeline()
        self.modeling_pipeline = create_modeling_pipeline()
        self.is_fitted = False
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données brutes
        
        Args:
            df (pd.DataFrame): Données brutes
            
        Returns:
            pd.DataFrame: Données nettoyées
        """
        logger.info(f"Début du nettoyage - {len(df)} lignes")
        
        # Application de la pipeline de prétraitement
        cleaned_df = self.preprocessing_pipeline.fit_transform(df)
        
        # Conversion en DataFrame si nécessaire
        if not isinstance(cleaned_df, pd.DataFrame):
            cleaned_df = pd.DataFrame(cleaned_df, columns=df.columns)
        
        # Suppression des doublons
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        duplicates_removed = initial_count - len(cleaned_df)
        
        if duplicates_removed > 0:
            logger.info(f"Doublons supprimés: {duplicates_removed}")
        
        # Gestion des valeurs manquantes critiques
        critical_columns = ['prix_dh', 'surface_m2']
        for col in critical_columns:
            if col in cleaned_df.columns:
                before = len(cleaned_df)
                cleaned_df = cleaned_df.dropna(subset=[col])
                after = len(cleaned_df)
                if before != after:
                    logger.info(f"Lignes avec {col} manquant supprimées: {before - after}")
        
        logger.info(f"Nettoyage terminé - {len(cleaned_df)} lignes restantes")
        
        return cleaned_df
    
    def prepare_for_modeling(self, df: pd.DataFrame, target_column: str = 'prix_dh') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prépare les données pour la modélisation
        
        Args:
            df (pd.DataFrame): Données nettoyées
            target_column (str): Nom de la colonne cible
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features (X) et target (y)
        """
        # Séparation des features et de la target
        X = df.drop(columns=[target_column])
        y = df[target_column].values
        
        # Application de la transformation
        X_transformed = self.modeling_pipeline.fit_transform(X)
        
        self.is_fitted = True
        
        return X_transformed, y
    
    def get_feature_names(self) -> List[str]:
        """
        Récupère les noms des features après transformation
        
        Returns:
            List[str]: Noms des features
        """
        if not self.is_fitted:
            raise ValueError("Le pipeline doit être fitté avant de récupérer les noms des features")
        
        feature_names = []
        
        # Features numériques
        numeric_features = ['surface_m2', 'nombre_chambres', 'age_bien', 'surface_par_chambre']
        feature_names.extend(numeric_features)
        
        # Features catégoriques (après OneHot encoding)
        try:
            cat_transformer = self.modeling_pipeline.named_transformers_['cat']
            onehot_encoder = cat_transformer.named_steps['onehot']
            cat_feature_names = onehot_encoder.get_feature_names_out(['localisation', 'type_bien'])
            feature_names.extend(cat_feature_names)
        except:
            # Fallback si l'accès aux noms échoue
            feature_names.extend(['cat_feature_' + str(i) for i in range(10)])
        
        return feature_names
    
    def save_cleaned_data(self, df: pd.DataFrame, filename: str = 'data/cleaned_properties.csv'):
        """
        Sauvegarde les données nettoyées
        
        Args:
            df (pd.DataFrame): Données nettoyées
            filename (str): Nom du fichier
        """
        df.to_csv(filename, index=False)
        logger.info(f"Données nettoyées sauvegardées dans {filename}")
    
    def generate_cleaning_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
        """
        Génère un rapport de nettoyage
        
        Args:
            original_df (pd.DataFrame): Données originales
            cleaned_df (pd.DataFrame): Données nettoyées
            
        Returns:
            dict: Rapport de nettoyage
        """
        report = {
            'original_rows': len(original_df),
            'cleaned_rows': len(cleaned_df),
            'rows_removed': len(original_df) - len(cleaned_df),
            'removal_percentage': (len(original_df) - len(cleaned_df)) / len(original_df) * 100,
            'missing_values_original': original_df.isnull().sum().to_dict(),
            'missing_values_cleaned': cleaned_df.isnull().sum().to_dict(),
            'columns_original': list(original_df.columns),
            'columns_cleaned': list(cleaned_df.columns),
            'new_features': [col for col in cleaned_df.columns if col not in original_df.columns]
        }
        
        return report

if __name__ == "__main__":
    # Exemple d'utilisation
    
    # Chargement des données
    try:
        df = pd.read_csv('data/raw_properties.csv')
        print(f"Données chargées: {len(df)} lignes")
        
        # Initialisation du cleaner
        cleaner = DataCleaner()
        
        # Nettoyage
        cleaned_df = cleaner.clean_data(df)
        
        # Sauvegarde
        cleaner.save_cleaned_data(cleaned_df)
        
        # Génération du rapport
        report = cleaner.generate_cleaning_report(df, cleaned_df)
        print(f"Rapport de nettoyage:")
        print(f"- Lignes originales: {report['original_rows']}")
        print(f"- Lignes nettoyées: {report['cleaned_rows']}")
        print(f"- Lignes supprimées: {report['rows_removed']} ({report['removal_percentage']:.1f}%)")
        print(f"- Nouvelles features: {report['new_features']}")
        
    except FileNotFoundError:
        print("Fichier de données non trouvé. Exécutez d'abord data_scraper.py")
