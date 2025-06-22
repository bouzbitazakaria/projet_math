"""
Script principal pour l'analyse exploratoire et la prédiction des prix immobiliers
Auteur: Projet Math IA
Date: Juin 2025
"""

import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Ajout du répertoire src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_scraper import RealEstateScraper, generate_sample_data
from src.data_pipeline import DataCleaner
from src.eda_analysis import RealEstateEDA
from src.modeling import RealEstatePricePredictor

def print_header(title: str):
    """Affiche un en-tête formaté"""
    print("\n" + "=" * 80)
    print(f"🏠 {title}")
    print("=" * 80)

def print_step(step: str):
    """Affiche une étape formatée"""
    print(f"\n📋 {step}")
    print("-" * 60)

def main():
    """
    Fonction principale qui exécute tout le pipeline d'analyse
    """
    print_header("PROJET D'ANALYSE EXPLORATOIRE DE DONNÉES IMMOBILIÈRES")
    print("Prédiction des prix immobiliers - Pipeline complet")
    print("Auteur: Projet Math IA")
    print("Date: Juin 2025")
    
    # ===== ÉTAPE 1: COLLECTE DES DONNÉES =====
    print_step("ÉTAPE 1: COLLECTE DES DONNÉES")
    
    try:
        # Vérifier si les données existent déjà
        if os.path.exists('data/raw_properties.csv'):
            print("✅ Données brutes trouvées, chargement...")
            df_raw = pd.read_csv('data/raw_properties.csv')
        else:
            print("🔍 Génération de données d'exemple...")
            print("(Pour un vrai projet, remplacez par du web scraping)")
            
            # Génération de données d'exemple
            df_raw = generate_sample_data(200)
            
            # Sauvegarde
            os.makedirs('data', exist_ok=True)
            df_raw.to_csv('data/raw_properties.csv', index=False)
            print("💾 Données sauvegardées dans 'data/raw_properties.csv'")
        
        print(f"📊 Données collectées: {len(df_raw)} propriétés")
        print(f"📈 Colonnes: {list(df_raw.columns)}")
        print(f"💰 Prix moyen: {df_raw['prix_dh'].mean():,.0f} DH")
        
    except Exception as e:
        print(f"❌ Erreur lors de la collecte: {e}")
        return
    
    # ===== ÉTAPE 2: NETTOYAGE DES DONNÉES =====
    print_step("ÉTAPE 2: NETTOYAGE ET PRÉPARATION DES DONNÉES")
    
    try:
        # Initialisation du nettoyeur
        cleaner = DataCleaner()
        
        # Nettoyage des données
        df_cleaned = cleaner.clean_data(df_raw)
        
        # Sauvegarde des données nettoyées
        cleaner.save_cleaned_data(df_cleaned, 'data/cleaned_properties.csv')
        
        # Génération du rapport de nettoyage
        cleaning_report = cleaner.generate_cleaning_report(df_raw, df_cleaned)
        
        print("✅ Nettoyage terminé!")
        print(f"📊 Données originales: {cleaning_report['original_rows']} lignes")
        print(f"📊 Données nettoyées: {cleaning_report['cleaned_rows']} lignes")
        print(f"🗑️ Supprimées: {cleaning_report['rows_removed']} lignes ({cleaning_report['removal_percentage']:.1f}%)")
        print(f"✨ Nouvelles features: {cleaning_report['new_features']}")
        
    except Exception as e:
        print(f"❌ Erreur lors du nettoyage: {e}")
        return
    
    # ===== ÉTAPE 3: ANALYSE EXPLORATOIRE (EDA) =====
    print_step("ÉTAPE 3: ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
    
    try:
        # Initialisation de l'analyseur EDA
        eda = RealEstateEDA(df_cleaned)
        
        # Exécution de l'analyse complète
        eda_report = eda.run_complete_eda()
        
        print("✅ Analyse EDA terminée!")
        print("📊 Visualisations générées:")
        print("   • Distribution des prix")
        print("   • Matrice de corrélation")
        print("   • Analyse par localisation")
        print("   • Relation prix-surface")
        print("   • Analyse par type de bien")
        print("   • Analyse par nombre de chambres")
        print("   • Tableau de bord interactif")
        
        # Affichage des insights clés
        print("\n🎯 Insights clés découverts:")
        for insight in eda_report['insights_cles']:
            print(f"   • {insight}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'EDA: {e}")
        # Continuer même si l'EDA échoue
    
    # ===== ÉTAPE 4: MODÉLISATION =====
    print_step("ÉTAPE 4: MODÉLISATION ET PRÉDICTION")
    
    try:
        # Initialisation du prédicteur
        predictor = RealEstatePricePredictor()
        
        # Préparation des données pour la modélisation
        X_train, X_test, y_train, y_test = predictor.prepare_data(df_cleaned)
        
        # Entraînement des modèles
        model_results = predictor.train_models(X_train, y_train, X_test, y_test)
        
        # Sélection du meilleur modèle
        best_model_name = predictor.select_best_model()
        
        # Génération des visualisations
        print("\n📊 Génération des visualisations de modélisation...")
        predictor.plot_model_comparison()
        predictor.plot_predictions_vs_actual(X_test, y_test)
        predictor.plot_feature_importance()
        
        # Sauvegarde du modèle
        os.makedirs('models', exist_ok=True)
        predictor.save_model('models/best_price_predictor.pkl')
        
        # Génération du rapport de modélisation
        modeling_report = predictor.generate_model_report()
        
        print("✅ Modélisation terminée!")
        print(f"🏆 Meilleur modèle: {best_model_name}")
        
        best_metrics = modeling_report['metriques_meilleur_modele']
        print(f"📊 Performances:")
        print(f"   • R²: {best_metrics['r2']:.4f}")
        print(f"   • RMSE: {best_metrics['rmse']:,.0f} DH")
        print(f"   • MAE: {best_metrics['mae']:,.0f} DH")
        print(f"   • MAPE: {best_metrics['mape']:.2f}%")
        
        # Affichage des recommandations
        print("\n💡 Recommandations d'amélioration:")
        for rec in modeling_report['recommandations']:
            print(f"   • {rec}")
        
        # Exemple de prédiction
        print_step("EXEMPLE DE PRÉDICTION")
        
        example_property = {
            'surface_m2': 120,
            'nombre_chambres': 3,
            'localisation': 'Casablanca',
            'type_bien': 'Appartement',
            'annee_construction': 2018
        }
        
        print("🏠 Propriété d'exemple:")
        for key, value in example_property.items():
            print(f"   • {key}: {value}")
        
        try:
            # predicted_price = predictor.predict_price(example_property)
            # print(f"\n💰 Prix prédit: {predicted_price:,.0f} DH")
            print("\n💰 Prédiction désactivée pour cette démo")
        except Exception as e:
            print(f"\n⚠️ Erreur lors de la prédiction: {e}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la modélisation: {e}")
        return
    
    # ===== RAPPORT FINAL =====
    print_step("RAPPORT FINAL DU PROJET")
    
    print("✅ Pipeline d'analyse complété avec succès!")
    print("\n📁 Fichiers générés:")
    print("   • data/raw_properties.csv - Données brutes")
    print("   • data/cleaned_properties.csv - Données nettoyées")
    print("   • visualizations/ - Graphiques et analyses")
    print("   • models/best_price_predictor.pkl - Meilleur modèle")
    
    print("\n📊 Visualisations disponibles:")
    print("   • Distribution des prix")
    print("   • Matrice de corrélation")
    print("   • Analyses par localisation et type")
    print("   • Comparaison des modèles")
    print("   • Analyse des prédictions")
    print("   • Importance des features")
    print("   • Tableau de bord interactif (HTML)")
    
    print("\n🎯 Résultats clés:")
    try:
        if 'best_metrics' in locals():
            print(f"   • Précision du modèle (R²): {best_metrics['r2']:.1%}")
            print(f"   • Erreur moyenne: {best_metrics['mae']:,.0f} DH")
            print(f"   • Modèle recommandé: {best_model_name}")
    except:
        pass
    
    print("\n🚀 Prochaines étapes recommandées:")
    print("   • Collecter plus de données réelles")
    print("   • Ajouter des features géographiques")
    print("   • Tester des modèles avancés (XGBoost, Neural Networks)")
    print("   • Déployer le modèle en production")
    print("   • Créer une application web de prédiction")
    
    print("\n" + "=" * 80)
    print("🎉 PROJET TERMINÉ AVEC SUCCÈS!")
    print("=" * 80)

if __name__ == "__main__":
    # Création des répertoires nécessaires
    os.makedirs('data', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Exécution du pipeline principal
    main()
