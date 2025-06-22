# 🏠 Projet d'Analyse Exploratoire de Données : Prédiction des Prix Immobiliers

## 📋 Description du Projet

Ce projet implémente une solution complète d'analyse exploratoire des données (EDA) et de modélisation pour la prédiction des prix immobiliers. Il inclut la collecte de données par web scraping, le nettoyage des données avec une pipeline automatisée, l'analyse exploratoire approfondie, et le développement de modèles de régression multiple.

**Auteur :** Projet Math IA  
**Date :** Juin 2025  
**Objectif :** Développer un modèle de régression multiple pour prédire les prix des biens immobiliers

---

## 🎯 Objectifs du Projet

1. **Collecte des données** : Scraper les annonces immobilières depuis des sites web
2. **Nettoyage des données** : Construire une pipeline de traitement avec pandas et scikit-learn
3. **Analyse exploratoire** : Explorer les relations entre variables et identifier les facteurs clés
4. **Modélisation** : Développer un modèle de régression multiple performant
5. **Évaluation** : Analyser les performances et l'importance des variables

---

## 📁 Structure du Projet

```
projet_math/
├── data/                           # Données
│   ├── raw_properties.csv         # Données brutes
│   └── cleaned_properties.csv     # Données nettoyées
├── src/                           # Code source
│   ├── data_scraper.py           # Module de web scraping
│   ├── data_pipeline.py          # Pipeline de nettoyage
│   ├── eda_analysis.py           # Analyse exploratoire
│   └── modeling.py               # Modélisation
├── notebooks/                     # Notebooks Jupyter
│   └── analyse_prix_immobilier.ipynb  # Notebook principal
├── visualizations/               # Graphiques générés
│   ├── price_distribution.png
│   ├── correlation_matrix.png
│   ├── model_comparison.png
│   └── interactive_dashboard.html
├── models/                       # Modèles sauvegardés
│   └── best_price_predictor.pkl
├── main.py                      # Script principal
├── requirements.txt             # Dépendances
└── README.md                   # Ce fichier
```

---

## 🚀 Installation et Configuration

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des dépendances

```bash
# Cloner le projet (si applicable)
cd projet_math

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales
- **pandas** : Manipulation des données
- **numpy** : Calculs numériques
- **scikit-learn** : Machine learning
- **matplotlib/seaborn** : Visualisation
- **plotly** : Graphiques interactifs
- **beautifulsoup4** : Web scraping
- **requests** : Requêtes HTTP

---

## 🔧 Utilisation

### Option 1 : Exécution complète automatique

```bash
python main.py
```

Ce script exécute automatiquement tout le pipeline :
1. Collecte/génération des données
2. Nettoyage et préparation
3. Analyse exploratoire (EDA)
4. Modélisation et évaluation

### Option 2 : Exécution par étapes

```bash
# 1. Collecte des données
python src/data_scraper.py

# 2. Nettoyage des données  
python src/data_pipeline.py

# 3. Analyse exploratoire
python src/eda_analysis.py

# 4. Modélisation
python src/modeling.py
```

### Option 3 : Notebook Jupyter

```bash
jupyter notebook notebooks/analyse_prix_immobilier.ipynb
```

---

## 📊 Fonctionnalités Principales

### 1. Collecte de Données (Web Scraping)
- **Module** : `src/data_scraper.py`
- **Fonctionnalités** :
  - Scraping respectueux des sites immobiliers
  - Gestion des erreurs et timeouts
  - Génération de données d'exemple pour les tests
  - Extraction automatique des prix, surfaces, localisations

### 2. Pipeline de Nettoyage
- **Module** : `src/data_pipeline.py`
- **Fonctionnalités** :
  - Suppression des doublons
  - Gestion des valeurs manquantes
  - Validation selon des règles métier
  - Détection et traitement des outliers (IQR)
  - Création de nouvelles features
  - Encodage des variables catégoriques

### 3. Analyse Exploratoire (EDA)
- **Module** : `src/eda_analysis.py`
- **Visualisations** :
  - Distribution des prix (histogramme, boxplot, Q-Q plot)
  - Matrice de corrélation
  - Analyse par localisation et type de bien
  - Relation prix-surface
  - Tableau de bord interactif (HTML)

### 4. Modélisation
- **Module** : `src/modeling.py`
- **Modèles testés** :
  - Régression Linéaire
  - Régression Ridge
  - Régression Lasso
  - Random Forest
  - Gradient Boosting
- **Métriques d'évaluation** :
  - R² (coefficient de détermination)
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)

---

## 📈 Résultats Attendus

### Visualisations Générées
1. **Distribution des prix** : Histogrammes et box plots
2. **Corrélations** : Heatmap des corrélations entre variables
3. **Analyses géographiques** : Prix par ville et région
4. **Comparaison des modèles** : Performances relatives
5. **Importance des features** : Variables les plus influentes
6. **Dashboard interactif** : Vue d'ensemble avec Plotly

### Modèles et Performances
- Comparaison de 5 algorithmes de régression
- Sélection automatique du meilleur modèle
- Optimisation des hyperparamètres
- Validation croisée pour éviter l'overfitting
- Sauvegarde du modèle final

---

## 🔍 Variables Analysées

### Variables Collectées
- **prix_dh** : Prix du bien en dirhams (variable cible)
- **surface_m2** : Surface en mètres carrés
- **nombre_chambres** : Nombre de chambres
- **localisation** : Ville ou zone géographique
- **type_bien** : Type de propriété (appartement, maison, villa, etc.)
- **annee_construction** : Année de construction

### Features Créées
- **prix_par_m2** : Prix au mètre carré
- **surface_par_chambre** : Surface moyenne par chambre
- **age_bien** : Âge du bien (2025 - année de construction)
- **categorie_surface** : Catégorisation de la surface
- **categorie_prix** : Catégorisation du prix

---

## 📋 Métriques et Seuils de Validation

### Règles de Validation des Données
- **Prix** : Entre 50 000 et 50 000 000 DH
- **Surface** : Entre 15 et 1 000 m²
- **Chambres** : Entre 0 et 10
- **Année de construction** : Entre 1950 et 2025

### Métriques de Performance
- **R² > 0.7** : Performance excellente
- **R² > 0.5** : Performance acceptable
- **MAPE < 15%** : Erreur relative acceptable

---

## 🛠️ Personnalisation

### Configuration du Web Scraping
```python
# Dans src/data_scraper.py
BASE_URL = "https://votre-site-immobilier.com"  # Modifier l'URL
MAX_PROPERTIES = 200  # Nombre max de propriétés
DELAY_RANGE = (1, 3)  # Délai entre requêtes
```

### Ajout de Nouvelles Features
```python
# Dans src/data_pipeline.py - Classe FeatureEngineer
def transform(self, X):
    # Ajouter vos nouvelles features ici
    X['nouvelle_feature'] = X['col1'] / X['col2']
    return X
```

### Ajout de Nouveaux Modèles
```python
# Dans src/modeling.py - Méthode _initialize_models
self.models['Nouveau_Modele'] = VotreModele(paramètres)
```

---

## 📊 Exemples de Sortie

### Statistiques Descriptives
```
📊 RAPPORT DE NETTOYAGE:
• Lignes originales: 200
• Lignes nettoyées: 185
• Lignes supprimées: 15 (7.5%)
• Nouvelles features créées: 5
```

### Performance des Modèles
```
🏆 Meilleur modèle: Random_Forest
📊 Performances:
   • R²: 0.8245
   • RMSE: 89,542 DH
   • MAE: 67,234 DH
   • MAPE: 12.3%
```

---

## 🔧 Dépannage

### Erreurs Communes

1. **Fichier de données non trouvé**
   ```bash
   # Solution : Exécuter d'abord la collecte
   python src/data_scraper.py
   ```

2. **Erreur d'import des modules**
   ```bash
   # Solution : Vérifier le PYTHONPATH
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

3. **Erreur de visualisation**
   ```bash
   # Solution : Installer les dépendances manquantes
   pip install matplotlib seaborn plotly
   ```

### Optimisation des Performances
- Réduire le nombre de propriétés pour des tests rapides
- Utiliser `n_jobs=-1` pour la parallélisation
- Ajuster les paramètres des modèles selon vos données

---

## 📚 Documentation Technique

### Architecture du Code
- **Modularité** : Chaque étape dans un module séparé
- **Réutilisabilité** : Classes et fonctions réutilisables
- **Extensibilité** : Facile d'ajouter de nouveaux modèles/features
- **Robustesse** : Gestion d'erreurs et validation des données

### Design Patterns Utilisés
- **Pipeline Pattern** : Pour le preprocessing
- **Strategy Pattern** : Pour les différents modèles
- **Factory Pattern** : Pour la création des transformateurs

---

## 🎯 Améliorations Futures

### Court Terme
- [ ] Interface web pour les prédictions
- [ ] API REST pour le modèle
- [ ] Collecte de données en temps réel
- [ ] Notifications de nouveaux biens

### Long Terme
- [ ] Modèles de deep learning
- [ ] Données géospatiales (GPS, cartes)
- [ ] Analyse de sentiment des descriptions
- [ ] Prédiction de tendances du marché

---

## 🤝 Contribution

### Comment Contribuer
1. Fork le projet
2. Créer une branche pour votre feature
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

### Standards de Code
- Suivre PEP 8 pour Python
- Documenter les fonctions avec des docstrings
- Ajouter des tests unitaires
- Commenter le code complexe

---

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

---

## 📞 Contact

**Auteur :** Projet Math IA  
**Email :** [votre-email@example.com]  
**Projet :** Analyse Exploratoire des Données Immobilières

---

## 🙏 Remerciements

- **Scikit-learn** pour les outils de machine learning
- **Pandas** pour la manipulation des données
- **Plotly** pour les visualisations interactives
- **BeautifulSoup** pour le web scraping
- **Matplotlib/Seaborn** pour les graphiques statistiques

---

*Dernière mise à jour : Juin 2025*
