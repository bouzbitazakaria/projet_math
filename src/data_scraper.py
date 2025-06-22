"""
Module de collecte de données immobilières par web scraping
Auteur: Projet Math IA
Date: Juin 2025
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
from typing import List, Dict, Optional
import re
from tqdm import tqdm
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealEstateScraper:
    """
    Classe pour scraper les données immobilières
    """
    
    def __init__(self, base_url: str, delay_range: tuple = (1, 3)):
        """
        Initialise le scraper
        
        Args:
            base_url (str): URL de base du site à scraper
            delay_range (tuple): Délai entre les requêtes (min, max) en secondes
        """
        self.base_url = base_url
        self.delay_range = delay_range
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.properties_data = []
    
    def get_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Récupère une page web et retourne un objet BeautifulSoup
        
        Args:
            url (str): URL de la page à récupérer
            
        Returns:
            Optional[BeautifulSoup]: Objet BeautifulSoup ou None en cas d'erreur
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la récupération de {url}: {e}")
            return None
    
  def extract_price(self, text: str) -> Optional[float]:
        """
        Extrait le prix d'un texte
        
        Args:
            text (str): Texte contenant le prix
            
        Returns:
            Optional[float]: Prix en DH ou None
        """
        if not text:
            return None
        
        # Recherche de motifs de prix (ex: "1 250 000 DH", "1,250,000 DH", etc.)
        price_patterns = [
            r'(\d{1,3}(?:[,\s]\d{3})*)\s*(?:DH|dh|Dh)',
            r'(\d+(?:\.\d+)?)\s*(?:DH|dh|Dh)',
            r'(\d{1,3}(?:[,\s]\d{3})*)'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text.replace(' ', '').replace(',', ''))
            if match:
                try:
                    return float(match.group(1).replace(',', '').replace(' ', ''))
                except ValueError:
                    continue
        
        return None
    
  def extract_surface(self, text: str) -> Optional[float]:
        """
        Extrait la surface d'un texte
        
        Args:
            text (str): Texte contenant la surface
            
        Returns:
            Optional[float]: Surface en m² ou None
        """
        if not text:
            return None
        
        surface_patterns = [
            r'(\d+(?:\.\d+)?)\s*m[²2]',
            r'(\d+(?:\.\d+)?)\s*m\s*[²2]',
            r'surface[:\s]*(\d+(?:\.\d+)?)',
        ]
        
        for pattern in surface_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
def extract_rooms(self, text: str) -> Optional[int]:
        """
        Extrait le nombre de chambres d'un texte
        
        Args:
            text (str): Texte contenant le nombre de chambres
            
        Returns:
            Optional[int]: Nombre de chambres ou None
        """
        if not text:
            return None
        
        room_patterns = [
            r'(\d+)\s*(?:chambre|bedroom|room)',
            r'(\d+)\s*ch',
            r'(\d+)\s*p[iè]ces?',
        ]
        
        for pattern in room_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return None
    
  def scrape_property_details(self, property_element) -> Dict:
        """
        Extrait les détails d'une propriété à partir d'un élément HTML
        
        Args:
            property_element: Élément HTML contenant les informations de la propriété
            
        Returns:
            Dict: Dictionnaire contenant les données de la propriété
        """
        property_data = {
            'prix_dh': None,
            'surface_m2': None,
            'nombre_chambres': None,
            'localisation': None,
            'type_bien': None,
            'annee_construction': None
        }
        
        try:
            # Extraction du texte complet de l'élément
            full_text = property_element.get_text(separator=' ', strip=True)
            
            # Extraction du prix
            property_data['prix_dh'] = self.extract_price(full_text)
            
            # Extraction de la surface
            property_data['surface_m2'] = self.extract_surface(full_text)
            
            # Extraction du nombre de chambres
            property_data['nombre_chambres'] = self.extract_rooms(full_text)
            
            # Extraction de la localisation (généralement dans une classe spécifique)
            location_element = property_element.find(['span', 'div'], class_=re.compile(r'location|address|ville'))
            if location_element:
                property_data['localisation'] = location_element.get_text(strip=True)
            
            # Extraction du type de bien
            type_element = property_element.find(['span', 'div'], class_=re.compile(r'type|category'))
            if type_element:
                property_data['type_bien'] = type_element.get_text(strip=True)
            
            # Extraction de l'année de construction
            year_match = re.search(r'(19|20)\d{2}', full_text)
            if year_match:
                property_data['annee_construction'] = int(year_match.group())
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des détails: {e}")
        
      return property_data
    
    def scrape_page(self, page_url: str) -> List[Dict]:
        """
        Scrape une page d'annonces immobilières
        
        Args:
            page_url (str): URL de la page à scraper
            
        Returns:
            List[Dict]: Liste des propriétés trouvées sur la page
        """
        soup = self.get_page(page_url)
        if not soup:
            return []
        
        properties = []
        
        # Sélecteurs CSS communs pour les annonces immobilières
        selectors = [
            '.property-item',
            '.listing-item',
            '.ad-item',
            '.property-card',
            '[class*="property"]',
            '[class*="listing"]',
            '[class*="annonce"]'
        ]
        
        property_elements = []
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                property_elements = elements
                break
        
        for element in property_elements[:50]:  # Limiter à 50 par page
            property_data = self.scrape_property_details(element)
            if property_data['prix_dh']:  # Seulement si on a un prix
                properties.append(property_data)
        
        
        return properties
    
    def scrape_multiple_pages(self, max_properties: int = 200) -> pd.DataFrame:
        """
        Scrape plusieurs pages jusqu'à atteindre le nombre maximum de propriétés
        
        Args:
            max_properties (int): Nombre maximum de propriétés à collecter
            
        Returns:
            pd.DataFrame: DataFrame contenant toutes les propriétés collectées
        """
        all_properties = []
        page = 1
        
        with tqdm(total=max_properties, desc="Collecte des données") as pbar:
            while len(all_properties) < max_properties:
                # Construction de l'URL de la page
                page_url = f"{self.base_url}?page={page}"
                
                logger.info(f"Scraping page {page}: {page_url}")
                
                # Scraping de la page
                properties = self.scrape_page(page_url)
                
                if not properties:
                    logger.info("Aucune propriété trouvée, arrêt du scraping")
                    break
                
                all_properties.extend(properties)
                pbar.update(len(properties))
                
                # Délai entre les requêtes
                delay = random.uniform(*self.delay_range)
                time.sleep(delay)
                
                page += 1
                
                # Sécurité: arrêt après 20 pages
                if page > 20:
                    break
        
        self.properties_data = all_properties[:max_properties]
        logger.info(f"Collecte terminée: {len(self.properties_data)} propriétés collectées")
        
        return pd.DataFrame(self.properties_data)
    
    def save_data(self, filename: str = 'data/raw_properties.csv'):
        """
        Sauvegarde les données collectées
        
        Args:
            filename (str): Nom du fichier de sauvegarde
        """
        if self.properties_data:
            df = pd.DataFrame(self.properties_data)
            df.to_csv(filename, index=False, encoding='utf-8')
            logger.info(f"Données sauvegardées dans {filename}")
        else:
            logger.warning("Aucune donnée à sauvegarder")

def generate_sample_data(n_samples: int = 150) -> pd.DataFrame:
    """
    Génère des données d'exemple pour le développement et les tests
    
    Args:
        n_samples (int): Nombre d'échantillons à générer
        
    Returns:
        pd.DataFrame: DataFrame avec des données d'exemple
    """
    import numpy as np
    
    np.random.seed(42)
    
    # Définition des paramètres
    villes = ['Casablanca', 'Rabat', 'Marrakech', 'Fès', 'Tanger', 'Agadir', 'Oujda', 'Meknès']
    types_bien = ['Appartement', 'Maison', 'Villa', 'Duplex', 'Studio']
    
    # Génération des données
    data = []
    for i in range(n_samples):
        # Surface entre 30 et 300 m²
        surface = np.random.uniform(30, 300)
        
        # Nombre de chambres basé sur la surface
        if surface < 50:
            chambres = np.random.choice([1, 2], p=[0.7, 0.3])
        elif surface < 100:
            chambres = np.random.choice([2, 3], p=[0.6, 0.4])
        elif surface < 150:
            chambres = np.random.choice([3, 4], p=[0.7, 0.3])
        else:
            chambres = np.random.choice([4, 5, 6], p=[0.5, 0.3, 0.2])
        
        # Localisation
        ville = np.random.choice(villes)
        
        # Type de bien
        type_bien = np.random.choice(types_bien)
        
        # Année de construction
        annee = np.random.randint(1980, 2025)
        
        # Prix basé sur surface, localisation, type, etc.
        prix_base = surface * np.random.uniform(8000, 15000)  # Prix au m²
        
        # Facteur ville
        facteurs_ville = {
            'Casablanca': 1.3, 'Rabat': 1.2, 'Marrakech': 1.1,
            'Fès': 0.9, 'Tanger': 1.0, 'Agadir': 1.1,
            'Oujda': 0.8, 'Meknès': 0.8
        }
        prix_base *= facteurs_ville[ville]
        
        # Facteur type de bien
        facteurs_type = {
            'Studio': 0.8, 'Appartement': 1.0, 'Duplex': 1.2,
            'Maison': 1.1, 'Villa': 1.4
        }
        prix_base *= facteurs_type[type_bien]
        
        # Facteur âge
        age = 2025 - annee
        facteur_age = max(0.7, 1 - age * 0.01)
        prix_base *= facteur_age
        
        # Ajout de bruit
        prix_final = prix_base * np.random.uniform(0.8, 1.2)
        
        data.append({
            'prix_dh': round(prix_final, 0),
            'surface_m2': round(surface, 1),
            'nombre_chambres': chambres,
            'localisation': ville,
            'type_bien': type_bien,
            'annee_construction': annee
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Configuration
    BASE_URL = "https://www.sarouty.ma/fr/recherche?c=1&fu=0&ob=mr"  # Remplacer par un vrai site
    
    # Option 1: Scraping réel (décommenter si vous avez un site cible)
    scraper = RealEstateScraper(BASE_URL)
    df = scraper.scrape_multiple_pages(max_properties=150)
    scraper.save_data('data/raw_properties.csv')
    
    # Option 2: Génération de données d'exemple (pour le développement)
    # print("Génération de données d'exemple...")
    # df = generate_sample_data(150)
    # df.to_csv('data/raw_properties.csv', index=False)
    # print(f"Données générées et sauvegardées: {len(df)} propriétés")
    # print(df.head())
