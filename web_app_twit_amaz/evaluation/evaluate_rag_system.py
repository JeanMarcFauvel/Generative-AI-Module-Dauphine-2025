#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script d'évaluation du système RAG

Ce script évalue les performances du système RAG sur un jeu de données d'évaluation.
Il compare les réponses générées par le système avec les données d'évaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from project_development_guide import RAGSystem
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm

class RAGEvaluator:
    """
    Classe pour évaluer les performances du système RAG.
    """
    
    def __init__(self, rag_system, eval_data_path):
        """
        Initialise l'évaluateur.
        
        Args:
            rag_system: Instance du système RAG à évaluer
            eval_data_path: Chemin vers le fichier de données d'évaluation
        """
        self.rag_system = rag_system
        self.eval_data_path = Path(eval_data_path)
        self.eval_data = None
        self.results = []
        
    def load_evaluation_data(self):
        """Charge les données d'évaluation."""
        if not self.eval_data_path.exists():
            raise FileNotFoundError(f"Le fichier d'évaluation {self.eval_data_path} n'existe pas.")
        
        self.eval_data = pd.read_csv(self.eval_data_path)
        print(f"Données d'évaluation chargées: {len(self.eval_data)} entrées")
        
    def evaluate_system(self, sample_size=None):
        """
        Évalue le système RAG sur les données d'évaluation.
        
        Args:
            sample_size: Nombre d'échantillons à évaluer (None pour tous)
        """
        if self.eval_data is None:
            self.load_evaluation_data()
        
        # Limiter la taille de l'échantillon si nécessaire
        eval_samples = self.eval_data
        if sample_size and sample_size < len(self.eval_data):
            eval_samples = self.eval_data.sample(n=sample_size, random_state=42)
        
        print(f"Évaluation sur {len(eval_samples)} échantillons...")
        
        for idx, row in tqdm(eval_samples.iterrows(), total=len(eval_samples)):
            # Utiliser le tweet du client comme question
            question = row.get('customer_tweet', '')
            # Utiliser le tweet de l'entreprise comme réponse attendue
            expected_answer = row.get('company_tweet', '')
            
            if not question:
                continue
                
            # Mesurer le temps de réponse
            start_time = time.time()
            generated_answer = self.rag_system.generate_response(question)
            response_time = time.time() - start_time
            
            # Calculer la similarité cosinus entre la réponse générée et la réponse attendue
            if expected_answer:
                expected_embedding = self.rag_system.get_embedding(expected_answer)
                generated_embedding = self.rag_system.get_embedding(generated_answer)
                similarity = self.rag_system.cosine_similarity_score(expected_embedding, generated_embedding)
            else:
                similarity = None
            
            # Enregistrer les résultats
            self.results.append({
                'question': question,
                'expected_answer': expected_answer,
                'generated_answer': generated_answer,
                'response_time': response_time,
                'similarity': similarity
            })
    
    def calculate_metrics(self):
        """Calcule les métriques d'évaluation."""
        if not self.results:
            return "Aucun résultat d'évaluation disponible."
        
        # Convertir les résultats en DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Calculer les métriques
        metrics = {
            'nombre_total_questions': len(results_df),
            'temps_moyen_reponse': results_df['response_time'].mean(),
            'temps_median_reponse': results_df['response_time'].median(),
            'temps_min_reponse': results_df['response_time'].min(),
            'temps_max_reponse': results_df['response_time'].max(),
        }
        
        # Calculer la similarité moyenne si disponible
        if 'similarity' in results_df.columns and results_df['similarity'].notna().any():
            metrics['similarite_moyenne'] = results_df['similarity'].mean()
            metrics['similarite_median'] = results_df['similarity'].median()
        
        return metrics
    
    def save_results(self, output_path='evaluation_results.json'):
        """Sauvegarde les résultats de l'évaluation."""
        if not self.results:
            print("Aucun résultat à sauvegarder.")
            return
        
        # Convertir les résultats en format JSON
        results_json = []
        for result in self.results:
            # Convertir les valeurs numpy en types Python natifs
            result_json = {}
            for key, value in result.items():
                if isinstance(value, np.float64):
                    result_json[key] = float(value)
                elif isinstance(value, np.int64):
                    result_json[key] = int(value)
                else:
                    result_json[key] = value
            results_json.append(result_json)
        
        # Sauvegarder les résultats
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'results': results_json,
                'metrics': self.calculate_metrics()
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Résultats sauvegardés dans {output_path}")
    
    def plot_response_times(self, output_path='response_times.png'):
        """Génère un graphique des temps de réponse."""
        if not self.results:
            print("Aucun résultat à visualiser.")
            return
        
        results_df = pd.DataFrame(self.results)
        
        plt.figure(figsize=(10, 6))
        plt.hist(results_df['response_time'], bins=20, alpha=0.7)
        plt.axvline(results_df['response_time'].mean(), color='r', linestyle='dashed', linewidth=2, label=f'Moyenne: {results_df["response_time"].mean():.2f}s')
        plt.xlabel('Temps de réponse (secondes)')
        plt.ylabel('Nombre de questions')
        plt.title('Distribution des temps de réponse')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path)
        print(f"Graphique des temps de réponse sauvegardé dans {output_path}")

def main():
    """Fonction principale pour l'évaluation du système RAG."""
    # Initialiser le système RAG
    rag_system = RAGSystem()
    
    # Charger les données d'entraînement
    train_data_path = Path("Generative-AI-Module-Dauphine-2025/data/twitter_data_clean_sample.csv")
    if train_data_path.exists():
        df = pd.read_csv(train_data_path)
        rag_system.initialize_chromadb(df)
        print(f"Données d'entraînement chargées: {len(df)} tweets")
    else:
        raise FileNotFoundError(f"Le fichier d'entraînement {train_data_path} n'existe pas.")
    
    # Chemin vers les données d'évaluation
    eval_data_path = Path("Generative-AI-Module-Dauphine-2025/data/twitter_data_clean_eval.csv")
    
    # Créer l'évaluateur
    evaluator = RAGEvaluator(rag_system, eval_data_path)
    
    # Évaluer le système
    evaluator.evaluate_system(sample_size=50)  # Limiter à 50 échantillons pour l'exemple
    
    # Afficher les métriques
    metrics = evaluator.calculate_metrics()
    print("\nMétriques d'évaluation:")
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    else:
        print(metrics)
    
    # Sauvegarder les résultats
    evaluator.save_results()
    
    # Générer le graphique des temps de réponse
    evaluator.plot_response_times()

if __name__ == "__main__":
    main() 