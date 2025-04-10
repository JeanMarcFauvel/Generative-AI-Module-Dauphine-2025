#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Guide de Développement - Application Web RAG avec OpenAI API

Ce script contient le code et les explications pour développer une application web 
utilisant le principe RAG (Retrieval-Augmented Generation) avec l'API OpenAI.
"""

# Standard library imports
import configparser
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import openai
from openai import OpenAI
import sklearn  # Correction de l'importation de scikit-learn
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.utils import embedding_functions


class RAGSystem:
    """
    Classe principale pour gérer le système RAG (Retrieval-Augmented Generation).
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialise le système RAG.
        
        Args:
            config_path (str): Chemin vers le fichier de configuration
        """
        # Charger la configuration
        self.config = configparser.ConfigParser()
        if config_path is None:
            config_path = Path(__file__).parent / "config.ini"
        self.config.read(config_path)
        self.openai_key = self.config.get('OPENAI_API', 'OPENAI_KEY')
        
        # Configuration de l'API OpenAI
        openai.api_key = self.openai_key
        self.client = OpenAI(api_key=self.openai_key)
        
        # Initialiser ChromaDB
        self.chroma_client = chromadb.Client()
        self.collection = None
    
    def get_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """
        Obtenir l'embedding d'un texte en utilisant l'API OpenAI.
        
        Args:
            text (str): Le texte à encoder
            model (str): Le modèle d'embedding à utiliser
            
        Returns:
            List[float]: Le vecteur d'embedding
        """
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding
    
    def cosine_similarity_score(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculer la similarité cosinus entre deux vecteurs.
        
        Args:
            vec1 (List[float]): Premier vecteur
            vec2 (List[float]): Deuxième vecteur
            
        Returns:
            float: Score de similarité entre 0 et 1
        """
        return cosine_similarity([vec1], [vec2])[0][0]
    
    def initialize_chromadb(self, data: pd.DataFrame, collection_name: str = "twitter_data") -> None:
        """
        Initialiser et peupler la base de données ChromaDB.
        
        Args:
            data (pd.DataFrame): DataFrame contenant les données à indexer
            collection_name (str): Nom de la collection ChromaDB
        """
        # Créer la collection
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.openai_key,
                model_name="text-embedding-3-small"
            )
        )
        
        # Préparer les données pour l'insertion
        documents = data['customer_tweet'].tolist()
        ids = [str(i) for i in range(len(documents))]
        metadatas = [{"source": "twitter"} for _ in documents]
        
        # Insérer les données dans ChromaDB
        self.collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
    
    def generate_response(self, query: str, k: int = 3) -> str:
        """
        Générer une réponse en utilisant le principe RAG.
        
        Args:
            query (str): La requête de l'utilisateur
            k (int): Nombre de documents similaires à récupérer
            
        Returns:
            str: Réponse générée
        """
        if self.collection is None:
            raise ValueError("ChromaDB n'a pas été initialisé. Appelez initialize_chromadb() d'abord.")
        
        # Rechercher les documents les plus pertinents
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Construire le contexte
        context = "\n".join(results['documents'][0])
        
        # Générer la réponse avec GPT
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Tu es un assistant qui aide à analyser les tweets. Utilise le contexte fourni pour répondre aux questions."},
                {"role": "user", "content": f"Contexte: {context}\n\nQuestion: {query}"}
            ]
        )
        
        return response.choices[0].message.content

def main():
    """
    Fonction principale pour tester le système RAG.
    """
    # Créer une instance du système RAG
    rag_system = RAGSystem()
    
    # Charger les données
    data_path = Path("Generative-AI-Module-Dauphine-2025/data/twitter_data_clean_sample.csv")
    if not data_path.exists():
        print(f"Erreur: Le fichier de données {data_path} n'existe pas.")
        return
    
    df = pd.read_csv(data_path)
    print(f"Nombre total de tweets: {len(df)}")
    
    # Initialiser ChromaDB
    rag_system.initialize_chromadb(df)
    
    # Test avec un tweet exemple
    test_query = "Quels sont les sujets les plus discutés dans les tweets?"
    response = rag_system.generate_response(test_query)
    print(f"\nQuestion: {test_query}")
    print(f"Réponse: {response}")

if __name__ == "__main__":
    main() 