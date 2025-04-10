from flask import Flask, render_template, request, jsonify
from project_development_guide import RAGSystem
from pathlib import Path

# Définir les chemins des templates et des fichiers statiques
app_dir = Path(__file__).parent
template_dir = app_dir / "templates"
static_dir = app_dir / "static"
data_dir = app_dir.parent / "data"

app = Flask(__name__, 
            template_folder=str(template_dir),
            static_folder=str(static_dir))

# Initialiser le système RAG
rag_system = RAGSystem()

# Charger les données
data_path = data_dir / "twitter_data_clean_sample.csv"
if data_path.exists():
    import pandas as pd
    df = pd.read_csv(data_path)
    rag_system.initialize_chromadb(df)
else:
    raise FileNotFoundError(f"Le fichier de données {data_path} n'existe pas.")

@app.route('/')
def index():
    """Route pour la page d'accueil"""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Route pour traiter les questions des utilisateurs"""
    try:
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'Question manquante'}), 400
        
        # Utiliser le système RAG pour générer une réponse
        response = rag_system.generate_response(question)
        
        return jsonify({'response': response})
    
    except Exception as e:
        print(f"Erreur lors du traitement de la requête : {str(e)}")
        return jsonify({'error': 'Erreur interne du serveur'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080) 