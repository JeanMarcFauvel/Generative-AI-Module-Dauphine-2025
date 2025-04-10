#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour lancer l'application Flask depuis le bon emplacement.
"""

import os
import sys
from pathlib import Path

# Ajouter le dossier app au chemin Python
app_dir = Path(__file__).parent / "app"
sys.path.append(str(app_dir))

# Importer et lancer l'application
from app import app

if __name__ == "__main__":
    app.run(debug=True, port=8080) 