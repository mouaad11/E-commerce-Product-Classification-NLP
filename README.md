# E-commerce Text Classification Project

![GitHub](https://img.shields.io/github/license/mouaad11/E-commerce-Product-Classification-NLP)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-yellow)

## English

### Overview
This project implements a machine learning system for automatically classifying e-commerce product listings into appropriate categories based on their textual descriptions. Using natural language processing (NLP) techniques and several classification algorithms, the system can predict whether a product belongs to "Household", "Books", "Electronics", or "Clothing & Accessories" categories with high accuracy.

### Dataset
The project uses a dataset from Kaggle: [E-commerce Text Classification](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification)

### Features
- Comprehensive text preprocessing pipeline with NLTK
- Data exploration and visualization with matplotlib and seaborn
- Implementation of multiple classification algorithms:
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - Random Forest
- Hyperparameter tuning using GridSearchCV
- Model performance comparison and evaluation
- Model persistence for future use
- Custom threshold detection to handle uncertain predictions

### Results
- Achieved 97% accuracy with the optimized Random Forest model
- Comprehensive performance analysis with confusion matrices and classification reports
- Visual comparisons of model performance

### Project Structure
```
├── ecommerce_data_mining_project.ipynb  # Main Jupyter notebook with all code
├── models/                              # Saved model files
│   ├── ecommerce_classifier.pkl         # Trained classifier model
│   └── label_mapping.pkl                # Category label mapping
├── ecommerceDataset.csv                 # Dataset (not included - download from Kaggle)
├── requirements.txt                     # Project dependencies
└── README.md                            # This file
```

### Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn
- wordcloud
- joblib

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/ecommerce-text-classification.git
cd ecommerce-text-classification

# Install dependencies
pip install -r requirements.txt

# Download required NLTK resources
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### Usage
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification)
2. Place the CSV file in the project root directory
3. Run the Jupyter notebook:
```bash
jupyter notebook ecommerce_data_mining_project.ipynb
```

4. To use the trained model for predictions:
```python
from predict import predict_category

# Test with sample text
result, probability = predict_category("wireless bluetooth headphones with noise cancellation")
print(f"Predicted category: {result}")
print(f"Confidence: {probability:.2f}")
```

### Challenges Addressed
- Handling class imbalance in the training data
- Optimizing hyperparameters for improved performance
- Managing multilingual text and heterogeneous product descriptions
- Integrating the Python backend with a React.js frontend (detailed in the report)

### Next Steps
- Expand to more product categories
- Implement multilingual support
- Deploy as a web service
- Integrate with a real e-commerce platform

### Note
The complete project report (PDF) is available in French.

---

## Français

### Aperçu
Ce projet implémente un système d'apprentissage automatique pour classer automatiquement les produits e-commerce dans des catégories appropriées en se basant sur leurs descriptions textuelles. En utilisant des techniques de traitement du langage naturel (NLP) et plusieurs algorithmes de classification, le système peut prédire si un produit appartient aux catégories "Household" (Maison), "Books" (Livres), "Electronics" (Électroniques) ou "Clothing & Accessories" (Vêtements & Accessoires) avec une grande précision.

### Jeu de données
Le projet utilise un jeu de données de Kaggle : [E-commerce Text Classification](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification)

### Fonctionnalités
- Pipeline complet de prétraitement de texte avec NLTK
- Exploration de données et visualisation avec matplotlib et seaborn
- Implémentation de plusieurs algorithmes de classification :
  - K plus proches voisins (KNN)
  - Naive Bayes
  - Random Forest (Forêt aléatoire)
- Optimisation des hyperparamètres avec GridSearchCV
- Comparaison et évaluation des performances des modèles
- Persistance du modèle pour utilisation future
- Détection de seuil personnalisée pour gérer les prédictions incertaines

### Résultats
- Précision de 97% obtenue avec le modèle Random Forest optimisé
- Analyse complète des performances avec matrices de confusion et rapports de classification
- Comparaisons visuelles des performances des modèles

### Structure du projet
```
├── ecommerce_data_mining_project.ipynb  # Notebook Jupyter principal avec tout le code
├── models/                              # Fichiers de modèles sauvegardés
│   ├── ecommerce_classifier.pkl         # Modèle de classification entraîné
│   └── label_mapping.pkl                # Mapping des étiquettes de catégories
├── ecommerceDataset.csv                 # Jeu de données (non inclus - à télécharger depuis Kaggle)
├── requirements.txt                     # Dépendances du projet
└── README.md                            # Ce fichier
```

### Prérequis
- Python 3.8+
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn
- wordcloud
- joblib

### Installation
```bash
# Cloner le dépôt
git clone https://github.com/votrenomdutilisateur/ecommerce-text-classification.git
cd ecommerce-text-classification

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les ressources NLTK nécessaires
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### Utilisation
1. Téléchargez le jeu de données depuis [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification)
2. Placez le fichier CSV dans le répertoire racine du projet
3. Exécutez le notebook Jupyter :
```bash
jupyter notebook ecommerce_data_mining_project.ipynb
```

4. Pour utiliser le modèle entraîné pour des prédictions :
```python
from predict import predict_category

# Tester avec un exemple de texte
resultat, probabilite = predict_category("écouteurs bluetooth sans fil avec annulation de bruit")
print(f"Catégorie prédite : {resultat}")
print(f"Confiance : {probabilite:.2f}")
```

### Défis relevés
- Gestion du déséquilibre des classes dans les données d'entraînement
- Optimisation des hyperparamètres pour améliorer les performances
- Gestion de textes multilingues et de descriptions de produits hétérogènes
- Intégration du backend Python avec un frontend React.js (détaillé dans le rapport)

### Prochaines étapes
- Élargir à plus de catégories de produits
- Implémenter le support multilingue
- Déployer en tant que service web
- Intégrer à une plateforme e-commerce réelle

### Note
Le rapport complet du projet (PDF) est disponible en français.
