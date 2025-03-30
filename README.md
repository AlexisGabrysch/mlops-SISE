# 🌸 Iris Predictor ML App

![](https://img.shields.io/badge/Python-3.9-blue)
![](https://img.shields.io/badge/Streamlit-1.x-ff4b4b)
![](https://img.shields.io/badge/FastAPI-0.x-009688)
![](https://img.shields.io/badge/Docker-Compose-2496ed)
![](https://img.shields.io/badge/Machine%20Learning-scikit--learn-f7931e)

Une application moderne de Machine Learning pour prédire l'espèce d'iris à partir de mesures de fleurs. Ce projet démontre l'intégration de FastAPI, Streamlit, MongoDB et Docker pour créer une application ML complète avec entrainement et déploiement.

![Iris Predictor App Preview](https://github.com/alexisgabrysch/mlops-SISE/blob/main/Image/iris.gif)

## 🚀 Fonctionnalités

- **Prédiction d'espèces d'iris** basée sur les mesures de sépales et pétales
- **Interface utilisateur moderne** avec visualisations interactives
- **Base de données MongoDB** pour le stockage persistent des données
- **Entrainement de modèles personnalisés** avec plusieurs algorithmes de ML
- **Architecture microservices** utilisant Docker et Docker Compose
- **Backend API RESTful** construit avec FastAPI

## 📋 Prérequis

- [Docker](https://www.docker.com/get-started) et [Docker Compose](https://docs.docker.com/compose/install/)
- Connexion Internet (pour accéder aux images Docker et aux dépendances)

## 🔧 Installation et démarrage

### 1. Cloner le dépôt

```bash
git clone https://github.com/alexisgabrysch/mlops-SISE.git
cd mlops-SISE
```

### 2. Lancer l'application

```bash
docker-compose up --build
```

L'application sera accessible aux adresses suivantes :
- Interface utilisateur Streamlit : http://localhost:8501
- API FastAPI : http://localhost:8000
- Documentation API : http://localhost:8000/docs

## 💻 Utilisation

### 1. Onglet Saisie de Données
- Entrez des noms de fruits dans le champ de saisie
- Les données sont stockées dans la base de données MongoDB
- Les ajouts récents sont affichés en temps réel

### 2. Onglet Visualisation de Données
- Consultez les données dans un format tabulaire
- Explorez les graphiques interactifs montrant la distribution des fruits
- Visualisez les proportions avec un graphique circulaire interactif

### 3. Onglet Prédiction
- Ajustez les curseurs pour spécifier les dimensions de la fleur d'iris
- Le graphique radar compare vos mesures aux moyennes des espèces
- Cliquez sur "Predict Species" pour obtenir la prédiction
- L'image et les informations sur l'espèce prédite sont affichées

### 4. Onglet Entrainement du Modèle
- Consultez les informations sur le modèle actuellement utilisé
- Sélectionnez un nouvel algorithme de ML à entrainer
- Personnalisez les hyperparamètres du modèle
- Visualisez les résultats d'entrainement avec métriques et matrices de confusion

## 🔌 API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/add/{fruit}` | GET | Ajoute le fruit spécifié à la base de données |
| `/list` | GET | Récupère la liste de tous les fruits |
| `/predict` | POST | Envoie les données pour prédiction d'espèce d'iris |
| `/model-info` | GET | Récupère les informations sur le modèle actuel |
| `/train` | POST | Entraine un nouveau modèle avec les paramètres spécifiés |

## 🧪 Technologies utilisées

### Frontend
- **Streamlit** - Interface utilisateur interactive
- **Plotly** - Visualisations de données interactives
- **Pandas** - Manipulation et analyse de données

### Backend
- **FastAPI** - Framework API REST haute performance
- **scikit-learn** - Algorithmes de machine learning
- **MongoDB** - Base de données NoSQL
- **Joblib** - Persistence des modèles ML

### Infrastructure
- **Docker** - Conteneurisation
- **Docker Compose** - Orchestration multi-conteneurs

## 📁 Structure du projet

```
mlops-SISE/
├── client/               # Application frontend Streamlit
│   ├── app.py            # Code de l'interface utilisateur
│   ├── Dockerfile        # Configuration du conteneur frontend
│   └── requirements.txt  # Dépendances Python pour le frontend
├── server/               # Serveur backend FastAPI
│   ├── app.py            # API endpoints
│   ├── train.py          # Logique d'entrainement des modèles
│   ├── model.pkl         # Modèle ML sauvegardé
│   ├── Dockerfile        # Configuration du conteneur backend
│   └── requirements.txt  # Dépendances Python pour le backend
├── docker-compose.yml    # Configuration multi-conteneurs
└── README.md             # Documentation du projet
```

## 👨‍💻 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou soumettre une pull request pour toute amélioration ou correction de bugs.
