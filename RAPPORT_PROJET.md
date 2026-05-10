# Rapport de projet

## Titre
Maintenance prédictive dans les installations industrielles à partir du SECOM Manufacturing Dataset

## Contexte
La maintenance prédictive vise à anticiper les défaillances d’équipements industriels avant qu’elles ne provoquent une panne ou un arrêt de production. Dans ce projet, l’objectif est de construire un modèle de classification capable de prédire l’état de défaillance d’une installation à partir des données du dataset SECOM, qui contient des mesures issues de capteurs industriels.

## Problématique
Le jeu de données SECOM présente deux difficultés principales :
- un fort déséquilibre entre les classes,
- de nombreuses valeurs manquantes dans les variables de procédé.

Le défi consiste donc à mettre en place une chaîne de traitement robuste, puis à entraîner un modèle capable d’identifier les pannes avec une précision acceptable.

## Objectif
L’objectif du projet est double :
- concevoir un pipeline de prétraitement adapté aux données tabulaires industrielles,
- comparer des approches de classification et retenir une architecture offrant une bonne capacité de détection des pannes.

## Données
Le fichier utilisé est `uci-secom.csv`.

Caractéristiques principales du jeu de données :
- 1 567 observations,
- 592 colonnes au total,
- une variable cible `Pass/Fail`,
- une colonne temporelle `Time`,
- une forte proportion de valeurs manquantes sur plusieurs capteurs.

La classe panne correspond à la valeur positive du problème de classification.

## Méthodologie

### Prétraitement
Le pipeline appliqué est le suivant :
- lecture du CSV,
- transformation de `Time` en variables cycliques (`hour`, `dayofweek`, `month`),
- séparation des variables explicatives et de la cible,
- imputation médiane des valeurs manquantes,
- ajout d’indicateurs de valeurs manquantes,
- standardisation des variables numériques.

### Modélisation
Une phase de comparaison de modèles tabulaires a montré que `HistGradientBoostingClassifier` était l’architecture la plus intéressante pour ce problème.

Le choix final repose donc sur :
- `HistGradientBoostingClassifier`,
- une validation séparée,
- une sélection du seuil de décision orientée vers la précision sur la classe panne, avec un rappel minimal en validation.

### Justification du choix
Ce type de modèle est particulièrement adapté aux données tabulaires hétérogènes et aux relations non linéaires entre capteurs. Il offre aussi un bon compromis entre performance, robustesse et simplicité d’intégration.

## Protocole d’évaluation
Les données ont été séparées de manière stratifiée en :
- 65 % pour l’entraînement,
- 17,5 % pour la validation,
- 17,5 % pour le test.

Les métriques utilisées sont :
- précision,
- rappel,
- F1-score,
- ROC-AUC,
- PR-AUC.

## Résultats
Sur le split retenu, le modèle final obtient pour la classe panne :
- précision : **28,1 %**,
- rappel : **42,9 %**,
- F1-score : **33,96 %**.

Les autres métriques globales observées sont :
- accuracy : **88,85 %**,
- ROC-AUC : **0,7552**,
- PR-AUC : **0,2081**.

## Interprétation
La précision de la classe panne dépasse le seuil demandé de 25 %, ce qui signifie qu’une part significative des alertes générées correspond bien à de vraies défaillances. Le rappel reste également correct, ce qui est important dans un contexte industriel où manquer une panne peut coûter cher.

Le modèle n’est toutefois pas parfait :
- le jeu de données reste très déséquilibré,
- la PR-AUC montre que la détection de la classe panne demeure difficile,
- une amélioration supplémentaire pourrait passer par un meilleur réglage des seuils, une optimisation hyperparamétrique plus poussée ou des modèles tabulaires avancés.

## Implémentation
Le script principal se trouve dans [secom_predictive_maintenance.py](secom_predictive_maintenance.py).

Il réalise automatiquement :
- le chargement des données,
- le prétraitement,
- l’entraînement du modèle,
- l’évaluation sur test,
- l’enregistrement des artefacts.

Les sorties produites sont :
- `artifacts/secom_hgb_artifact.joblib`,
- `artifacts/secom_metrics.json`,
- `artifacts/secom_confusion_matrix.png`.

## Exécution
Pour lancer l’expérience :

```bash
c:/Users/oussa/OneDrive/Desktop/deepleaeninf/.venv/Scripts/python.exe secom_predictive_maintenance.py
```

## Conclusion
Ce projet montre qu’un pipeline de maintenance prédictive basé sur des données tabulaires industrielles peut être construit efficacement à partir du SECOM Manufacturing Dataset. L’approche retenue permet d’atteindre une précision supérieure à 25 % sur la classe panne, tout en conservant un rappel utile pour un cas d’usage industriel.

## Perspectives
Pour aller plus loin, on peut envisager :
- un réglage plus fin du seuil de décision,
- une recherche d’hyperparamètres plus large,
- des modèles spécialisés pour données tabulaires,
- une comparaison avec XGBoost ou LightGBM si ces bibliothèques sont disponibles.