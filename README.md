# Maintenance prédictive sur le jeu SECOM

Le rapport complet du projet est disponible dans [RAPPORT_PROJET.md](RAPPORT_PROJET.md).

Pour exécuter l'expérience principale :

```bash
c:/Users/oussa/OneDrive/Desktop/deepleaeninf/.venv/Scripts/python.exe secom_predictive_maintenance.py
```

Les artefacts générés sont enregistrés dans `artifacts/`.

## Publication GitHub

Le dépôt est préparé pour une publication sur GitHub Pages. Après avoir envoyé le projet sur un dépôt public GitHub, le workflow `.github/workflows/pages.yml` génère une page web statique avec :

- le notebook rendu en HTML,
- le rapport du projet en HTML,
- une page d'accueil de navigation.

Pour une exécution locale reproductible, installez les dépendances avec :

```bash
pip install -r requirements.txt
```

Puis lancez le script principal ou exportez le notebook en HTML si besoin :

```bash
jupyter nbconvert --to html secom_predictive_maintenance.ipynb
```