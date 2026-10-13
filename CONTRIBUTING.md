# Guide de contribution

Ce guide explique comment travailler à plusieurs sur le repo sans se gêner.

## Règle d'or

**On ne pousse jamais directement sur `main`.** Chaque modification passe par une branche et une pull request relue par au moins un autre membre.

## Démarrer

```bash
git clone <url-du-repo>
cd <nom-du-repo>
git checkout main
git pull
```

## Travailler sur une tâche

1. Mets à jour `main` :
   ```bash
   git checkout main
   git pull
   ```
2. Crée une branche pour ta tâche :
   ```bash
   git checkout -b feature/nom-de-la-tache
   ```
   Exemples : `feature/decoupage-documents`, `fix/erreur-encodage`, `docs/readme`.
3. Fais des commits petits et clairs :
   ```bash
   git add .
   git commit -m "Ajoute le découpage des documents en passages"
   ```
4. Envoie ta branche :
   ```bash
   git push -u origin feature/nom-de-la-tache
   ```
5. Sur GitHub, ouvre une **pull request** vers `main`, décris ce que tu as fait et demande une relecture.
6. Après validation, fusionne, puis supprime la branche.

## Bonnes pratiques

- Une branche = une tâche. Évite les branches qui durent des semaines.
- Fais `git pull` sur `main` avant de commencer et régulièrement pendant le travail.
- Ne commite jamais de clés d'API, mots de passe ou fichiers `.env`.
- Ne commite pas de gros fichiers de données : mets-les dans `data/raw/` (ignoré par git) et partage-les via le drive du groupe.
- Écris des messages de commit qui disent ce qui change, pas « modif » ou « test ».

## En cas de conflit

Si Git signale un conflit au moment de fusionner `main` dans ta branche :

```bash
git checkout feature/ma-branche
git merge main
# ouvre les fichiers en conflit, garde le bon contenu, supprime les marqueurs <<<<<<< ======= >>>>>>>
git add .
git commit
```

En cas de doute, demande au groupe avant de forcer quoi que ce soit (`git push --force` est à éviter).

## Répartition du travail

Utilise l'onglet **Issues** de GitHub : une issue par tâche, assignée à une personne. Cela permet à tout le monde de voir qui fait quoi.
