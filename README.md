# IA - TD (Professeur : Pierre Andry)

---

## TD1 - Question 1

<img src="./Part-1/generatedPlots/test.png"/>

### Remarques

- Le modèle entrainé commet très peu d'erreurs avec un bruit < 30%
- L'utilisation de ce modèle n'est donc plus fiable à partir d'un buitage > 30%

## TD1 - Question 2 : Widrow-Hoff

<img src="./Part-2-Widrow-Hoff/generatedPlots/LearningCurve.png"/>

### Remarques

- Evolution de l'erreur du modèle au cours de son apprentissage
- A partir de 200 itérations, le modèle est quasi parfait pour être utilisé
- Le modèle est arrêté quand son err < 10⁻⁶, sans cette étape d'arrêt, le modèle partirai dans une boucle infinie

<img src="./Part-2-Widrow-Hoff/generatedPlots/Widrow-Hoff.png"/>

### Remarques

- Modèle entrainé beaucoup plus robuste et fiable que le précédent
- Jusqu'à 20% de bruitage, le modèle ne commet aucune erreur
