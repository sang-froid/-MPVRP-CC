# MPVRP-CC Solver with OR-Tools

Solveur pour le Multi-Product Vehicle Routing Problem with Changeover Cost utilisant Google OR-Tools.

## Installation
```bash
pip install -r requirements.txt
```

## Utilisation

### Résoudre une instance unique
```bash
python solver.py instances/small/MPVRP_S_001_s9_d1_p2.dat solutions/small/Sol_MPVRP_S_001.dat 300
```

Ou avec le script:
```bash
run_solver.bat instances\small\MPVRP_S_001_s9_d1_p2.dat 300
```

### Tester toutes les instances
```bash
test_all_small.bat
test_all_medium.bat
test_all_large.bat
```

## Architecture

### Stratégie de résolution

1. **Décomposition par produit**: Résoudre un VRP pour chaque produit indépendamment
2. **Consolidation**: Assigner les routes aux véhicules disponibles
3. **Optimisation**: Minimiser les changeovers en groupant les produits

### Structure du code
```
src/
├── data_loader.py      - Chargement des instances
├── model.py            - Modèles de données
├── ortools_solver.py   - Solveur OR-Tools
├── validator.py        - Validation des solutions
└── solution_writer.py  - Écriture au format .dat
```

## Algorithme

OR-Tools utilise:
- **First Solution Strategy**: PATH_CHEAPEST_ARC
- **Metaheuristic**: Guided Local Search
- **Contraintes**: Capacité des véhicules, satisfaction de la demande

## Auteur

[Votre nom]