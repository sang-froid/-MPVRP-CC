"""
MPVRP-CC Solver – Version OR-Tools CP-SAT avec heuristique initiale et visualisation
Solveur exact par programmation par contraintes avec solution de départ
"""

import sys
import time
import platform
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from ortools.sat.python import cp_model


# =============================================================================
# STRUCTURES DE DONNÉES
# =============================================================================

@dataclass
class Instance:
    """Représente une instance complète du problème MPVRP-CC"""
    name: str                      # Nom de l'instance
    P: int                         # Nombre de produits
    D: int                         # Nombre de dépôts
    G: int                         # Nombre de garages
    S: int                         # Nombre de stations
    V: int                         # Nombre de véhicules
    transition_costs: np.ndarray   # Coûts de changement de produit
    vehicles: list                 # Liste des véhicules (id, capacité, garage, produit_initial)
    depots: list                   # Liste des dépôts
    garages: list                  # Liste des garages
    stations: list                 # Liste des stations


# =============================================================================
# CHARGEMENT D'INSTANCE (FORMAT STANDARD)
# =============================================================================

def load_instance(path: str) -> Instance:
    """
    Charge une instance depuis un fichier .dat
    
    Format attendu :
    Ligne 1: P D G S V (entiers)
    Lignes suivantes: matrice des coûts de transition (P x P)
    Puis véhicules, dépôts, garages, stations
    """
    with open(path) as f:
        # Lire toutes les lignes non vides et sans commentaires
        raw = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # Trouver la première ligne avec uniquement des nombres
    i = 0
    while True:
        parts = raw[i].split()
        # Vérifier si tous les éléments sont des nombres
        if all(p.replace('.', '').replace('-', '').isdigit() for p in parts):
            break
        i += 1

    # Lire les dimensions de l'instance
    P, D, G, S, V = map(int, raw[i].split())
    i += 1

    # Charger la matrice des coûts de transition entre produits
    transition_costs = np.zeros((P, P), dtype=int)
    for p in range(P):
        # Convertir les flottants en entiers (comme demandé)
        transition_costs[p] = list(map(int, map(float, raw[i].split())))
        i += 1

    # Charger les véhicules
    vehicles = []
    for _ in range(V):
        vid, cap, gid, prod = raw[i].split()
        vehicles.append((int(vid), int(float(cap)), int(gid), int(float(prod))))
        i += 1

    # Charger les dépôts
    depots = []
    for _ in range(D):
        parts = list(map(float, raw[i].split()))
        # Stocks par produit (à partir de la 4ème colonne)
        stocks = {p+1: int(parts[3+p]) for p in range(P)}
        depots.append((int(parts[0]), parts[1], parts[2], stocks))
        i += 1

    # Charger les garages
    garages = []
    for _ in range(G):
        gid, x, y = map(float, raw[i].split())
        garages.append((int(gid), x, y))
        i += 1

    # Charger les stations
    stations = []
    for _ in range(S):
        parts = list(map(float, raw[i].split()))
        # Extraire les demandes par produit (colonnes 4+)
        demands = {}
        for p in range(P):
            demand_val = parts[3 + p]
            if demand_val > 0:  # Ne conserver que les demandes positives
                demands[p+1] = int(demand_val)
        stations.append((int(parts[0]), parts[1], parts[2], demands))
        i += 1

    return Instance(
        name=Path(path).stem,
        P=P, D=D, G=G, S=S, V=V,
        transition_costs=transition_costs,
        vehicles=vehicles,
        depots=depots,
        garages=garages,
        stations=stations
    )


# =============================================================================
# HEURISTIQUE DE CONSTRUCTION INITIALE
# =============================================================================

def construire_solution_heuristique(instance: Instance) -> Tuple[Dict, float]:
    """
    Construit une solution initiale faisable (heuristique).
    
    Principe simple :
    1. Chaque véhicule garde son produit initial
    2. Il dessert les stations les plus proches qui demandent ce produit
    3. Remplit sa capacité au maximum possible
    
    Retourne un dictionnaire avec les valeurs des variables et le coût total.
    """
    print("\n🔧 Construction de la solution heuristique...")
    
    # Initialiser les structures pour stocker la solution
    solution = {}
    V, S, P = instance.V, instance.S, instance.P
    
    # 1. Préparer les données utiles pour l'heuristique
    
    # Pour chaque station, avoir la liste des produits demandés
    demandes_par_station = []
    for s in range(S):
        station_info = instance.stations[s]
        demandes = station_info[3]  # Dictionnaire {produit: quantité}
        demandes_par_station.append(demandes)
    
    # Pour chaque véhicule
    for v in range(V):
        vid, capacite, gid, produit_initial = instance.vehicles[v]
        
        # Coordonnées du garage de ce véhicule
        garage_x, garage_y = instance.garages[gid-1][1], instance.garages[gid-1][2]
        
        print(f"  Véhicule {vid} (Produit {produit_initial}, Capacité {capacite}):")
        
        # 2. Lister les stations qui demandent le produit initial du véhicule
        stations_candidates = []
        for s in range(S):
            if produit_initial in demandes_par_station[s]:
                station_info = instance.stations[s]
                sid, sx, sy, _ = station_info
                
                # Calculer la distance entre garage et station
                distance = math.hypot(sx - garage_x, sy - garage_y)
                demande = demandes_par_station[s][produit_initial]
                
                stations_candidates.append((s, sid, distance, demande))
        
        # 3. Trier par distance (plus proche d'abord)
        stations_candidates.sort(key=lambda x: x[2])
        
        # 4. Assigner les livraisons dans l'ordre de proximité
        capacite_restante = capacite
        stations_assignees = []
        
        for s_idx, sid, distance, demande in stations_candidates:
            if capacite_restante <= 0:
                break
            
            # Calculer combien on peut livrer
            quantite_a_livrer = min(demande, capacite_restante)
            
            if quantite_a_livrer > 0:
                # Stocker dans la solution
                solution[(v, s_idx, produit_initial-1)] = quantite_a_livrer
                
                # Mettre à jour la capacité
                capacite_restante -= quantite_a_livrer
                stations_assignees.append((sid, quantite_a_livrer))
        
        # Afficher le résultat pour ce véhicule
        if stations_assignees:
            stations_str = ", ".join([f"St{sid}({q})" for sid, q in stations_assignees])
            print(f"    → Dessert {len(stations_assignees)} stations: {stations_str}")
            print(f"    → Capacité utilisée: {capacite - capacite_restante}/{capacite}")
        else:
            print(f"    → Aucune station assignée (pas de demande pour P{produit_initial})")
    
    # 5. Calculer le coût de cette solution heuristique
    cout_total = 0
    for (v, s, p), quantite in solution.items():
        if quantite > 0:
            # Coût de distance
            vid, _, gid, _ = instance.vehicles[v]
            garage_x, garage_y = instance.garages[gid-1][1], instance.garages[gid-1][2]
            station_x, station_y = instance.stations[s][1], instance.stations[s][2]
            distance = int(math.hypot(station_x - garage_x, station_y - garage_y) * 100)
            
            # Coût de changement de produit (si différent du produit initial)
            produit_initial = instance.vehicles[v][3]
            if produit_initial != 0 and produit_initial != p+1:
                cout_changement = instance.transition_costs[produit_initial-1][p]
            else:
                cout_changement = 0
            
            cout_total += distance + cout_changement
    
    print(f"✓ Solution heuristique construite (coût estimé: {cout_total:.2f})")
    return solution, cout_total


# =============================================================================
# VISUALISATION DES RÉSULTATS
# =============================================================================

def visualiser_solution(instance: Instance, routes: List[Dict], output_file: Optional[str] = None):
    """
    Crée une visualisation graphique des routes et des stations.
    
    Cette fonction génère :
    1. Une carte avec les stations, garages et dépôts
    2. Les routes colorées par véhicule
    3. Les quantités livrées affichées
    4. Une légende complète
    """
    print("\n🎨 Création de la visualisation...")
    
    # Créer la figure avec deux subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'MPVRP-CC Solution - {instance.name}', fontsize=16, fontweight='bold')
    
    # =========================================================================
    # SUBPLOT 1 : CARTE GÉOGRAPHIQUE DES ROUTES
    # =========================================================================
    
    # Définir une palette de couleurs pour les véhicules
    couleurs_vehicules = plt.cm.Set3(np.linspace(0, 1, instance.V))
    
    # 1. Tracer les garages
    for gid, x, y in instance.garages:
        ax1.scatter(x, y, s=200, c='red', marker='s', label='Garage' if gid == 1 else "", 
                   edgecolors='black', linewidth=2, zorder=5)
        ax1.text(x, y, f'G{gid}', fontsize=10, ha='center', va='center', 
                fontweight='bold', color='white')
    
    # 2. Tracer les dépôts
    for did, x, y, _ in instance.depots:
        ax1.scatter(x, y, s=150, c='orange', marker='D', label='Dépôt' if did == 1 else "",
                   edgecolors='black', linewidth=2, zorder=5)
        ax1.text(x, y, f'D{did}', fontsize=10, ha='center', va='center',
                fontweight='bold', color='black')
    
    # 3. Tracer les stations (avec leur demande totale)
    for sid, x, y, demands in instance.stations:
        demande_totale = sum(demands.values())
        # Taille proportionnelle à la demande totale
        taille = 50 + min(demande_totale / 100, 100)
        
        ax1.scatter(x, y, s=taille, c='lightblue', marker='o', 
                   edgecolors='black', linewidth=1, zorder=3)
        
        # Afficher l'ID de la station
        ax1.text(x, y, f'{sid}', fontsize=8, ha='center', va='center',
                fontweight='bold')
        
        # Afficher les demandes par produit
        texte_demande = ""
        for p, q in demands.items():
            texte_demande += f"P{p}:{q}\n"
        if texte_demande:
            ax1.text(x, y + 2, texte_demande.strip(), fontsize=6, 
                    ha='center', va='bottom', color='darkblue')
    
    # 4. Tracer les routes
    for route in routes:
        v_id = route['vehicle']
        couleur = couleurs_vehicules[v_id-1]
        
        # Coordonnées du garage de ce véhicule
        gid = route['garage']
        garage_coords = None
        for gar in instance.garages:
            if gar[0] == gid:
                garage_coords = (gar[1], gar[2])
                break
        
        if garage_coords:
            gx, gy = garage_coords
            
            # Tracer la ligne du garage à chaque station
            for station_id, livraisons in route['stations'].items():
                # Trouver les coordonnées de la station
                station_coords = None
                for sta in instance.stations:
                    if sta[0] == station_id:
                        station_coords = (sta[1], sta[2])
                        break
                
                if station_coords:
                    sx, sy = station_coords
                    
                    # Calculer la quantité totale livrée à cette station
                    quantite_totale = sum(q for _, q in livraisons)
                    
                    # Tracer la ligne (plus épaisse pour les grandes quantités)
                    epaisseur = 1 + min(quantite_totale / 1000, 3)
                    ax1.plot([gx, sx], [gy, sy], color=couleur, 
                            linewidth=epaisseur, alpha=0.7, zorder=2)
                    
                    # Ajouter une flèche pour indiquer la direction
                    ax1.annotate('', xy=(sx, sy), xytext=(gx, gy),
                                arrowprops=dict(arrowstyle='->', color=couleur, 
                                              lw=epaisseur*0.8, alpha=0.5))
                    
                    # Afficher les quantités livrées le long de la ligne
                    mid_x, mid_y = (gx + sx) / 2, (gy + sy) / 2
                    texte_livraison = ""
                    for p, q in livraisons:
                        texte_livraison += f"P{p}:{q}\n"
                    
                    ax1.text(mid_x, mid_y, texte_livraison.strip(), 
                            fontsize=6, ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.3", 
                                     facecolor='white', alpha=0.8,
                                     edgecolor=couleur))
    
    # Configuration de la carte
    ax1.set_title('Carte des routes et stations', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Coordonnée X', fontsize=12)
    ax1.set_ylabel('Coordonnée Y', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9)
    
    # Ajuster les limites pour avoir un peu de marge
    all_x = [p[1] for p in instance.stations] + [g[1] for g in instance.garages]
    all_y = [p[2] for p in instance.stations] + [g[2] for g in instance.garages]
    marge = max(max(all_x) - min(all_x), max(all_y) - min(all_y)) * 0.1
    ax1.set_xlim(min(all_x) - marge, max(all_x) + marge)
    ax1.set_ylim(min(all_y) - marge, max(all_y) + marge)
    
    # =========================================================================
    # SUBPLOT 2 : STATISTIQUES ET INFORMATIONS
    # =========================================================================
    
    # Créer un tableau texte avec les informations
    ax2.axis('off')
    ax2.set_title('Statistiques de la solution', fontsize=14, fontweight='bold', pad=20)
    
    # Préparer les données pour le tableau
    donnees_tableau = []
    
    # 1. Informations générales
    donnees_tableau.append(["INFORMATIONS GÉNÉRALES", ""])
    donnees_tableau.append(["Instance:", instance.name])
    donnees_tableau.append(["Stations:", f"{instance.S}"])
    donnees_tableau.append(["Véhicules:", f"{instance.V}"])
    donnees_tableau.append(["Produits:", f"{instance.P}"])
    donnees_tableau.append(["", ""])
    
    # 2. Statistiques des routes
    donnees_tableau.append(["STATISTIQUES DES ROUTES", ""])
    donnees_tableau.append(["Routes créées:", f"{len(routes)}"])
    
    if routes:
        capacite_utilisee_totale = sum(r['capacity_used'] for r in routes)
        capacite_totale = sum(r['total_capacity'] for r in routes)
        taux_utilisation = (capacite_utilisee_totale / capacite_totale * 100) if capacite_totale > 0 else 0
        
        donnees_tableau.append(["Capacité utilisée:", f"{capacite_utilisee_totale}/{capacite_totale} ({taux_utilisation:.1f}%)"])
        donnees_tableau.append(["", ""])
    
    # 3. Détail par véhicule
    donnees_tableau.append(["DÉTAIL PAR VÉHICULE", ""])
    
    for route in routes:
        v_id = route['vehicle']
        cap_util = route['capacity_used']
        cap_tot = route['total_capacity']
        taux = (cap_util / cap_tot * 100) if cap_tot > 0 else 0
        stations_servies = len(route['stations'])
        
        donnees_tableau.append([
            f"Véhicule {v_id}:",
            f"{cap_util}/{cap_tot} ({taux:.1f}%), {stations_servies} stations"
        ])
    
    # 4. Détail des livraisons par produit
    donnees_tableau.append(["", ""])
    donnees_tableau.append(["LIVRAISONS PAR PRODUIT", ""])
    
    # Calculer les totaux par produit
    livraisons_par_produit = {}
    for route in routes:
        for station_id, livraisons in route['stations'].items():
            for p, q in livraisons:
                if p not in livraisons_par_produit:
                    livraisons_par_produit[p] = 0
                livraisons_par_produit[p] += q
    
    for p in sorted(livraisons_par_produit.keys()):
        donnees_tableau.append([f"Produit {p}:", f"{livraisons_par_produit[p]} unités"])
    
    # Créer le tableau
    tableau = ax2.table(
        cellText=donnees_tableau,
        cellLoc='left',
        loc='center',
        colWidths=[0.4, 0.6]
    )
    
    # Styliser le tableau
    tableau.auto_set_font_size(False)
    tableau.set_fontsize(9)
    tableau.scale(1, 1.5)
    
    # Colorer les en-têtes
    for i in range(len(donnees_tableau)):
        for j in range(2):
            cell = tableau[i, j]
            if i == 0 or "INFORMATIONS" in donnees_tableau[i][0] or "STATISTIQUES" in donnees_tableau[i][0] or "DÉTAIL" in donnees_tableau[i][0] or "LIVRAISONS" in donnees_tableau[i][0]:
                cell.set_facecolor('#4A90E2')
                cell.set_text_props(weight='bold', color='white')
            elif i % 2 == 0:
                cell.set_facecolor('#F0F0F0')
    
    # =========================================================================
    # SAUVEGARDE ET AFFICHAGE
    # =========================================================================
    
    plt.tight_layout()
    
    if output_file:
        # Sauvegarder l'image
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Visualisation sauvegardée dans: {output_file}")
    
    # Afficher le graphique
    plt.show()
    
    print("✓ Visualisation créée avec succès!")


# =============================================================================
# SOLVEUR PRINCIPAL AVEC OR-TOOLS CP-SAT
# =============================================================================

def solve_with_ortools(instance: Instance, temps_limite: float = 30.0):
    """
    Résout le problème MPVRP-CC avec OR-Tools CP-SAT.
    
    Utilise une solution heuristique comme point de départ (Hint).
    """
    # Début du chronomètre
    debut_modelisation = time.time()
    
    # Créer le modèle CP-SAT
    model = cp_model.CpModel()
    
    # Dimensions du problème
    V, S, P = instance.V, instance.S, instance.P
    
    print(f"\n🧮 Modélisation CP-SAT en cours...")
    print(f"  Variables: {V} véhicules × {S} stations × {P} produits = {V*S*P} variables")

    # =========================================================================
    # DÉCLARATION DES VARIABLES
    # =========================================================================
    
    # 1. Variables principales : quantité livrée par véhicule v à station s de produit p
    x = {}
    for v in range(V):
        for s in range(S):
            for p in range(P):
                # Limite supérieure = min(capacité véhicule, demande station)
                demande_max = instance.stations[s][3].get(p+1, 0)
                capacite_vehicule = instance.vehicles[v][1]
                limite_superieure = min(demande_max, capacite_vehicule)
                
                # Créer la variable entière
                x[v, s, p] = model.NewIntVar(
                    0, 
                    limite_superieure, 
                    f"livraison_v{v}_s{s}_p{p}"
                )

    # 2. Variables binaires de visite (si un véhicule visite une station)
    visite = {}
    for v in range(V):
        for s in range(S):
            visite[v, s] = model.NewBoolVar(f"visite_v{v}_s{s}")

    # =========================================================================
    # CONTRAINTES DU PROBLÈME
    # =========================================================================
    
    print("  Ajout des contraintes...")

    # 1. CONTRAINTE : Satisfaction complète de la demande
    # Chaque demande (station, produit) doit être entièrement satisfaite
    contraintes_demande = 0
    for s_idx, (s_id, _, _, demands) in enumerate(instance.stations):
        for p, demande_totale in demands.items():
            if demande_totale > 0:  # Seulement pour les produits demandés
                # La somme des livraisons de tous les véhicules = demande totale
                somme_livraisons = sum(x[v, s_idx, p-1] for v in range(V))
                model.Add(somme_livraisons == demande_totale)
                contraintes_demande += 1
    
    print(f"    → {contraintes_demande} contraintes de demande")

    # 2. CONTRAINTE : Lien entre livraison et visite
    # Si un véhicule livre quelque chose à une station, alors il la visite
    for v in range(V):
        for s in range(S):
            # Calculer la somme des livraisons de ce véhicule à cette station
            total_livraison = sum(x[v, s, p] for p in range(P))
            
            # Si total_livraison > 0, alors visite[v,s] = 1
            model.Add(total_livraison > 0).OnlyEnforceIf(visite[v, s])
            model.Add(total_livraison == 0).OnlyEnforceIf(visite[v, s].Not())
    
    print(f"    → {V*S} contraintes de lien livraison-visite")

    # 3. CONTRAINTE : Capacité des véhicules
    # La somme des livraisons d'un véhicule ne dépasse pas sa capacité
    for v in range(V):
        vid, capacite, _, _ = instance.vehicles[v]
        total_livraison_vehicule = sum(
            x[v, s, p] for s in range(S) for p in range(P)
        )
        model.Add(total_livraison_vehicule <= capacite)
    
    print(f"    → {V} contraintes de capacité")

    # =========================================================================
    # FONCTION OBJECTIF À MINIMISER
    # =========================================================================
    
    print("  Construction de la fonction objectif...")
    
    # Initialiser les termes de l'objectif
    termes_objectif = []
    
    # 1. Coût de distance (garage → station)
    for v in range(V):
        _, _, gid, _ = instance.vehicles[v]
        # Coordonnées du garage du véhicule
        garage_x, garage_y = instance.garages[gid-1][1], instance.garages[gid-1][2]
        
        for s_idx, (s_id, sx, sy, _) in enumerate(instance.stations):
            # Calculer la distance euclidienne (convertie en entier)
            distance = int(math.hypot(sx - garage_x, sy - garage_y) * 100)
            
            # Ajouter au coût si la station est visitée
            termes_objectif.append(distance * visite[v, s_idx])
    
    # 2. Coût de changement de produit
    for v in range(V):
        _, _, _, produit_initial = instance.vehicles[v]
        
        if produit_initial != 0:  # Si le véhicule a un produit initial défini
            for s_idx in range(S):
                for p in range(P):
                    if produit_initial != p+1:  # Si différent du produit initial
                        cout_changement = instance.transition_costs[produit_initial-1][p]
                        
                        # Ajouter le coût si le véhicule livre ce produit à cette station
                        termes_objectif.append(cout_changement * visite[v, s_idx])
    
    # 3. Coût fixe par visite (pour encourager la consolidation)
    coefficient_consolidation = 50  # Poids pour réduire le nombre de visites
    for v in range(V):
        for s in range(S):
            termes_objectif.append(coefficient_consolidation * visite[v, s])
    
    # Définir la fonction objectif à minimiser
    model.Minimize(sum(termes_objectif))
    
    temps_modelisation = time.time() - debut_modelisation
    print(f"✓ Modélisation terminée en {temps_modelisation:.3f}s")
    print(f"  Objectif: somme de {len(termes_objectif)} termes")

    # =========================================================================
    # AJOUT DE LA SOLUTION HEURISTIQUE COMME POINT DE DÉPART (HINT)
    # =========================================================================
    
    print("\n🎯 Ajout de la solution heuristique comme point de départ...")
    
    # Construire la solution heuristique
    solution_heuristique, cout_heuristique = construire_solution_heuristique(instance)
    
    # Créer une solution partielle (Hint) pour OR-Tools
    if solution_heuristique:
        # Pour chaque variable de la solution heuristique, ajouter un Hint
        for (v, s, p), valeur in solution_heuristique.items():
            model.AddHint(x[v, s, p], valeur)
        
        # Aussi pour les variables de visite
        for v in range(V):
            for s in range(S):
                # Déterminer si le véhicule visite la station dans la solution heuristique
                visite_heuristique = any(
                    solution_heuristique.get((v, s, p), 0) > 0 
                    for p in range(P)
                )
                model.AddHint(visite[v, s], 1 if visite_heuristique else 0)
        
        print(f"✓ {len(solution_heuristique)} hints ajoutés")
        print(f"✓ Solution initiale: coût = {cout_heuristique:.2f}")
    else:
        print("⚠️ Aucune solution heuristique générée")

    # =========================================================================
    # CONFIGURATION ET RÉSOLUTION
    # =========================================================================
    
    print("\n⚙️ Configuration du solveur...")
    
    # Créer le solveur
    solver = cp_model.CpSolver()
    
    # Paramètres de configuration
    solver.parameters.max_time_in_seconds = temps_limite  # Temps maximum
    solver.parameters.num_search_workers = 4          # Utilisation multi-thread
    solver.parameters.log_search_progress = True      # Affichage des progrès
    solver.parameters.relative_gap_limit = 0.01       # Écart relatif accepté (1%)
    
    # Information sur la plateforme
    print(f"  Plateforme: {platform.system()} {platform.release()}")
    print(f"  Temps limite: {solver.parameters.max_time_in_seconds}s")
    print(f"  Workers: {solver.parameters.num_search_workers}")

    # =========================================================================
    # RÉSOLUTION
    # =========================================================================
    
    print("\n🔍 Lancement de la résolution CP-SAT...")
    print("-" * 50)
    
    debut_resolution = time.time()
    status = solver.Solve(model)
    temps_resolution = time.time() - debut_resolution
    
    print("-" * 50)
    
    # =========================================================================
    # ANALYSE DES RÉSULTATS
    # =========================================================================
    
    print(f"\n📊 Résultats de la résolution:")
    print(f"  Temps de résolution: {temps_resolution:.3f}s")
    print(f"  Temps total (modélisation + résolution): {temps_modelisation + temps_resolution:.3f}s")
    
    # Dictionnaire des statuts
    statuts = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FAISABLE",
        cp_model.INFEASIBLE: "IRRÉALISABLE",
        cp_model.MODEL_INVALID: "MODÈLE INVALIDE",
        cp_model.UNKNOWN: "INCONNU"
    }
    
    statut_lisible = statuts.get(status, "STATUT NON RECONNU")
    print(f"  Statut: {statut_lisible} ({status})")

    # =========================================================================
    # EXTRACTION DE LA SOLUTION
    # =========================================================================
    
    routes = []
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"\n✅ Solution trouvée!")
        print(f"  Valeur objectif: {solver.ObjectiveValue():.2f}")
        
        # Si on avait une solution heuristique, comparer l'amélioration
        if solution_heuristique:
            amelioration = cout_heuristique - solver.ObjectiveValue()
            if amelioration > 0:
                print(f"  Amélioration par rapport à l'heuristique: {amelioration:.2f} ({amelioration/cout_heuristique*100:.1f}%)")
            else:
                print(f"  L'heuristique était déjà optimale")
        
        # Vérifier que toutes les demandes sont satisfaites
        print("\n📋 Vérification des demandes satisfaites:")
        toutes_demandes_satisfaites = True
        
        for s_idx, (s_id, _, _, demands) in enumerate(instance.stations):
            for p, demande_totale in demands.items():
                if demande_totale > 0:
                    livraison_totale = sum(
                        solver.Value(x[v, s_idx, p-1]) for v in range(V)
                    )
                    statut = "✓" if livraison_totale == demande_totale else "✗"
                    
                    if livraison_totale != demande_totale:
                        toutes_demandes_satisfaites = False
                    
                    print(f"    {statut} Station {s_id}, Produit {p}: "
                          f"demandé {demande_totale}, livré {livraison_totale}")
        
        if toutes_demandes_satisfaites:
            print("  ✓ Toutes les demandes sont satisfaites!")
        else:
            print("  ⚠️ Certaines demandes ne sont pas entièrement satisfaites")
        
        # =====================================================================
        # CONSTRUCTION DES ROUTES POUR L'AFFICHAGE
        # =====================================================================
        
        print("\n🚚 Extraction des routes...")
        route_id = 1
        
        for v in range(V):
            vid, capacite, gid, produit_initial = instance.vehicles[v]
            
            # Collecter toutes les livraisons de ce véhicule
            livraisons_par_station = {}
            capacite_utilisee = 0
            
            for s_idx in range(S):
                station_livraisons = []
                for p in range(P):
                    quantite = solver.Value(x[v, s_idx, p])
                    if quantite > 0:
                        station_livraisons.append((p+1, quantite))
                        capacite_utilisee += quantite
                
                if station_livraisons:
                    s_id = instance.stations[s_idx][0]
                    livraisons_par_station[s_id] = station_livraisons
            
            # Si le véhicule a au moins une livraison, créer une route
            if livraisons_par_station:
                # Calculer le coût de changement de produit pour cette route
                cout_changement_route = 0
                produits_livres = set(p for livs in livraisons_par_station.values() for p, _ in livs)
                
                if produit_initial != 0:
                    for produit in produits_livres:
                        if produit != produit_initial:
                            cout_changement_route += instance.transition_costs[produit_initial-1][produit-1]
                
                # Créer l'objet route
                route = {
                    "id": route_id,
                    "vehicle": vid,
                    "garage": gid,
                    "depot": instance.depots[0][0] if instance.depots else 1,
                    "init_product": produit_initial,
                    "capacity_used": capacite_utilisee,
                    "total_capacity": capacite,
                    "changeover_cost": cout_changement_route,
                    "stations": livraisons_par_station
                }
                
                routes.append(route)
                route_id += 1
                
                # Afficher un résumé de la route
                print(f"  Route {route['id']} (Véhicule {vid}):")
                print(f"    Produit initial: {produit_initial if produit_initial != 0 else 'aucun'}")
                print(f"    Capacité: {capacite_utilisee}/{capacite} ({capacite_utilisee/capacite*100:.1f}%)")
                print(f"    Coût changement: {cout_changement_route}")
                print(f"    Stations: {len(livraisons_par_station)}")
                for s_id, livraisons in livraisons_par_station.items():
                    produits_str = ", ".join([f"P{p}={q}" for p, q in livraisons])
                    print(f"      - Station {s_id}: {produits_str}")
        
        print(f"\n✓ {len(routes)} routes extraites")
        
    else:
        print(f"\n❌ Aucune solution trouvée")
        print(f"  Statut: {statut_lisible}")
        print(f"  Suggestions:")
        print(f"    1. Vérifier que la demande totale ≤ capacité totale des véhicules")
        print(f"    2. Augmenter le temps de calcul")
        print(f"    3. Vérifier les données de l'instance")
    
    return routes, solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else float('inf')


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def main():
    """
    Fonction principale du programme.
    
    Usage: python solver_visualisation.py instance.dat solution.dat [temps_limite]
    """
    # Vérification des arguments
    if len(sys.argv) < 3:
        print("Usage: python solver_visualisation.py instance.dat solution.dat [temps_limite]")
        print("\nArguments:")
        print("  instance.dat   : Fichier d'instance au format .dat")
        print("  solution.dat   : Fichier de sortie pour la solution")
        print("  temps_limite   : (optionnel) Temps limite en secondes (défaut: 30)")
        print("\nExemples:")
        print("  python solver_visualisation.py instance.dat solution.dat")
        print("  python solver_visualisation.py instance.dat solution.dat 60")
        sys.exit(1)
    
    # Récupérer le temps limite si fourni
    temps_limite = 30.0
    if len(sys.argv) >= 4:
        try:
            temps_limite = float(sys.argv[3])
        except ValueError:
            print(f" Temps limite invalide, utilisation de la valeur par défaut: 30s")
    
    # Affichage de l'en-tête
    print("=" * 70)
    print("MPVRP-CC SOLVEUR AVEC HEURISTIQUE INITIALE ET VISUALISATION")
    print(f"OR-Tools CP-SAT - Temps limite: {temps_limite}s")
    print("=" * 70)
    
    # Début du chronomètre global
    debut_global = time.time()
    
    # =========================================================================
    # ÉTAPE 1: CHARGEMENT DE L'INSTANCE
    # =========================================================================
    
    print("\n CHARGEMENT DE L'INSTANCE")
    print("-" * 40)
    
    try:
        inst = load_instance(sys.argv[1])
        print(f"✓ Instance chargée: {inst.name}")
        print(f"  Stations: {inst.S}")
        print(f"  Produits: {inst.P}")
        print(f"  Véhicules: {inst.V}")
        print(f"  Dépôts: {inst.D}")
        print(f"  Garages: {inst.G}")
    except Exception as e:
        print(f" Erreur lors du chargement de l'instance: {e}")
        sys.exit(1)
    
    # =========================================================================
    # ÉTAPE 2: ANALYSE DE FAISABILITÉ
    # =========================================================================
    
    print("\n ANALYSE DE FAISABILITÉ")
    print("-" * 40)
    
    # Calculer la demande totale
    demande_totale = 0
    for _, _, _, demands in inst.stations:
        demande_totale += sum(demands.values())
    
    # Calculer la capacité totale
    capacite_totale = sum(v[1] for v in inst.vehicles)
    
    print(f"  Demande totale: {demande_totale} unités")
    print(f"  Capacité totale: {capacite_totale} unités")
    
    if demande_totale > capacite_totale:
        print(f"  ATTENTION: Demande > Capacité")
        print(f"  Ratio: {demande_totale/capacite_totale:.2f}")
        print(f"  Le solveur va essayer de trouver une solution, mais c'est impossible")
    else:
        print(f"  ✓ Capacité suffisante")
        print(f"  Marge: {capacite_totale - demande_totale} unités")
        print(f"  Ratio: {demande_totale/capacite_totale:.2f}")
    
    # =========================================================================
    # ÉTAPE 3: RÉSOLUTION AVEC OR-TOOLS
    # =========================================================================
    
    print("\n RÉSOLUTION AVEC OR-TOOLS CP-SAT")
    print("-" * 40)
    
    routes, cout_total = solve_with_ortools(inst, temps_limite)
    
    # =========================================================================
    # ÉTAPE 4: VISUALISATION GRAPHIQUE
    # =========================================================================
    
    if routes:
        # Générer un nom de fichier pour la visualisation
        visu_file = f"visualisation_{inst.name}.png"
        
        # Créer la visualisation
        try:
            visualiser_solution(inst, routes, visu_file)
        except Exception as e:
            print(f"\n Erreur lors de la création de la visualisation: {e}")
            print("  La visualisation nécessite matplotlib. Installez-le avec:")
            print("  pip install matplotlib")
    
    # =========================================================================
    # ÉTAPE 5: SAUVEGARDE DE LA SOLUTION
    # =========================================================================
    
    print("\n SAUVEGARDE DE LA SOLUTION")
    print("-" * 40)
    
    try:
        with open(sys.argv[2], 'w') as f:
            # Écrire l'en-tête
            f.write(f"# Solution pour l'instance: {inst.name}\n")
            f.write(f"# Générée le: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Coût total: {cout_total:.2f}\n")
            f.write(f"# Nombre de routes: {len(routes)}\n")
            f.write(f"# Temps limite utilisé: {temps_limite}s\n")
            f.write("#\n")
            f.write("# Format: RouteID VehiculeID GarageID DepotID ProduitInit Charge\n")
            f.write("#         Liste des livraisons: StationID(Produit=Quantité,...)\n")
            f.write("#\n")
            
            # Écrire chaque route
            for route in routes:
                # Ligne principale de la route
                f.write(f"{route['id']} {route['vehicle']} {route['garage']} ")
                f.write(f"{route['depot']} {route['init_product']} ")
                f.write(f"{route['capacity_used']}\n")
                
                # Ligne des livraisons
                livraisons_str = []
                for s_id, produits in route['stations'].items():
                    produits_str = ",".join([f"{p}={q}" for p, q in produits])
                    livraisons_str.append(f"{s_id}({produits_str})")
                
                f.write("  " + " ".join(livraisons_str) + "\n")
            
            # Écrire un résumé
            f.write("#\n")
            f.write("# RÉSUMÉ\n")
            f.write(f"# Demande totale: {demande_totale}\n")
            
            if routes:
                capacite_utilisee_totale = sum(r['capacity_used'] for r in routes)
                stations_servies = sum(len(r['stations']) for r in routes)
                f.write(f"# Capacité utilisée: {capacite_utilisee_totale}\n")
                f.write(f"# Nombre de stations servies: {stations_servies}\n")
                f.write(f"# Taux d'utilisation capacité: {capacite_utilisee_totale/capacite_totale*100:.1f}%\n")
        
        print(f"✓ Solution sauvegardée dans: {sys.argv[2]}")
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")
    
    # =========================================================================
    # ÉTAPE 6: RÉCAPITULATIF FINAL
    # =========================================================================
    
    temps_total = time.time() - debut_global
    
    print("\n" + "=" * 70)
    print("RÉCAPITULATIF FINAL")
    print("=" * 70)
    
    print(f" Temps total d'exécution: {temps_total:.3f}s")
    print(f" Coût total de la solution: {cout_total:.2f}")
    print(f" Nombre de routes générées: {len(routes)}")
    
    if routes:
        # Statistiques détaillées
        capacite_utilisee_totale = sum(r['capacity_used'] for r in routes)
        stations_servies = sum(len(r['stations']) for r in routes)
        
        print(f" Capacité utilisée: {capacite_utilisee_totale}/{capacite_totale} "
              f"({capacite_utilisee_totale/capacite_totale*100:.1f}%)")
        print(f" Stations servies: {stations_servies}/{inst.S} "
              f"({stations_servies/inst.S*100:.1f}%)")
        print(f" Demande satisfaite: {capacite_utilisee_totale}/{demande_totale} "
              f"({capacite_utilisee_totale/demande_totale*100:.1f}%)")
        
        # Répartition par véhicule
        print("\n Répartition par véhicule:")
        for route in routes:
            taux_utilisation = route['capacity_used'] / route['total_capacity'] * 100
            print(f"  Véhicule {route['vehicle']}: {route['capacity_used']}/"
                  f"{route['total_capacity']} ({taux_utilisation:.1f}%)")
    
    print("\n" + "=" * 70)
    print("FIN DE L'EXÉCUTION")
    print("=" * 70)


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    main()