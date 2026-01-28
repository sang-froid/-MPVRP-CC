"""
MPVRP-CC Solver – Version corrigée avec livraisons fractionnées
"""

import sys
import time
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from ortools.sat.python import cp_model


@dataclass
class Instance:
    name: str
    P: int
    D: int
    G: int
    S: int
    V: int
    transition_costs: np.ndarray
    vehicles: list
    depots: list
    garages: list
    stations: list


def load_instance(path: str) -> Instance:
    with open(path) as f:
        raw = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    i = 0
    while True:
        parts = raw[i].split()
        if all(p.replace('.', '').replace('-', '').isdigit() for p in parts):
            break
        i += 1

    P, D, G, S, V = map(int, raw[i].split())
    i += 1

    transition_costs = np.zeros((P, P), dtype=int)
    for p in range(P):
        transition_costs[p] = list(map(int, map(float, raw[i].split())))
        i += 1

    vehicles = []
    for _ in range(V):
        vid, cap, gid, prod = raw[i].split()
        vehicles.append((int(vid), int(float(cap)), int(gid), int(float(prod))))
        i += 1

    depots = []
    for _ in range(D):
        parts = list(map(float, raw[i].split()))
        stocks = {p+1: int(parts[3+p]) for p in range(P)}
        depots.append((int(parts[0]), parts[1], parts[2], stocks))
        i += 1

    garages = []
    for _ in range(G):
        gid, x, y = map(float, raw[i].split())
        garages.append((int(gid), x, y))
        i += 1

    stations = []
    for _ in range(S):
        parts = list(map(float, raw[i].split()))
        demands = {}
        for p in range(P):
            demand_val = parts[3 + p]
            if demand_val > 0:
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


def solve_with_ortools(instance: Instance):
    model = cp_model.CpModel()

    V, S, P = instance.V, instance.S, instance.P
    
    print(f"  Détail des stations et demandes:")
    for idx, (s_id, x, y, demands) in enumerate(instance.stations):
        print(f"  Station {s_id}: {demands}")

    # 1. Variables de livraison fractionnées
    x = {}
    for v in range(V):
        for s in range(S):
            for p in range(P):
                demande_max = instance.stations[s][3].get(p+1, 0)
                capacite_vehicule = instance.vehicles[v][1]
                max_val = min(demande_max, capacite_vehicule)
                x[v, s, p] = model.NewIntVar(0, max_val, f"x_{v}_{s}_{p}")

    # 2. Variables de visite
    visite = {}
    for v in range(V):
        for s in range(S):
            visite[v, s] = model.NewBoolVar(f"visite_{v}_{s}")

    # 3. Contraintes de demande
    print("\n  Contraintes de demande:")
    demand_constraints = 0
    for s_idx, (s_id, _, _, demands) in enumerate(instance.stations):
        for p, demande_totale in demands.items():
            if demande_totale > 0:
                constraint = sum(x[v, s_idx, p-1] for v in range(V))
                model.Add(constraint == demande_totale)
                demand_constraints += 1
                print(f"    Station {s_id}, Produit {p}: {demande_totale} unités")
    
    print(f"  Total contraintes de demande: {demand_constraints}")

    # 4. Lien entre visite et livraison
    for v in range(V):
        for s in range(S):
            # Somme des livraisons du véhicule v à la station s
            total_livraison = sum(x[v, s, p] for p in range(P))
            
            # Si total_livraison > 0, alors visite = 1
            model.Add(total_livraison > 0).OnlyEnforceIf(visite[v, s])
            model.Add(total_livraison == 0).OnlyEnforceIf(visite[v, s].Not())

    # 5. Contraintes de capacité
    print("\n  Capacités des véhicules:")
    for v in range(V):
        vid, capacite, gid, init_p = instance.vehicles[v]
        total_livraison_vehicule = sum(
            x[v, s, p] for s in range(S) for p in range(P)
        )
        model.Add(total_livraison_vehicule <= capacite)
        print(f"    Véhicule {vid}: capacité {capacite}")

    # 6. Objectif simplifié d'abord (juste minimiser le nombre de visites)
    objective_terms = []
    
    # Coût de visite (pour encourager la consolidation)
    for v in range(V):
        for s in range(S):
            objective_terms.append(1000 * visite[v, s])  # Coût fixe par visite
    
    # Coût de distance
    for v in range(V):
        _, _, gid, _ = instance.vehicles[v]
        gx, gy = instance.garages[gid-1][1], instance.garages[gid-1][2]
        
        for s_idx, (s_id, sx, sy, _) in enumerate(instance.stations):
            dist = int(math.hypot(sx - gx, sy - gy) * 100)  # Échelle
            objective_terms.append(dist * visite[v, s_idx])

    model.Minimize(sum(objective_terms))

    # 7. Résolution
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.num_search_workers = 4
    solver.parameters.log_search_progress = True
    
    print("\n  Résolution en cours...")
    status = solver.Solve(model)
    
    print(f"\n  Statut: {status}")
    print(f"  Statut OPTIMAL: {cp_model.OPTIMAL}")
    print(f"  Statut FEASIBLE: {cp_model.FEASIBLE}")

    # 8. Vérification de la solution
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"\n  Solution trouvée!")
        print(f"  Valeur objectif: {solver.ObjectiveValue()}")
        
        # Vérifier les demandes satisfaites
        for s_idx, (s_id, _, _, demands) in enumerate(instance.stations):
            for p, demande_totale in demands.items():
                if demande_totale > 0:
                    livraison_totale = sum(
                        solver.Value(x[v, s_idx, p-1]) for v in range(V)
                    )
                    print(f"    Station {s_id}, Produit {p}: demandé {demande_totale}, livré {livraison_totale}")
        
        # Extraire les routes
        routes = []
        for v in range(V):
            vid, cap, gid, init_p = instance.vehicles[v]
            
            deliveries_by_station = {}
            total_livraison_vehicule = 0
            
            for s_idx in range(S):
                station_livraisons = []
                for p in range(P):
                    qty = solver.Value(x[v, s_idx, p])
                    if qty > 0:
                        station_livraisons.append((p+1, qty))
                        total_livraison_vehicule += qty
                
                if station_livraisons:
                    s_id = instance.stations[s_idx][0]
                    deliveries_by_station[s_id] = station_livraisons
            
            if deliveries_by_station:
                route_info = {
                    "vehicle": vid,
                    "garage": gid,
                    "depot": instance.depots[0][0],
                    "init_product": init_p,
                    "capacity_used": total_livraison_vehicule,
                    "total_capacity": cap,
                    "stations": deliveries_by_station
                }
                routes.append(route_info)
                print(f"\n  Véhicule {vid}:")
                print(f"    Capacité utilisée: {total_livraison_vehicule}/{cap}")
                for s_id, livraisons in deliveries_by_station.items():
                    produits = ", ".join(f"P{p}={q}" for p, q in livraisons)
                    print(f"    Station {s_id}: {produits}")
        
        return routes, solver.ObjectiveValue()
    else:
        print(f"\n  Aucune solution trouvée!")
        print(f"  Statut: {status}")
        return [], float('inf')


def main():
    if len(sys.argv) < 3:
        print("Usage: python solver_fractionne.py instance.dat solution.dat")
        sys.exit(1)

    print("="*70)
    print("MPVRP-CC Solver – Livraisons fractionnées (version corrigée)")
    print("="*70)

    start = time.time()
    inst = load_instance(sys.argv[1])
    
    print(f"\nInstance: {inst.name}")
    print(f"Stations: {inst.S}, Produits: {inst.P}, Véhicules: {inst.V}")
    
    # Calcul demande totale
    demande_totale = 0
    for _, _, _, demands in inst.stations:
        demande_totale += sum(demands.values())
    print(f"Demande totale: {demande_totale} unités")
    
    # Capacité totale des véhicules
    capacite_totale = sum(v[1] for v in inst.vehicles)
    print(f"Capacité totale véhicules: {capacite_totale} unités")
    
    if demande_totale > capacite_totale:
        print(f"ATTENTION: Demande totale ({demande_totale}) > Capacité totale ({capacite_totale})")
        print("Le problème pourrait être insoluble!")
    
    routes, cout = solve_with_ortools(inst)
    comp_time = time.time() - start

    print(f"\n" + "="*70)
    print(f"Résumé:")
    print(f"  Temps de calcul: {comp_time:.3f}s")
    print(f"  Coût total: {cout:.2f}")
    print(f"  Nombre de routes: {len(routes)}")
    
    if routes:
        # Vérification finale
        demande_satisfaite = 0
        for route in routes:
            for s_id, livraisons in route["stations"].items():
                for p, q in livraisons:
                    demande_satisfaite += q
        
        print(f"  Demande satisfaite: {demande_satisfaite}/{demande_totale}")
        
        if demande_satisfaite == demande_totale:
            print("  ✓ Toute la demande est satisfaite!")
        else:
            print(f"  ✗ Demande partiellement satisfaite: {demande_satisfaite}/{demande_totale}")
    
    print("="*70)


if __name__ == "__main__":
    main()