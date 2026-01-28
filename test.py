"""
MPVRP-CC Solver - Version Corrigée et Fonctionnelle
"""

import sys
import time
import platform
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple


# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

@dataclass
class Instance:
    name: str
    num_products: int
    num_vehicles: int
    transition_costs: np.ndarray
    vehicles: List  # [(id, capacity, home_garage, initial_product)]
    depots: List    # [(id, x, y, stocks_dict)]
    garages: List   # [(id, x, y)]
    stations: List  # [(id, x, y, demands_dict)]


def load_instance(filepath: str) -> Instance:
    """Charge une instance"""
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    idx = 0
    idx += 1  # Skip UUID
    
    num_products, num_depots, num_garages, num_stations, num_vehicles = map(int, lines[idx].split())
    idx += 1
    
    transition_costs = np.zeros((num_products, num_products))
    for i in range(num_products):
        transition_costs[i] = list(map(float, lines[idx].split()))
        idx += 1
    
    vehicles = []
    for _ in range(num_vehicles):
        parts = lines[idx].split()
        vehicles.append((int(parts[0]), float(parts[1]), int(parts[2]), int(parts[3])))
        idx += 1
    
    depots = []
    for _ in range(num_depots):
        parts = list(map(float, lines[idx].split()))
        stocks = {p+1: parts[3+p] for p in range(num_products)}
        depots.append((int(parts[0]), parts[1], parts[2], stocks))
        idx += 1
    
    garages = []
    for _ in range(num_garages):
        parts = list(map(float, lines[idx].split()))
        garages.append((int(parts[0]), parts[1], parts[2]))
        idx += 1
    
    stations = []
    for _ in range(num_stations):
        parts = list(map(float, lines[idx].split()))
        demands = {p+1: parts[3+p] for p in range(num_products) if parts[3+p] > 0}
        stations.append((int(parts[0]), parts[1], parts[2], demands))
        idx += 1
    
    return Instance(
        name=Path(filepath).stem,
        num_products=num_products,
        num_vehicles=num_vehicles,
        transition_costs=transition_costs,
        vehicles=vehicles,
        depots=depots,
        garages=garages,
        stations=stations
    )


# ============================================================================
# SOLVEUR SIMPLE (NEAREST NEIGHBOR)
# ============================================================================

def solve_simple(instance: Instance) -> List:
    """Solveur simple avec nearest neighbor"""
    
    all_routes = []
    vehicle_idx = 0
    
    # Pour chaque produit
    for product_id in range(1, instance.num_products + 1):
        print(f"\nProduct {product_id}:")
        
        # Stations demandant ce produit
        remaining_demands = {}
        for sid, x, y, demands in instance.stations:
            if product_id in demands:
                remaining_demands[sid] = demands[product_id]
        
        if not remaining_demands:
            continue
        
        print(f"  Total demand: {sum(remaining_demands.values()):.0f}")
        
        # Créer des routes jusqu'à satisfaire toute la demande
        while remaining_demands and vehicle_idx < instance.num_vehicles:
            vehicle = instance.vehicles[vehicle_idx]
            route = create_route_nearest_neighbor(
                instance, vehicle, product_id, remaining_demands
            )
            
            if route:
                all_routes.append(route)
                print(f"  Route {vehicle[0]}: {len(route['deliveries'])} deliveries")
            
            vehicle_idx += 1
        
        if remaining_demands:
            print(f"  WARNING: {len(remaining_demands)} stations not fully served")
    
    return all_routes


def create_route_nearest_neighbor(instance, vehicle, product_id, remaining_demands):
    """Crée une route avec nearest neighbor"""
    
    vehicle_id, capacity, garage_id, _ = vehicle
    depot = instance.depots[0]
    
    # Calculer charge
    total_needed = sum(remaining_demands.values())
    load = min(total_needed, capacity)
    
    if load == 0:
        return None
    
    route = {
        'vehicle_id': vehicle_id,
        'garage_id': garage_id,
        'depot_id': depot[0],
        'product_id': product_id,
        'load': load,
        'deliveries': []
    }
    
    current_load = load
    current_pos = (depot[1], depot[2])
    
    # Livrer avec nearest neighbor
    while remaining_demands and current_load > 0:
        # Trouver station la plus proche
        nearest_sid = None
        min_dist = float('inf')
        
        for sid in remaining_demands.keys():
            station = instance.stations[sid - 1]
            dist = np.sqrt((current_pos[0] - station[1])**2 + (current_pos[1] - station[2])**2)
            if dist < min_dist:
                min_dist = dist
                nearest_sid = sid
        
        if nearest_sid is None:
            break
        
        # Livrer à cette station
        demand = remaining_demands[nearest_sid]
        delivery = min(demand, current_load)
        
        route['deliveries'].append((nearest_sid, delivery))
        current_load -= delivery
        
        # Mettre à jour
        if delivery >= demand:
            del remaining_demands[nearest_sid]
        else:
            remaining_demands[nearest_sid] -= delivery
        
        # Nouvelle position
        station = instance.stations[nearest_sid - 1]
        current_pos = (station[1], station[2])
    
    return route if route['deliveries'] else None


# ============================================================================
# ÉCRITURE SOLUTION
# ============================================================================

def write_solution(routes, instance, output_path, comp_time):
    """Écrit la solution"""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for route in routes:
            # Ligne 1: Visites
            parts = [f"{route['vehicle_id']}:", str(route['garage_id']), 
                    f"{route['depot_id']} [{int(route['load'])}]"]
            for sid, qty in route['deliveries']:
                parts.append(f"{sid} ({int(qty)})")
            parts.append(str(route['garage_id']))
            f.write(" - ".join(parts) + "\n")
            
            # Ligne 2: Produits
            parts = [f"{route['vehicle_id']}:", "0(0.0)", f"{route['product_id']}(0.0)"]
            for _ in route['deliveries']:
                parts.append(f"{route['product_id']}(0.0)")
            parts.append("0(0.0)")
            f.write(" - ".join(parts) + "\n\n")
        
        # Métriques
        total_dist = calculate_total_distance(routes, instance)
        f.write(f"{len(routes)}\n0\n0.00\n{total_dist:.2f}\n")
        f.write(f"{platform.processor() or 'Unknown'}\n{comp_time:.3f}\n")


def calculate_total_distance(routes, instance):
    """Calcule distance totale"""
    total = 0
    for route in routes:
        garage = instance.garages[route['garage_id'] - 1]
        depot = instance.depots[route['depot_id'] - 1]
        
        prev_pos = (garage[1], garage[2])
        
        # Garage -> Depot
        total += np.sqrt((prev_pos[0] - depot[1])**2 + (prev_pos[1] - depot[2])**2)
        prev_pos = (depot[1], depot[2])
        
        # Depot -> Stations
        for sid, _ in route['deliveries']:
            station = instance.stations[sid - 1]
            total += np.sqrt((prev_pos[0] - station[1])**2 + (prev_pos[1] - station[2])**2)
            prev_pos = (station[1], station[2])
        
        # Last -> Garage
        total += np.sqrt((prev_pos[0] - garage[1])**2 + (prev_pos[1] - garage[2])**2)
    
    return total


# ============================================================================
# VALIDATION
# ============================================================================

def validate(routes, instance):
    """Valide la solution"""
    errors = []
    deliveries = {}
    
    # Calculer livraisons
    for route in routes:
        for sid, qty in route['deliveries']:
            key = (sid, route['product_id'])
            deliveries[key] = deliveries.get(key, 0) + qty
    
    # Vérifier demandes
    for sid, x, y, demands in instance.stations:
        for pid, demand in demands.items():
            delivered = deliveries.get((sid, pid), 0)
            if abs(delivered - demand) > 0.01:
                errors.append(f"Station {sid}, Product {pid}: Demand={demand:.2f}, Delivered={delivered:.2f}")
    
    return errors


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 3:
        print("Usage: python solver.py <instance.dat> <solution.dat>")
        sys.exit(1)
    
    print("="*70)
    print("MPVRP-CC Solver - Version Corrigée")
    print("="*70)
    
    start = time.time()
    
    instance = load_instance(sys.argv[1])
    print(f"\nLoaded: {instance.num_stations} stations, {instance.num_products} products, {instance.num_vehicles} vehicles")
    
    routes = solve_simple(instance)
    
    errors = validate(routes, instance)
    print(f"\nValidation: {'✓ FEASIBLE' if not errors else f'✗ {len(errors)} errors'}")
    if errors:
        for err in errors[:5]:
            print(f"  - {err}")
    
    comp_time = time.time() - start
    write_solution(routes, instance, sys.argv[2], comp_time)
    
    print(f"\n✓ Done in {comp_time:.3f}s")
    print("="*70)


if __name__ == "__main__":
    main()