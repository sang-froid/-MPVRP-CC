"""
Solveur MPVRP-CC utilisant Google OR-Tools
Stratégie: Résoudre par produit puis consolider
"""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
from typing import Dict, List, Tuple, Set
from .model import Solution, Route, RouteStep
from .data_loader import Instance


class ORToolsSolver:
    """
    Solveur basé sur OR-Tools pour MPVRP-CC
    
    Stratégie:
    1. Résoudre un VRP par produit (ignorer les changeovers)
    2. Consolider les routes pour minimiser les véhicules
    3. Optimiser l'ordre des produits pour minimiser les changeovers
    """
    
    def __init__(self, instance: Instance):
        self.instance = instance
        
    def solve(self, time_limit_seconds: int = 600) -> Solution:
        """
        Résout le problème MPVRP-CC
        
        Args:
            time_limit_seconds: Limite de temps en secondes
        
        Returns:
            Solution trouvée
        """
        print("\n" + "="*70)
        print("OR-Tools Solver - MPVRP-CC")
        print("="*70)
        
        solution = Solution(instance_name=self.instance.name)
        
        # Étape 1: Résoudre un sous-problème VRP par produit
        print("\nStep 1: Solving VRP for each product...")
        routes_by_product = self._solve_vrp_per_product(time_limit_seconds)
        
        # Étape 2: Consolider les routes sur les véhicules disponibles
        print("\nStep 2: Consolidating routes on vehicles...")
        consolidated_routes = self._consolidate_routes(routes_by_product)
        
        # Étape 3: Construire la solution finale
        print("\nStep 3: Building final solution...")
        for route in consolidated_routes:
            solution.add_route(route)
        
        # Calculer les métriques
        solution.calculate_metrics(self.instance)
        
        print(f"\n✓ Solution generated:")
        print(f"  - Vehicles used: {solution.total_vehicles_used}/{self.instance.num_vehicles}")
        print(f"  - Total distance: {solution.total_distance:.2f}")
        print(f"  - Changeovers: {solution.total_changeovers}")
        print(f"  - Total cost: {solution.total_cost:.2f}")
        
        return solution
    
    def _solve_vrp_per_product(self, time_limit: int) -> Dict[int, List[List[int]]]:
        """
        Résout un VRP pour chaque produit indépendamment
        
        Returns:
            Dict[product_id] = [route1, route2, ...] où chaque route est une liste de station_ids
        """
        routes_by_product = {}
        
        for product_id in range(1, self.instance.num_products + 1):
            print(f"  Solving VRP for product {product_id}...")
            
            # Trouver les stations qui ont besoin de ce produit
            stations_needing_product = [
                station for station in self.instance.stations
                if product_id in station.demand
            ]
            
            if not stations_needing_product:
                print(f"    No stations need product {product_id}")
                continue
            
            # Résoudre le VRP pour ce produit
            routes = self._solve_single_product_vrp(
                product_id, 
                stations_needing_product,
                time_limit // self.instance.num_products
            )
            
            routes_by_product[product_id] = routes
            print(f"    Found {len(routes)} routes for product {product_id}")
        
        return routes_by_product
    
    def _solve_single_product_vrp(
        self, 
        product_id: int, 
        stations: List, 
        time_limit: int
    ) -> List[List[int]]:
        """
        Résout un VRP classique pour un produit donné
        
        Args:
            product_id: ID du produit
            stations: Liste des stations à servir
            time_limit: Limite de temps
        
        Returns:
            Liste de routes (chaque route = liste de station_ids)
        """
        if not stations:
            return []
        
        # Créer la matrice de distances
        num_locations = len(stations) + 1  # +1 pour le dépôt
        distance_matrix = self._create_distance_matrix_for_product(stations)
        
        # Créer les demandes
        demands = [0]  # Dépôt
        for station in stations:
            demands.append(int(station.demand.get(product_id, 0)))
        
        # Capacités des véhicules
        vehicle_capacities = [int(v.capacity) for v in self.instance.vehicles]
        
        # Créer le modèle OR-Tools
        manager = pywrapcp.RoutingIndexManager(
            num_locations,
            len(self.instance.vehicles),
            0  # Dépôt = index 0
        )
        
        routing = pywrapcp.RoutingModel(manager)
        
        # Fonction de distance
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node])
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        # Fonction de demande
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return demands[from_node]
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        
        # Ajouter contrainte de capacité
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,
            vehicle_capacities,
            True,
            'Capacity'
        )
        
        # Paramètres de recherche
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = time_limit
        
        # Résoudre
        solution = routing.SolveWithParameters(search_parameters)
        
        # Extraire les routes
        routes = []
        if solution:
            for vehicle_id in range(len(self.instance.vehicles)):
                index = routing.Start(vehicle_id)
                route = []
                
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    if node_index > 0:  # Pas le dépôt
                        station = stations[node_index - 1]
                        route.append(station.id)
                    index = solution.Value(routing.NextVar(index))
                
                if route:
                    routes.append(route)
        
        return routes
    
    def _create_distance_matrix_for_product(self, stations: List) -> np.ndarray:
        """Crée une matrice de distances pour un ensemble de stations"""
        num_locations = len(stations) + 1
        distance_matrix = np.zeros((num_locations, num_locations))
        
        # Position du dépôt
        depot = self.instance.depots[0]
        depot_pos = (depot.x, depot.y)
        
        # Distances dépôt <-> stations
        for i, station in enumerate(stations):
            station_pos = (station.x, station.y)
            dist = np.sqrt(
                (depot_pos[0] - station_pos[0])**2 + 
                (depot_pos[1] - station_pos[1])**2
            )
            distance_matrix[0][i + 1] = dist
            distance_matrix[i + 1][0] = dist
        
        # Distances entre stations
        for i, station1 in enumerate(stations):
            for j, station2 in enumerate(stations):
                if i != j:
                    pos1 = (station1.x, station1.y)
                    pos2 = (station2.x, station2.y)
                    dist = np.sqrt(
                        (pos1[0] - pos2[0])**2 + 
                        (pos1[1] - pos2[1])**2
                    )
                    distance_matrix[i + 1][j + 1] = dist
        
        return distance_matrix
    
    def _consolidate_routes(self, routes_by_product: Dict[int, List[List[int]]]) -> List[Route]:
        """
        Consolide les routes de différents produits sur les véhicules disponibles
        
        Stratégie:
        1. Assigner les routes aux véhicules
        2. Grouper les produits par véhicule pour minimiser les changeovers
        
        Args:
            routes_by_product: Dict[product_id] = [[station_ids], ...]
        
        Returns:
            Liste de routes consolidées
        """
        consolidated_routes = []
        vehicle_idx = 0
        
        # Pour chaque produit
        for product_id in sorted(routes_by_product.keys()):
            product_routes = routes_by_product[product_id]
            
            for station_ids in product_routes:
                if vehicle_idx >= len(self.instance.vehicles):
                    print(f"  WARNING: Not enough vehicles! Skipping route.")
                    break
                
                vehicle = self.instance.vehicles[vehicle_idx]
                route = self._create_route(vehicle, product_id, station_ids)
                
                if route and route.is_valid():
                    consolidated_routes.append(route)
                    vehicle_idx += 1
        
        return consolidated_routes
    
    def _create_route(
        self, 
        vehicle, 
        product_id: int, 
        station_ids: List[int]
    ) -> Route:
        """
        Crée une route complète pour un véhicule livrant un produit
        
        Args:
            vehicle: Véhicule à utiliser
            product_id: Produit à livrer
            station_ids: Liste des IDs de stations
        
        Returns:
            Route complète
        """
        route = Route(vehicle_id=vehicle.id)
        
        # Départ du garage
        route.add_step(RouteStep(
            node_type='garage',
            node_id=vehicle.home_garage,
            product_id=0,
            quantity=0
        ))
        
        # Calculer la quantité totale à livrer
        total_quantity = 0
        for station_id in station_ids:
            station = self.instance.get_station_by_id(station_id)
            total_quantity += station.demand.get(product_id, 0)
        
        # Charger au dépôt (on prend le premier dépôt pour simplifier)
        depot = self.instance.depots[0]
        load_quantity = min(total_quantity, vehicle.capacity)
        
        route.add_step(RouteStep(
            node_type='depot',
            node_id=depot.id,
            product_id=product_id,
            quantity=load_quantity
        ))
        
        # Livraisons aux stations
        remaining_load = load_quantity
        for station_id in station_ids:
            station = self.instance.get_station_by_id(station_id)
            demand = station.demand.get(product_id, 0)
            delivery = min(demand, remaining_load)
            
            if delivery > 0:
                route.add_step(RouteStep(
                    node_type='station',
                    node_id=station_id,
                    product_id=product_id,
                    quantity=delivery
                ))
                remaining_load -= delivery
        
        # Retour au garage
        route.add_step(RouteStep(
            node_type='garage',
            node_id=vehicle.home_garage,
            product_id=0,
            quantity=0
        ))
        
        return route