"""
Module de chargement des instances MPVRP-CC
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path


@dataclass
class Vehicle:
    """Représente un véhicule"""
    id: int
    capacity: float
    home_garage: int
    initial_product: int


@dataclass
class Depot:
    """Représente un dépôt"""
    id: int
    x: float
    y: float
    stock: Dict[int, float]


@dataclass
class Garage:
    """Représente un garage"""
    id: int
    x: float
    y: float


@dataclass
class Station:
    """Représente une station-service"""
    id: int
    x: float
    y: float
    demand: Dict[int, float]
    
    def get_total_demand(self) -> float:
        """Retourne la demande totale de la station"""
        return sum(self.demand.values())
    
    def get_products_needed(self) -> List[int]:
        """Retourne la liste des produits nécessaires"""
        return [p for p, q in self.demand.items() if q > 0]


@dataclass
class Instance:
    """Représente une instance complète du problème"""
    name: str
    uuid: str
    num_products: int
    num_depots: int
    num_garages: int
    num_stations: int
    num_vehicles: int
    transition_costs: np.ndarray
    vehicles: List[Vehicle]
    depots: List[Depot]
    garages: List[Garage]
    stations: List[Station]
    
    def get_distance(self, node1_type: str, node1_id: int, 
                     node2_type: str, node2_id: int) -> float:
        """Calcule la distance euclidienne entre deux nœuds"""
        pos1 = self._get_position(node1_type, node1_id)
        pos2 = self._get_position(node2_type, node2_id)
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _get_position(self, node_type: str, node_id: int) -> Tuple[float, float]:
        """Retourne la position (x, y) d'un nœud"""
        if node_type == 'garage':
            return (self.garages[node_id - 1].x, self.garages[node_id - 1].y)
        elif node_type == 'depot':
            return (self.depots[node_id - 1].x, self.depots[node_id - 1].y)
        elif node_type == 'station':
            return (self.stations[node_id - 1].x, self.stations[node_id - 1].y)
        else:
            raise ValueError(f"Type de nœud inconnu: {node_type}")
    
    def get_transition_cost(self, from_product: int, to_product: int) -> float:
        """Retourne le coût de changement de produit"""
        if from_product == 0 or to_product == 0:
            return 0.0
        return self.transition_costs[from_product - 1][to_product - 1]
    
    def get_total_demand(self) -> Dict[int, float]:
        """Retourne la demande totale par produit"""
        total_demand = {}
        for station in self.stations:
            for product_id, quantity in station.demand.items():
                total_demand[product_id] = total_demand.get(product_id, 0) + quantity
        return total_demand
    
    def get_station_by_id(self, station_id: int) -> Station:
        """Retourne une station par son ID"""
        return self.stations[station_id - 1]


class InstanceLoader:
    """Charge les instances depuis des fichiers .dat"""
    
    @staticmethod
    def load(filepath: str) -> Instance:
        """Charge une instance depuis un fichier .dat"""
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        idx = 0
        
        # UUID
        uuid = lines[idx].replace('#', '').strip()
        idx += 1
        
        # Paramètres globaux
        params = list(map(int, lines[idx].split()))
        num_products, num_depots, num_garages, num_stations, num_vehicles = params
        idx += 1
        
        # Matrice des coûts de transition
        transition_costs = np.zeros((num_products, num_products))
        for i in range(num_products):
            costs = list(map(float, lines[idx].split()))
            transition_costs[i] = costs
            idx += 1
        
        # Véhicules
        vehicles = []
        for v in range(num_vehicles):
            parts = lines[idx].split()
            vehicles.append(Vehicle(
                id=int(parts[0]),
                capacity=float(parts[1]),
                home_garage=int(parts[2]),
                initial_product=int(parts[3])
            ))
            idx += 1
        
        # Dépôts
        depots = []
        for d in range(num_depots):
            parts = list(map(float, lines[idx].split()))
            depot_id = int(parts[0])
            x, y = parts[1], parts[2]
            stock = {p + 1: parts[3 + p] for p in range(num_products)}
            depots.append(Depot(depot_id, x, y, stock))
            idx += 1
        
        # Garages
        garages = []
        for g in range(num_garages):
            parts = list(map(float, lines[idx].split()))
            garages.append(Garage(int(parts[0]), parts[1], parts[2]))
            idx += 1
        
        # Stations
        stations = []
        for s in range(num_stations):
            parts = list(map(float, lines[idx].split()))
            station_id = int(parts[0])
            x, y = parts[1], parts[2]
            demand = {}
            for p in range(num_products):
                qty = parts[3 + p]
                if qty > 0:
                    demand[p + 1] = qty
            stations.append(Station(station_id, x, y, demand))
            idx += 1
        
        return Instance(
            name=filepath.stem,
            uuid=uuid,
            num_products=num_products,
            num_depots=num_depots,
            num_garages=num_garages,
            num_stations=num_stations,
            num_vehicles=num_vehicles,
            transition_costs=transition_costs,
            vehicles=vehicles,
            depots=depots,
            garages=garages,
            stations=stations
        )