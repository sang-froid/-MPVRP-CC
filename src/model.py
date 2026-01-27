"""
Modèles de données pour les solutions MPVRP-CC
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import copy


@dataclass
class RouteStep:
    """Représente une étape dans une route"""
    node_type: str  # 'garage', 'depot', 'station'
    node_id: int
    product_id: int
    quantity: float
    cumulative_changeover_cost: float = 0.0
    
    def __str__(self):
        if self.node_type == 'garage':
            return f"Garage {self.node_id}"
        elif self.node_type == 'depot':
            return f"Depot {self.node_id} [Load {self.quantity:.0f} of P{self.product_id}]"
        elif self.node_type == 'station':
            return f"Station {self.node_id} [Deliver {self.quantity:.0f} of P{self.product_id}]"


@dataclass
class Route:
    """Représente une tournée complète d'un véhicule"""
    vehicle_id: int
    steps: List[RouteStep] = field(default_factory=list)
    total_distance: float = 0.0
    total_changeovers: int = 0
    total_changeover_cost: float = 0.0
    
    def add_step(self, step: RouteStep):
        """Ajoute une étape à la route"""
        self.steps.append(step)
    
    def is_valid(self) -> bool:
        """Vérifie que la route a une structure valide"""
        if len(self.steps) < 2:
            return False
        return (self.steps[0].node_type == 'garage' and 
                self.steps[-1].node_type == 'garage')
    
    def calculate_changeovers(self, instance) -> Tuple[int, float]:
        """Calcule le nombre et le coût des changeovers"""
        if len(self.steps) < 2:
            return 0, 0.0
        
        changeovers = 0
        total_cost = 0.0
        cumulative_cost = 0.0
        
        for i in range(1, len(self.steps)):
            prev_product = self.steps[i-1].product_id
            curr_product = self.steps[i].product_id
            
            if (prev_product != curr_product and 
                prev_product != 0 and curr_product != 0):
                changeovers += 1
                cost = instance.get_transition_cost(prev_product, curr_product)
                total_cost += cost
                cumulative_cost += cost
            
            self.steps[i].cumulative_changeover_cost = cumulative_cost
        
        self.total_changeovers = changeovers
        self.total_changeover_cost = total_cost
        
        return changeovers, total_cost
    
    def calculate_distance(self, instance) -> float:
        """Calcule la distance totale de la route"""
        distance = 0.0
        
        for i in range(len(self.steps) - 1):
            curr_step = self.steps[i]
            next_step = self.steps[i + 1]
            
            distance += instance.get_distance(
                curr_step.node_type, curr_step.node_id,
                next_step.node_type, next_step.node_id
            )
        
        self.total_distance = distance
        return distance
    
    def get_load_profile(self) -> List[float]:
        """Retourne le profil de charge du véhicule"""
        load_profile = [0.0]
        current_load = 0.0
        
        for step in self.steps[1:]:
            if step.node_type == 'depot':
                current_load += step.quantity
            elif step.node_type == 'station':
                current_load -= step.quantity
            load_profile.append(current_load)
        
        return load_profile
    
    def copy(self):
        """Crée une copie profonde de la route"""
        return copy.deepcopy(self)


@dataclass
class Solution:
    """Représente une solution complète au problème MPVRP-CC"""
    instance_name: str
    routes: List[Route] = field(default_factory=list)
    total_vehicles_used: int = 0
    total_changeovers: int = 0
    total_changeover_cost: float = 0.0
    total_distance: float = 0.0
    total_cost: float = 0.0
    processor: str = "Unknown"
    computation_time: float = 0.0
    is_feasible: bool = True
    
    def add_route(self, route: Route):
        """Ajoute une route à la solution"""
        if route.is_valid():
            self.routes.append(route)
    
    def calculate_metrics(self, instance):
        """Calcule toutes les métriques de la solution"""
        self.total_vehicles_used = 0
        self.total_changeovers = 0
        self.total_changeover_cost = 0.0
        self.total_distance = 0.0
        
        for route in self.routes:
            if len(route.steps) > 2:
                self.total_vehicles_used += 1
                route.calculate_distance(instance)
                route.calculate_changeovers(instance)
                
                self.total_distance += route.total_distance
                self.total_changeovers += route.total_changeovers
                self.total_changeover_cost += route.total_changeover_cost
        
        self.total_cost = self.total_distance + self.total_changeover_cost
        return self.total_cost
    
    def get_deliveries_summary(self) -> Dict[Tuple[int, int], float]:
        """Retourne un résumé des livraisons"""
        deliveries = {}
        for route in self.routes:
            for step in route.steps:
                if step.node_type == 'station':
                    key = (step.node_id, step.product_id)
                    deliveries[key] = deliveries.get(key, 0) + step.quantity
        return deliveries
    
    def copy(self):
        """Crée une copie profonde de la solution"""
        return copy.deepcopy(self)
    
    def __str__(self):
        return (f"Solution {self.instance_name}: "
                f"Cost={self.total_cost:.2f}, "
                f"Vehicles={self.total_vehicles_used}, "
                f"Distance={self.total_distance:.2f}, "
                f"Changeovers={self.total_changeovers}")