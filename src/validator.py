"""
Validation des solutions MPVRP-CC
"""

from typing import List, Tuple, Dict
from .model import Solution
from .data_loader import Instance


class Validator:
    """Valide les solutions MPVRP-CC"""
    
    def __init__(self, instance: Instance):
        self.instance = instance
    
    def validate(self, solution: Solution) -> Tuple[bool, List[str]]:
        """Valide une solution complète"""
        errors = []
        
        errors.extend(self._check_demand_satisfaction(solution))
        errors.extend(self._check_vehicle_capacity(solution))
        errors.extend(self._check_route_structure(solution))
        errors.extend(self._check_depot_stock(solution))
        
        is_valid = len(errors) == 0
        solution.is_feasible = is_valid
        
        return is_valid, errors
    
    def _check_demand_satisfaction(self, solution: Solution) -> List[str]:
        """Vérifie que toutes les demandes sont satisfaites"""
        errors = []
        deliveries = solution.get_deliveries_summary()
        
        for station in self.instance.stations:
            for product_id, demand in station.demand.items():
                key = (station.id, product_id)
                delivered = deliveries.get(key, 0)
                
                if abs(delivered - demand) > 0.01:
                    errors.append(
                        f"Station {station.id}, Product {product_id}: "
                        f"Demand={demand:.2f}, Delivered={delivered:.2f}"
                    )
        
        return errors
    
    def _check_vehicle_capacity(self, solution: Solution) -> List[str]:
        """Vérifie que la capacité des véhicules est respectée"""
        errors = []
        
        for route in solution.routes:
            vehicle = self.instance.vehicles[route.vehicle_id - 1]
            load_profile = route.get_load_profile()
            max_load = max(load_profile) if load_profile else 0
            
            if max_load > vehicle.capacity + 0.01:
                errors.append(
                    f"Route {route.vehicle_id}: "
                    f"Max load {max_load:.2f} > Capacity {vehicle.capacity:.2f}"
                )
        
        return errors
    
    def _check_route_structure(self, solution: Solution) -> List[str]:
        """Vérifie la structure des routes"""
        errors = []
        
        for route in solution.routes:
            if len(route.steps) < 2:
                errors.append(f"Route {route.vehicle_id}: Too few steps")
                continue
            
            if route.steps[0].node_type != 'garage':
                errors.append(f"Route {route.vehicle_id}: Must start at garage")
            
            if route.steps[-1].node_type != 'garage':
                errors.append(f"Route {route.vehicle_id}: Must end at garage")
            
            vehicle = self.instance.vehicles[route.vehicle_id - 1]
            if route.steps[0].node_id != vehicle.home_garage:
                errors.append(
                    f"Route {route.vehicle_id}: "
                    f"Must start at home garage {vehicle.home_garage}"
                )
        
        return errors
    
    def _check_depot_stock(self, solution: Solution) -> List[str]:
        """Vérifie que les stocks des dépôts sont suffisants"""
        errors = []
        usage = {}
        
        for route in solution.routes:
            for step in route.steps:
                if step.node_type == 'depot':
                    key = (step.node_id, step.product_id)
                    usage[key] = usage.get(key, 0) + step.quantity
        
        for depot in self.instance.depots:
            for product_id, stock in depot.stock.items():
                key = (depot.id, product_id)
                used = usage.get(key, 0)
                
                if used > stock + 0.01:
                    errors.append(
                        f"Depot {depot.id}, Product {product_id}: "
                        f"Stock={stock:.2f}, Used={used:.2f}"
                    )
        
        return errors