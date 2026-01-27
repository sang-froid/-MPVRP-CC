"""
Écriture des solutions au format .dat
"""

from pathlib import Path
from .model import Solution


class SolutionWriter:
    """Écrit les solutions au format .dat spécifié"""
    
    @staticmethod
    def write(solution: Solution, output_path: str):
        """Écrit une solution au format .dat"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            # Routes
            for route in solution.routes:
                if len(route.steps) <= 2:
                    continue
                
                visit_line = SolutionWriter._format_visit_sequence(route)
                f.write(visit_line + '\n')
                
                product_line = SolutionWriter._format_product_sequence(route)
                f.write(product_line + '\n')
                
                f.write('\n')
            
            # Métriques
            f.write(f"{solution.total_vehicles_used}\n")
            f.write(f"{solution.total_changeovers}\n")
            f.write(f"{solution.total_changeover_cost:.2f}\n")
            f.write(f"{solution.total_distance:.2f}\n")
            f.write(f"{solution.processor}\n")
            f.write(f"{solution.computation_time:.3f}\n")
        
        print(f"  ✓ Solution saved to: {output_path}")
    
    @staticmethod
    def _format_visit_sequence(route) -> str:
        """Formate la ligne de séquence de visites"""
        parts = [f"{route.vehicle_id}:"]
        
        for step in route.steps:
            if step.node_type == 'garage':
                parts.append(str(step.node_id))
            elif step.node_type == 'depot':
                parts.append(f"{step.node_id} [{step.quantity:.0f}]")
            elif step.node_type == 'station':
                parts.append(f"{step.node_id} ({step.quantity:.0f})")
        
        return " - ".join(parts)
    
    @staticmethod
    def _format_product_sequence(route) -> str:
        """Formate la ligne de séquence de produits"""
        parts = [f"{route.vehicle_id}:"]
        
        for step in route.steps:
            cost = step.cumulative_changeover_cost
            parts.append(f"{step.product_id}({cost:.1f})")
        
        return " - ".join(parts)