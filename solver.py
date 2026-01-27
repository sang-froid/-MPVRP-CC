"""
Solveur principal MPVRP-CC avec OR-Tools
"""

import sys
import time
import platform
from pathlib import Path

from src import (
    InstanceLoader,
    ORToolsSolver,
    Validator,
    SolutionWriter
)


def solve_instance(
    instance_path: str, 
    output_path: str, 
    time_limit: int = 600
):
    """
    Résout une instance MPVRP-CC avec OR-Tools
    
    Args:
        instance_path: Chemin vers le fichier d'instance .dat
        output_path: Chemin vers le fichier de solution .dat
        time_limit: Limite de temps en secondes
    """
    start_time = time.time()
    
    print("=" * 70)
    print("MPVRP-CC Solver with OR-Tools")
    print("Multi-Product Vehicle Routing Problem with Changeover Cost")
    print("=" * 70)
    print(f"Instance: {instance_path}")
    print(f"Output: {output_path}")
    print(f"Time limit: {time_limit}s")
    print()
    
    # Chargement de l'instance
    print("="*70)
    print("Loading instance...")
    print("="*70)
    
    try:
        instance = InstanceLoader.load(instance_path)
    except Exception as e:
        print(f"ERROR loading instance: {e}")
        sys.exit(1)
    
    print(f"  UUID: {instance.uuid}")
    print(f"  Products: {instance.num_products}")
    print(f"  Depots: {instance.num_depots}")
    print(f"  Garages: {instance.num_garages}")
    print(f"  Stations: {instance.num_stations}")
    print(f"  Vehicles: {instance.num_vehicles}")
    
    # Afficher la demande totale
    total_demand = instance.get_total_demand()
    print(f"\n  Total demand by product:")
    for prod_id in sorted(total_demand.keys()):
        print(f"    Product {prod_id}: {total_demand[prod_id]:.0f}")
    
    # Résolution avec OR-Tools
    print("\n" + "="*70)
    print("Solving with OR-Tools...")
    print("="*70)
    
    solver = ORToolsSolver(instance)
    solution = solver.solve(time_limit)
    
    if solution is None:
        print("\nERROR: No solution found!")
        sys.exit(1)
    
    # Validation
    print("\n" + "="*70)
    print("Validation")
    print("="*70)
    
    validator = Validator(instance)
    is_valid, errors = validator.validate(solution)
    
    if is_valid:
        print("  ✓ Solution is FEASIBLE")
    else:
        print("  ✗ Solution has CONSTRAINT VIOLATIONS:")
        for error in errors[:10]:
            print(f"    - {error}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more errors")
    
    # Métriques finales
    total_time = time.time() - start_time
    solution.computation_time = total_time
    solution.processor = platform.processor() or "Unknown Processor"
    
    print("\n" + "="*70)
    print("Final Solution Summary")
    print("="*70)
    print(f"  Total cost: {solution.total_cost:.2f}")
    print(f"  Distance: {solution.total_distance:.2f}")
    print(f"  Changeovers: {solution.total_changeovers}")
    print(f"  Changeover cost: {solution.total_changeover_cost:.2f}")
    print(f"  Vehicles used: {solution.total_vehicles_used}/{instance.num_vehicles}")
    print(f"  Feasible: {'YES' if is_valid else 'NO'}")
    print(f"  Computation time: {total_time:.3f}s")
    print(f"  Processor: {solution.processor}")
    
    # Écriture de la solution
    print("\n" + "="*70)
    print("Writing solution...")
    print("="*70)
    
    try:
        SolutionWriter.write(solution, output_path)
    except Exception as e:
        print(f"ERROR writing solution: {e}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✓ Done!")
    print("="*70)
    
    return solution


def main():
    """Point d'entrée principal"""
    if len(sys.argv) < 3:
        print("Usage: python solver.py <instance_file> <solution_file> [time_limit]")
        print()
        print("Arguments:")
        print("  instance_file  : Path to instance .dat file")
        print("  solution_file  : Path to output solution .dat file")
        print("  time_limit     : Time limit in seconds (default: 600)")
        print()
        print("Example:")
        print("  python solver.py instances/small/MPVRP_S_001_s9_d1_p2.dat solutions/small/Sol_MPVRP_S_001.dat 300")
        sys.exit(1)
    
    instance_path = sys.argv[1]
    output_path = sys.argv[2]
    time_limit = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    
    if not Path(instance_path).exists():
        print(f"ERROR: Instance file not found: {instance_path}")
        sys.exit(1)
    
    try:
        solve_instance(instance_path, output_path, time_limit)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()