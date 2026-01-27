
"""
MPVRP-CC Solver Package - OR-Tools Version
Multi-Product Vehicle Routing Problem with Changeover Cost
"""

from .data_loader import Instance, InstanceLoader, Vehicle, Depot, Garage, Station
from .model import Solution, Route, RouteStep
from .ortools_solver import ORToolsSolver
from .validator import Validator
from .solution_writer import SolutionWriter

__all__ = [
    'Instance',
    'InstanceLoader',
    'Vehicle',
    'Depot',
    'Garage',
    'Station',
    'Solution',
    'Route',
    'RouteStep',
    'ORToolsSolver',
    'Validator',
    'SolutionWriter'
]

__version__ = '2.0.0-ortools'