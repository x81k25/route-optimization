# core optimization algorithms - new pipeline structure

# import modules to enable src.core.module_name.function() syntax  
from . import _4_reporting as report  # renamed module

# new pipeline modules are imported directly in main.py
# legacy optimize.py has been replaced by:
# - _3_0_optimization.py (master coordinator)
# - _3_1_optimization_primary_day_assignment.py
# - _3_2_optimization_secondary_day_clustering.py  
# - _3_3_optimization_route_optimization.py
# - _3_4_optimization_cluster_balancing.py
# - _3_5_optimization_detailed_routing.py