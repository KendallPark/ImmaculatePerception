from typing import Dict, Type, List
import importlib
from pathlib import Path

import inspect
import pkgutil

from .experiment import Experiment

# Add your experiments to this registry if you want to skip the auto-discovery in run.py]
# This should be a little faster than auto-discovery, but you need to remember to add new experiments here.
_experiment_registry: List[Type[Experiment]] = [
]


class Registry:
  """Registry that auto-discovers all Experiment subclasses"""

  def __init__(self):
    self._experiments: Dict[str, Type[Experiment]] = {}
    self._initialized = False
    for experiment in _experiment_registry:
      self.register(experiment)

  def register(self, experiment_class: Type[Experiment], skip_collisions: bool = False) -> None:
    """Register a single experiment class"""
    if not inspect.isclass(experiment_class) or not issubclass(experiment_class, Experiment):
      raise TypeError(f"{experiment_class.__name__} is not a subclass of Experiment")

    # Skip the base class itself
    if experiment_class is Experiment:
      return

    name = experiment_class.__name__.lower()
    if name in self._experiments and skip_collisions:
      return
    if name in self._experiments:
      raise ValueError(f"Experiment {name} is already registered")

    self._experiments[name] = experiment_class
    # print(f"Registered experiment: {experiment_class.__name__}")

  def contains(self, name: str) -> bool:
    """Check if a experiment is registered"""
    return name.lower() in self._experiments

  def get_experiment(self, name: str) -> Type[Experiment]:
    """Get a experiment by name"""
    name = name.lower()
    if name not in self._experiments:
      raise KeyError(f"Experiment {name} not found in registry")
    return self._experiments[name]

  def get_all_experiments(self) -> List[Type[Experiment]]:
    """Get all registered experiments"""
    return list(self._experiments.values())

  def autodiscover(self, package_path: str = "experiments") -> None:
    """
    Auto-discover and register all Experiment subclasses

    Args:
      package_path: Name of package_path to search in (default: 'experiments')
    """
    if self._initialized:
      return

    try:
      # # Import the package
      package = importlib.import_module(package_path)
      package_dir = Path(package.__file__).parent

      # Walk through all modules in the package
      for module_info in pkgutil.walk_packages([str(package_dir)], prefix=f"{package_path}."):
        try:
          module = importlib.import_module(module_info.name)
          # Find all Experiment subclasses in the module
          for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and issubclass(obj, Experiment) and obj is not Experiment and
                obj.__module__ == module.__name__):
              self.register(obj, skip_collisions=True)
        except ImportError as e:
          print(f"Failed to import {module_info.name}: {e}")

    except ImportError as e:
      print(f"Could not import experiments package: {e}")
    except ValueError as e:
      print(f"Package discovery failed: {e}")

    self._initialized = True


# Create the global registry instance
registry = Registry()
