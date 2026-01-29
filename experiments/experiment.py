import abc
import dataclasses
from typing import Optional, Dict, Any

try:
  import wandb
except ImportError:
  wandb = None


@dataclasses.dataclass
class Experiment(abc.ABC):
  """Base class for all experiments. All experiments should inherit from this class.

  @dataclasses.dataclass
  class MyExperiment(Experiment):
    # Define the experiment parameters
    max_steps: int = 1000
    batch_size: int = 4
    wandb_project: Optional[str] = "my_project"

    def run(self):
      self.init_wandb()
      # Run the experiment logic
      tr.TrainingArguments(max_steps=self.max_steps, per_device_train_batch_size=self.batch_size)
      # ...
  """

  # WandB Logging Configuration
  wandb_project: Optional[str] = None
  wandb_entity: Optional[str] = None
  debug: bool = False

  def params(self) -> Dict[str, Any]:
    """Return a dictionary of the experiment parameters.

    Can be used to log the parameters to a logger or to a tracking service like Weights & Biases.
    e.g. `wandb.init(project="my_project", config=self.params())`
    """
    return dataclasses.asdict(self)

  def init_wandb(self, name: Optional[str] = None):
    """
    Initializes Weights & Biases logging.

    Logic:
    1. If `debug` is True -> Do NOT log.
    2. If `wandb` is not installed -> Warning and skip.
    3. If `wandb_project` is set -> Initialize.
    """
    if self.debug:
      print("Debug mode enabled: WandB logging disabled.")
      return

    if self.wandb_project:
      if wandb is not None:
        wandb.init(
          project=self.wandb_project,
          entity=self.wandb_entity,
          name=name,
          config={'exp_config': self.params()}
        )
      else:
        print("Warning: `wandb_project` is set, but `wandb` is not installed.")

  def execute(self):
    """
    Template method that initializes services (like WandB) and then runs the experiment.
    Callers should invoke this method instead of `run()`.
    """
    self.init_wandb()
    self.run()

  @property
  def run_name(self) -> str:
    """
    Returns a run name for logging (e.g. wandb).
    Can be overridden by subclasses to include hyperparams.
    """
    return self.__class__.__name__

  @abc.abstractmethod
  def run(self):
    """
    Run the experiment.
    """
    pass
