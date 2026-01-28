# Use import statements for packages and modules only, not for individual types, classes, or functions.
# https://google.github.io/styleguide/pyguide.html#316-import-statements
# Follow this for external libraries.
import transformers as tr

import sys
import experiments


def main():
  experiment_name = sys.argv[1]
  remaining_args = sys.argv[2:]
  # Auto-discover all experiments within the experiments directory
  if not experiments.registry.contains(experiment_name):
    experiments.registry.autodiscover()

  # Get the experiment class from the registry
  exp_class = experiments.registry.get_experiment(experiment_name)

  # Parse the remaining arguments
  (experiment,) = tr.HfArgumentParser(exp_class).parse_args_into_dataclasses(remaining_args)

  # Run the experiment
  experiment.execute()


if __name__ == "__main__":
  main()
