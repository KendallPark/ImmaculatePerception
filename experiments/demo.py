# Use import statements for packages and modules only, not for individual types, classes, or functions.
# https://google.github.io/styleguide/pyguide.html#316-import-statements
# Follow this for external libraries.
import datasets as hf_datasets
import dataclasses
import transformers as tr

from typing import List

# from IPython import embed

import experiments


@dataclasses.dataclass
class MyExperiment(experiments.Experiment):
  param_1: bool = False
  param_2: int = 42
  # Default values for mutable fields like Lists have to be a 0-argument callable within a field:
  param_3: List[str] = dataclasses.field(
    default_factory=lambda: ["foo", "bar"],
    metadata={"aliases": ["--param_3", "-p3"], "help": "Special text for param_3!"},
  )
  param_4: str = "default"

  def run(self):
    print(
      f"Running MyExperiment with param_1={self.param_1}, param_2={self.param_2}, param_3={self.param_3}, param_4={self.param_4}"
    )


@dataclasses.dataclass
class MyExperimentAllCaps(MyExperiment):
  param_3: List[str] = tr.hf_argparser.HfArg(
    default_factory=lambda: ["FOO", "BAR"],
    aliases=["--param_3", "-p3"],
    help="HFArg() is a helper that is more concise than dataclasses.field()",
  )
  param_4: str = "DEFAULT"


@dataclasses.dataclass
class DemoModelTraining(experiments.Experiment):
  """A simple experiment to demonstrate the use of the Experient and StaticTrainer class."""

  # Define the experiment parameters
  max_steps: int = 1000
  logging_steps: int = 100
  save_steps: int = 100
  batch_size: int = 4
  learning_rate: float = 5e-5
  context_length: int = 512
  eval_size: int = 100  # Number of examples to sample from the test set; should be higher when not debugging
  train_size: int = 1000  # Number of examples to sample from the train set; should be higher when not debugging
  output_dir: str = "output/v2/demo"

  def run(self):
    """Run the experiment. This method should contain the experiment logic."""
    training_args = tr.TrainingArguments(
      output_dir=self.output_dir,
      per_device_train_batch_size=self.batch_size,
      per_device_eval_batch_size=self.batch_size,
      num_train_epochs=1,
      optim="adamw_torch",  # Default
      learning_rate=self.learning_rate,
      adam_epsilon=1e-8,  # Default
      weight_decay=0.01,  # The weight decay to apply to all layers except all bias and LayerNorm weights in AdamW.
      max_grad_norm=1.0,
      lr_scheduler_type="linear",  # Default
      warmup_steps=50,
      eval_strategy="steps",
      save_strategy="steps",
      logging_strategy="steps",
      max_steps=self.max_steps,  # Will also be used for LR scheduler
      logging_steps=self.logging_steps,
      save_steps=self.save_steps,
      log_level="info",
      report_to="none",
    )  # Or "wandb" or "none"

    tokenizer = tr.GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = hf_datasets.load_dataset(
      "fancyzhx/amazon_polarity",
      split={
        "train": f"train[:{self.train_size}]",
        "test": f"test[:{self.eval_size}]",
      },
    )

    def add_title(example):
      example["content"] = f'{example["title"]}\n{example["content"]}'
      return example

    def tokenize(examples):
      return tokenizer(
        examples["content"],
        padding="max_length",
        max_length=self.context_length,
        truncation=True,
      )

    # Prepare dataset
    # TODO(kendallpark): Add datapacking (squeeze in more examples per context_length)
    dataset = dataset.map(add_title)
    dataset = dataset.map(tokenize, batched=True).remove_columns(
      ["content", "title", "label"]
    )
    dataset = dataset.with_format("torch")  # This is key for fast execution

    model = tr.GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

    trainer = tr.Trainer(
      model=model,
      args=training_args,
      train_dataset=dataset["train"],
      eval_dataset=dataset["test"],
    )

    trainer.train()
