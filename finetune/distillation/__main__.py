"""python -m finetune.distillation {inspect,smoke} CONFIG."""

import argparse
import json

from .config import ExperimentConfig
from .models import parameter_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect", help="Exact meta-device counts; no weights or downloads"
    )
    inspect.add_argument("config")
    smoke = commands.add_parser(
        "smoke", help="Bounded synthetic experiment, never a pretrained teacher"
    )
    smoke.add_argument("config")
    smoke.add_argument(
        "--run-dir",
        required=True,
        help="New output directory; existing directories are refused",
    )
    args = parser.parse_args()
    config = ExperimentConfig.load(args.config)
    if args.command == "inspect":
        result = {
            "purpose": config.purpose,
            "teacher": parameter_report(config.teacher),
            "student": parameter_report(config.student),
            "training_only_hidden_projector_parameters": (
                len(config.hidden.student_layers)
                * config.student.dim
                * config.teacher.dim
                if config.loss.hidden_weight
                else 0
            ),
        }
    else:
        from .smoke import run_smoke

        result = run_smoke(config, args.run_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
