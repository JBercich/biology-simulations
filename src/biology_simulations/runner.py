#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import logging
from typing import Any

from biology_simulations.logger import setup_logging
from biology_simulations.epidemiology.seir import run_seir_simulation


class SetWidthHelpFormatter(argparse.HelpFormatter):
	"""Utility help formatter for 80-character stdout line length."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs, width=80)


# Entrypoint command-line parser for running a single simulation. Builds
# the experiment runtime using a basic lightweight argparse config as a
# baseline starting point for the codebase.

parser: argparse.ArgumentParser = argparse.ArgumentParser(
	description="Run a biology simulation from the repository.",
	formatter_class=SetWidthHelpFormatter,
)
parser.add_argument(
	"-v",
	"--verbose",
	action="store_true",
	default=False,
	help="provide verbose runtime logging",
)

simulation_parser: argparse._SubParsersAction = parser.add_subparsers(
	title="simulation",
	description="type simulation to run with optional parameters",
	help="name of the simulation to run",
	required=True,
	dest="simulation",
)

seir_parser: Any = simulation_parser.add_parser(
	"seir",
	description="Run a SEIR (susceptible, exposed, infected, recovered) epidemic "
	"modelling simulation showing the interactions of a single population group "
	"affected by one or more infection strains.",
	help="model a susceptible, exposed, infected, and recovered epidemic",
	formatter_class=SetWidthHelpFormatter,
)
seir_parser.add_argument(
	"-b",
	"--birth-rate",
	help="environmental static birth rate for the modelled system; the rate is applied "
	"per time-step (defaults to 0.03)",
	type=float,
	default=0.03,
)
seir_parser.add_argument(
	"-d",
	"--death-rate",
	help="environmental static death rate for the modelled system; the rate is applied "
	"per time-step (defaults to 0.03)",
	type=float,
	default=0.011,
)
seir_parser.add_argument(
	"-k",
	"--max-population",
	help="a maximum limit to the number of people considered within the simulation "
	"model; the combination of all population groups cannot exceed the value (defaults "
	"to 100,000)",
	type=int,
	default=100_000,
)
seir_parser.add_argument(
	"-n",
	"--num-days",
	help="number of days that the interval is set to run; time-step interactions are "
	"run per day time-step and is dependent on system limitations (defaults to 365)",
	type=float,
	default=365.0,
)
seir_parser.add_argument(
	"-t",
	"--time-step",
	help="a fractional day time-step delta on which updates are performed for the "
	"modelled system; a slower granularity removes the impact of perturbations in the "
	"modelled results (defaults to 1.0)",
	type=float,
	default=1.0,
)
seir_parser.add_argument(
	"-s",
	"--num-susceptible",
	help="starting number of susceptible people, a susceptible person can be infected "
	"by a disease so the value should be equal to the number of people in the maximum "
	"population that are not immune (defaults to 100,000)",
	type=int,
	default=100_000,
)
seir_parser.add_argument(
	"-r",
	"--num-recovered",
	help="starting number of recovered people, a recovered person cannot be infected by "
	"any infection strain a second time and are considered immune (defaults to 0)",
	type=int,
	default=0,
)
seir_parser.add_argument(
	"-i",
	"--infection",
	help="one or more infection strains; requires four fields corresponding to the "
	"starting number of infected people, the rate of infection per time-step, the "
	"virulence rate per time-step, and the recover rate per time-step; all values are "
	"needed and more than one strain can be specified to allow for more complex "
	"simulation environments (defaults to 1.0, 1.0, 1.0, 1.0)",
	type=float,
	nargs=4,
	action="append",
	default=[[1.0, 1.0, 1.0, 1.0]],
	metavar=("NUM_INFECTED", "INFECTION_RATE", "VIRULENCE_RATE", "RECOVERY_RATE"),
)


def main() -> None:
	"""Main runner for executing a target simulation."""

	# Parse the runtime environment and log relevant information for the
	# current execution of the simulation.
	args: argparse.Namespace = parser.parse_args()
	setup_logging(args.verbose)
	logging.info(f"Selected simulation: {args.simulation}")
	logging.debug("Configuration loaded from parsed arguments:")
	for k, v in args._get_kwargs():
		logging.debug(f"\t{k.rjust(16)} = {v}")
	logging.debug("Setting up the simulation runner...")

	# Select the approriate simulation setting and send off the config
	# for the simulation to be run.
	if args.simulation == "seir":
		logging.info("Launching seir simulation...")
		if len(args.infection) > 1:
			args.infection.pop(0)
		run_seir_simulation(
			args.birth_rate,
			args.death_rate,
			args.max_population,
			args.num_days,
			args.time_step,
			args.num_susceptible,
			args.num_recovered,
			args.infection if isinstance(args.infection, list) else [args.infection],
		)
	else:
		raise ValueError(f"Invalid simulation type provided: '{args.simulation}'.")


if __name__ == "__main__":
	main()
