#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class Infection:
	"""Infection namespace container."""

	num_infected: float
	infection_rate: float
	virulence_rate: float
	recovery_rate: float

	def __post_init__(self) -> None:
		# Validation of the infection dataclass fields.
		if self.num_infected < 0:
			raise ValueError("Infection cannot have negative num_infected.")
		if self.infection_rate < 0:
			raise ValueError("Infection cannot have negative infection_rate.")
		if self.virulence_rate < 0:
			raise ValueError("Infection cannot have negative virulence_rate.")
		if self.recovery_rate < 0:
			raise ValueError("Infection cannot have negative recovery_rate.")


class SEIR:
	"""Susceptible-exposed-infected-recovered simulation system. The
	class contains the environment required for a continuous computer
	simulation provided by a time-step based approach to simulate
	the SEIR model.

	Args:
		birth_rate (float): Birth rate of the population.
		death_rate (float): Death rate of the population.
		max_population (int): Maximum population size.
		num_days (float): Number of days to run the simulation.
		time_step (float): Time step for each simulation iteration.
		num_susceptible (int): Initial number of susceptible
			individuals.
		num_recovered (int): Initial number of recovered individuals.
		infections (list[tuple[float, float, float, float]]): List of
			infections with their respective parameters: num_infected,
			infection_rate, virulence_rate, recovery_rate.
	"""

	def __init__(
		self,
		birth_rate: float,
		death_rate: float,
		max_population: int,
		num_days: float,
		time_step: float,
		num_susceptible: int,
		num_recovered: int,
		infections: list[tuple[float, float, float, float]],
	) -> None:
		self.birth_rate: float = birth_rate
		self.death_rate: float = death_rate
		self.max_population: int = max_population
		self.num_days: float = num_days
		self.time_step: float = time_step
		self.num_susceptible: int = num_susceptible
		self.num_recovered: int = num_recovered
		self.infections: list[Infection] = [Infection(*i) for i in infections]

	def step_num_susceptible(
		self,
		last_susceptible: NDArray[Any],
		last_infected: NDArray[Any],
		infection_rates: NDArray[Any],
		population_size: int,
	) -> Any:
		"""Calculate the change in the number of susceptible individuals
		for the next time step based on the current state of the
		simulation.

		Args:
			last_susceptible (NDArray[Any]): Last known number of
				susceptible individuals.
			last_infected (NDArray[Any]): Last known number of
				infected individuals.
			infection_rates (NDArray[Any]): Infection rates for each
				infection strain.
			population_size (int): Total population size.

		Returns:
			Any: The change in the number of susceptible individuals
				for the next time step.
		"""
		return (
			self.birth_rate
			* population_size
			* (1 - population_size / self.max_population)
			- self.death_rate * last_susceptible
			- np.sum(
				infection_rates * last_infected * last_susceptible / population_size
			)
		)

	def step_num_infected(
		self,
		last_susceptible: NDArray[Any],
		last_infected: NDArray[Any],
		infection_rates: NDArray[Any],
		virulence_rates: NDArray[Any],
		recovery_rates: NDArray[Any],
		population_size: int,
	) -> Any:
		"""Calculate the change in the number of infected individuals
		for the next time step based on the current state of the
		simulation.

		Args:
			last_susceptible (NDArray[Any]): Last known number of
				susceptible individuals.
			last_infected (NDArray[Any]): Last known number of
				infected individuals.
			infection_rates (NDArray[Any]): Infection rates for each
				infection strain.
			virulence_rates (NDArray[Any]): Virulence rates for each
				infection strain.
			recovery_rates (NDArray[Any]): Recovery rates for each
				infection strain.
			population_size (int): Total population size.

		Returns:
			Any: The change in the number of infected individuals for
				the next time step.
		"""
		return (
			-self.death_rate * last_infected
			- virulence_rates * last_infected
			+ infection_rates * last_infected * last_susceptible / population_size
			- recovery_rates * last_infected
		)

	def step_num_recovered(
		self,
		last_infected: NDArray[Any],
		last_recovered: NDArray[Any],
		recovery_rates: NDArray[Any],
	) -> Any:
		"""Calculate the change in the number of recovered individuals
		for the next time step based on the current state of the
		simulation.

		Args:
			last_infected (NDArray[Any]): Last known number of
				infected individuals.
			last_recovered (NDArray[Any]): Last known number of
				recovered individuals.
			recovery_rates (NDArray[Any]): Recovery rates for each
				infection strain.

		Returns:
			Any: The change in the number of recovered individuals for
				the next time step.
		"""
		return -self.death_rate * last_recovered + np.sum(recovery_rates * last_infected)

	def run(self, float_dtype: Any = np.float32) -> pd.DataFrame:
		"""Run the SEIR simulation and return the results as a pandas
		DataFrame.

		Args:
			float_dtype (Any): The numpy data type to use for floating
				point numbers in the simulation. Defaults to np.float32.
		"""
		# Prepare the simulation environment infection values to collect
		# into numpy arrays to allow for optimised array operations with
		# numpy on each time step.
		num_steps: int = int(self.num_days / self.time_step)
		logging.debug(f"SEIR simulation: Preparting to run {num_steps} steps...")
		infection_rates: NDArray[Any] = np.array(
			[i.infection_rate for i in self.infections],
			dtype=float_dtype,
		)
		virulence_rates: NDArray[Any] = np.array(
			[i.virulence_rate for i in self.infections],
			dtype=float_dtype,
		)
		recovery_rates: NDArray[Any] = np.array(
			[i.recovery_rate for i in self.infections],
			dtype=float_dtype,
		)

		# Setup numpy arrays for result sets in the different recorded
		# time intervals for each simulation step. Initial values can be
		# set for each population group.
		strains: int = len(self.infections)
		r_susceptible: NDArray[Any] = np.zeros(num_steps, dtype=float_dtype)
		r_infected: NDArray[Any] = np.zeros((num_steps, strains), dtype=float_dtype)
		r_recovered: NDArray[Any] = np.zeros(num_steps, dtype=float_dtype)
		r_time_step: NDArray[Any] = np.zeros(num_steps, dtype=float_dtype)
		r_susceptible[0] = self.num_susceptible
		r_infected[0] = [i.num_infected for i in self.infections]
		r_recovered[0] = self.num_recovered

		# Cycle through each time step and update the population groups
		# in individual array operations. The intervals run on provided
		# time step amounts until the simulation is completed.
		N: int
		for i in range(1, num_steps):
			N = r_susceptible[i - 1] + np.sum(r_infected[i - 1]) + r_recovered[i - 1]
			step_susceptible: Any = self.step_num_susceptible(
				r_susceptible[i - 1],
				r_infected[i - 1],
				infection_rates,
				N,
			)
			step_infected: Any = self.step_num_infected(
				r_susceptible[i - 1],
				r_infected[i - 1],
				infection_rates,
				virulence_rates,
				recovery_rates,
				N,
			)
			step_recovered: Any = self.step_num_recovered(
				r_infected[i - 1],
				r_recovered[i - 1],
				recovery_rates,
			)
			r_susceptible[i] = r_susceptible[i - 1] + self.time_step * step_susceptible
			r_infected[i] = r_infected[i - 1] + self.time_step * step_infected
			r_recovered[i] = r_recovered[i - 1] + self.time_step * step_recovered
			r_time_step[i] = r_time_step[i - 1] + self.time_step
			if i % 100000 == 0:
				logging.debug(
					f"SEIR simulation: Completed {i} of {num_steps} steps "
					f"({(i / num_steps) * 100:.2f}%)"
				)

		# Format data results into a pandas dataframe with infections
		# in a wide-format so the time-step can be the index-key.
		df = pd.DataFrame(dtype=float_dtype)
		df["Timestamp"] = r_time_step
		df["NumberOfSusceptiblePeople"] = r_susceptible
		df["NumberOfRecoveredPeople"] = r_recovered
		for i, _ in enumerate(self.infections):
			df[f"NumberOfInfectedPeople_{i}"] = r_infected.T[i]
		df["TotalPopulationSize"] = df.loc[:, df.columns != "Timestamp"].sum(axis=1)
		return df


def run_seir_simulation(
	birth_rate: float,
	death_rate: float,
	max_population: int,
	num_days: float,
	time_step: float,
	num_susceptible: int,
	num_recovered: int,
	infections: list[tuple[float, float, float, float]],
) -> None:
	"""Run the SEIR simulation with the provided parameters. This
	function serves as the entry point for running the simulation and
	handles logging and validation of parameters.

	Args:
		birth_rate (float): Birth rate of the population.
		death_rate (float): Death rate of the population.
		max_population (int): Maximum population size.
		num_days (float): Number of days to run the simulation.
		time_step (float): Time step for each simulation iteration.
		num_susceptible (int): Initial number of susceptible
			individuals.
		num_recovered (int): Initial number of recovered individuals.
		infections (list[tuple[float, float, float, float]]): List of
			infections with their respective parameters: num_infected,
			infection_rate, virulence_rate, recovery_rate.

	Returns:
		None: This function does not return any value. It runs the
			simulation and logs the results; data is saved to a file in
			the configured output directory denoted by a timestamp.
	"""
	logging.info("SEIR simulation: Starting up simulation runtime...")

	# Validate the simulation parameters to ensure they are within
	# acceptable ranges and types. Raise ValueError if any parameter
	# is invalid. This helps to ensure that the simulation runs with
	# valid parameters and avoids runtime errors during the simulation.
	logging.debug("SEIR simulation: Validating simulation parameters...")
	if birth_rate < 0:
		raise ValueError("Birth rate cannot be negative.")
	if death_rate < 0:
		raise ValueError("Death rate cannot be negative.")
	if max_population <= 0:
		raise ValueError("Maximum population must be greater than zero.")
	if num_days <= 0:
		raise ValueError("Number of days must be greater than zero.")
	if time_step <= 0:
		raise ValueError("Time step must be greater than zero.")
	if num_susceptible < 0:
		raise ValueError("Number of susceptible individuals cannot be negative.")
	if num_recovered < 0:
		raise ValueError("Number of recovered individuals cannot be negative.")
	if not infections:
		raise ValueError("Infections list cannot be empty.")
	logging.info("SEIR simulation: Running the simulation...")

	# Run the SEIR simulation with the provided parameters. Calulate the
	# time taken to run the simulation and log the results.
	start_time: float = time.time()
	df: pd.DataFrame = SEIR(
		birth_rate,
		death_rate,
		max_population,
		num_days,
		time_step,
		num_susceptible,
		num_recovered,
		infections,
	).run()
	end_time: float = time.time()
	elapsed_time: float = end_time - start_time
	logging.info(f"SEIR simulation: Simulation completed in {elapsed_time:.2f} seconds.")
	logging.debug("SEIR simulation: Simulation run completed, preparing outputs...")

	# Prepare the output DataFrame and save it to a file.
	logging.info("Simulation completed, preparing outputs...")
	output_dir: str = "./out/seir"
	output_pth: str = f"{output_dir}/{int(time.time())}.csv"
	os.makedirs(output_dir, exist_ok=True)
	df.to_csv(output_pth, index=False)
	logging.info(f"Simulation output saved to {output_pth}")
