#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
from dataclasses import dataclass


from pydantic import BaseModel, ConfigDict
import pandas as pd
import numpy as np


@dataclass
class Infection:
	num_infected: float
	infection_rate: float
	virulence_rate: float
	recovery_rate: float

	def __post_init__(self) -> None:
		if self.num_infected < 0:
			raise ValueError("Infection cannot have negative num_infected.")
		if self.infection_rate < 0:
			raise ValueError("Infection cannot have negative infection_rate.")
		if self.virulence_rate < 0:
			raise ValueError("Infection cannot have negative virulence_rate.")
		if self.recovery_rate < 0:
			raise ValueError("Infection cannot have negative recovery_rate.")


class SEIR(BaseModel):
	"""
	TODO
	"""

	model_config = ConfigDict(
		title="SEIR Simulation",
		str_min_length=1,
		str_strip_whitespace=True,
		arbitrary_types_allowed=True,
		use_attribute_docstrings=True,
	)

	reproduction_rate: float
	"""
	Reproduction rate.
	"""
	death_rate: float
	"""
	Death rate.
	"""
	maximum_population_size: float
	"""
	Maximum population.
	"""
	number_of_days: float
	"""
	Number of days.
	"""
	delta_timestep: float
	"""
	Delta step.
	"""
	number_of_susceptible: float
	"""
	Number of susceptible.
	"""
	infections: list[Infection]
	"""
	Infection list.
	"""
	number_of_recovered: float
	"""
	Number of recovered.
	"""

	@staticmethod
	def dSdt(death_rate, reproduction_rate, maximum_population_size, S, beta, I, size):
		return (
			reproduction_rate * size * (1 - size / maximum_population_size)
			- death_rate * S
			- np.sum(beta * I * S / size)
		)

	@staticmethod
	def dIdt(death_rate, gamma, alpha, S, beta, I, size):
		return -death_rate * I - alpha * I + beta * I * S / size - gamma * I

	@staticmethod
	def dRdt(death_rate, R, gamma, I):
		return -death_rate * R + np.sum(gamma * I)

	def simulation(self):
		"""Runs the SEIR simulation and returns the
		results as a Pandas DataFrame.

		The simulation calculates the number of
		susceptible, infected, and recovered individuals
		over time, based on the provided parameters.

		:return: A Pandas DataFrame containing the
			simulation results. The DataFrame has columns
			for 'Susceptible', 'Recovered', 'Time', and
			each infection strain, as well as a 'Total'
			column representing the sum of all individuals
			(S + I + R).
		:rtype: pd.DataFrame
		"""
		# Set simulation constants
		steps = int(self.number_of_days / self.delta_timestep)
		strains = len(self.infections)

		# Initialise parameter wrappers
		beta = np.array([i.infection_rate for i in self.infections], dtype=float)
		alpha = np.array([i.virulence_rate for i in self.infections], dtype=float)
		gamma = np.array([i.recovery_rate for i in self.infections], dtype=float)

		# Initialise result arrays
		S = np.zeros(steps, dtype=float)
		I = np.zeros((steps, strains), dtype=float)
		R = np.zeros(steps, dtype=float)
		T = np.zeros(steps, dtype=float)

		# Set starting values
		S[0] = self.number_of_susceptible
		I[0] = [i.number_of_infected for i in self.infections]
		R[0] = self.number_of_recovered

		# Cycle through each time step to iterate
		for ind in range(1, steps):
			# Calculate the derivatives
			size = S[ind - 1] + np.sum(I[ind - 1]) + R[ind - 1]
			dS = SEIR.dSdt(
				self.death_rate,
				self.reproduction_rate,
				self.maximum_population_size,
				S[ind - 1],
				beta,
				I[ind - 1],
				size,
			)
			dI = SEIR.dIdt(
				self.death_rate, gamma, alpha, S[ind - 1], beta, I[ind - 1], size
			)
			dR = SEIR.dRdt(self.death_rate, R[ind - 1], gamma, I[ind - 1])
			# Update each results group
			S[ind] = S[ind - 1] + self.delta_timestep * dS
			I[ind] = I[ind - 1] + self.delta_timestep * dI
			R[ind] = R[ind - 1] + self.delta_timestep * dR
			T[ind] = T[ind - 1] + self.delta_timestep

		# Create a dataframe to return
		df = pd.DataFrame()
		df["Susceptible"] = S
		df["Recovered"] = R
		df["Time"] = T
		for i, infection in enumerate(self.infections):
			df[f"{infection.name}"] = I.T[i]
		df["Total"] = df.loc[:, df.columns != "Time"].sum(axis=1)
		return df


infections = [
	# Infection(
	# 	name="",
	# 	number_of_infected=100e3,
	# 	infection_rate=0.2,
	# 	virulence_rate=0.091,
	# 	recovery_rate=0.0001,
	# )
]
number_of_recovered = 0.2 / (0.091 + 0.0001 + 0.011)
df1 = SEIR(
	reproduction_rate=0.03,
	death_rate=0.011,
	number_of_days=365.0,
	maximum_population_size=200e6,
	delta_timestep=1.0,
	number_of_susceptible=200e6
	- np.sum([x.number_of_infected for x in infections])
	- number_of_recovered,
	infections=infections,
	number_of_recovered=number_of_recovered,
).simulation()


def run_seir_simulation(
	birth_rate: float,
	death_rate: float,
	max_population: int,
	num_days: float,
	time_step: float,
	num_susceptible: int,
	num_recovered: int,
	infections: list[tuple[float, float, float]],
) -> None:
	logging.info("Starting up seir simulation runtime")
	logging.debug("Validating simulation parameters and building environment")
	infections = [Infection(*x) for x in infections]
	logging.debug("Validating simulation artifact location")
	logging.debug("Running simulation...")
	logging.debug("Prepating simulation outputs and summaries")
	logging.debug("Output simulation summaries and save output artifacts")
	logging.info("Simulation completed successfully!")
