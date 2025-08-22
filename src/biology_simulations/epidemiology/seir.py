#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from pydantic import BaseModel, ConfigDict
import pandas as pd
import numpy as np

from biology_simulations.models import default_model_config


class SEIRInfection(BaseModel):
	"""
	TODO
	"""

	model_config = ConfigDict(title="SEIR Infection", **default_model_config)

	name: str
	"""
	TODO
	"""
	number_of_infected: float
	"""
	Number of infected
	"""
	infection_rate: float
	"""
	Infection rate
	"""
	virulence_rate: float
	"""
	Virulence
	"""
	recovery_rate: float
	"""
	Recovery rate
	"""


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
	infections: list[SEIRInfection]
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
		"""
		Docstring for dRdt

		:param death_rate: Description
		:param R: Description
		:param gamma: Description
		:param I: Description
		:return: Description
		:rtype: Any
		"""

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
	SEIRInfection(
		name="",
		number_of_infected=100e3,
		infection_rate=0.2,
		virulence_rate=0.091,
		recovery_rate=0.0001,
	)
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

infections: list[SEIRInfection] = [
	SEIRInfection(
		name="",
		number_of_infected=100e3,
		infection_rate=0.2,
		virulence_rate=0.091,
		recovery_rate=0.0001,
	),
	SEIRInfection(
		name="",
		number_of_infected=1.0,
		infection_rate=0.2,
		virulence_rate=0.069,
		recovery_rate=0.0022,
	),
	SEIRInfection(
		name="",
		number_of_infected=1.0,
		infection_rate=0.2,
		virulence_rate=0.051,
		recovery_rate=0.0042,
	),
	SEIRInfection(
		name="",
		number_of_infected=1.0,
		infection_rate=0.2,
		virulence_rate=0.039,
		recovery_rate=0.01,
	),
	SEIRInfection(
		name="",
		number_of_infected=1.0,
		infection_rate=0.2,
		virulence_rate=0.025,
		recovery_rate=0.0169,
	),
	SEIRInfection(
		name="",
		number_of_infected=1.0,
		infection_rate=0.2,
		virulence_rate=0.008,
		recovery_rate=0.0301,
	),
]

betas = np.array([i.infection_rate for i in infections])
alphas = np.array([i.virulence_rate for i in infections])
gammas = np.array([i.recovery_rate for i in infections])

number_of_recovered = np.sum(betas / (alphas + gammas + 0.011))
# df2 = SEIR(
# 	a=0.03,
# 	b=0.011,
# 	N=365.0,
# 	K=200e6,
# 	dt=1.0,
# 	S0=200e6 - np.sum([x.n for x in infections]) - R0,
# 	I0=infections,
# 	R0=R0,
# ).simulation()
