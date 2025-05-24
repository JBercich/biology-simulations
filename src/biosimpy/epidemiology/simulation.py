#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from dataclasses import dataclass

import numpy as np
import polars as pl

from biosimpy.epidemiology.models import SEIR, Environment, Infection, Strain


@dataclass
class SEIRSimulation:
	environment: Environment
	start_conditions: SEIR

	@staticmethod
	def step_num_susceptible(
		env: Environment,
		population_size: float,
		num_susceptible: float,
		num_infected: np.ndarray,
		strain_infection_rate: np.ndarray,
	) -> np.ndarray:
		return (
			env.reproduction_rate
			* population_size
			* (1 - population_size / env.maximum_population)
			- env.fatality_rate * num_susceptible
			- np.sum(
				strain_infection_rate * num_infected * num_susceptible / population_size
			)
		)

	@staticmethod
	def step_num_infected(
		env: Environment,
		population_size: float,
		num_susceptible: float,
		num_infected: np.ndarray,
		strain_infection_rate: np.ndarray,
		strain_virulence: np.ndarray,
		strain_recovery_rate: np.ndarray,
	) -> np.ndarray:
		return (
			-env.fatality_rate * num_infected
			- strain_virulence * num_infected
			+ strain_infection_rate * num_infected * num_susceptible / population_size
			- strain_recovery_rate * num_infected
		)

	@staticmethod
	def step_num_recovered(
		env: Environment,
		num_infected: np.ndarray,
		num_recovered: float,
		strain_recovery_rate: np.ndarray,
	):
		return -env.fatality_rate * num_recovered + np.sum(
			strain_recovery_rate * num_infected
		)

	def run(self, num_steps: int, time_step: float) -> pl.DataFrame:
		strains: list[Strain] = [s.strain for s in self.start_conditions.infections]
		infection_rate: np.ndarray = np.array([s.infection_rate for s in strains])
		virulence: np.ndarray = np.array([s.virulence for s in strains])
		recovery_rate: np.ndarray = np.array([s.recovery_rate for s in strains])

		num_susceptible: np.ndarray = np.zeros(num_steps, dtype=float)
		num_infected: np.ndarray = np.zeros((num_steps, len(strains)), dtype=float)
		num_recovered: np.ndarray = np.zeros(num_steps, dtype=float)
		time_steps: np.ndarray = np.zeros(num_steps, dtype=float)

		num_susceptible[0] = self.start_conditions.num_susceptible
		num_infected[0] = [s.num_infected for s in self.start_conditions.infections]
		num_recovered[0] = self.start_conditions.num_recovered

		for i in range(1, num_steps):
			population_size: float = (
				num_susceptible[i - 1]
				+ np.sum(num_infected[i - 1])
				+ num_recovered[i - 1]
			)

			step_susceptible: np.ndarray = self.step_num_susceptible(
				self.environment,
				population_size,
				num_susceptible[i - 1],
				num_infected[i - 1],
				infection_rate,
			)
			step_infected: np.ndarray = self.step_num_infected(
				self.environment,
				population_size,
				num_susceptible[i - 1],
				num_infected[i - 1],
				infection_rate,
				virulence,
				recovery_rate,
			)
			step_recovered: np.ndarray = self.step_num_recovered(
				self.environment,
				num_infected[i - 1],
				num_recovered[i - 1],
				recovery_rate,
			)

			num_susceptible[i] = num_susceptible[i - 1] + time_step * step_susceptible
			num_infected[i] = num_infected[i - 1] + time_step * step_infected
			num_recovered[i] = num_recovered[i - 1] + time_step * step_recovered
			time_steps[i] = time_steps[i - 1] + time_step

		return pl.DataFrame(
			{f"strains_{i}": num_infected.T[i] for i in range(len(strains))}
			| {
				"susceptible": num_susceptible,
				"recovered": num_recovered,
				"time": time_steps,
			}
		)


import time

t0 = time.time()
result: pl.DataFrame = SEIRSimulation(
	Environment(0.1, 0.1, 1000.0),
	SEIR(
		[
			Infection(Strain(0.1, 0.2, 0.3), 100.0),
			Infection(Strain(0.01, 0.02, 0.03), 200.0),
		],
		1000.0,
		1000.0,
	),
).run(100000, 0.1)

print(result)
print(time.time() - t0)
