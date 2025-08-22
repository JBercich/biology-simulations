from pydantic import BaseModel
import pandas as pd
import numpy as np


class SimulationBaseModel(BaseModel):
	class Config:
		frozen = True
		extra = "forbid"
		min_anystr_length = 1
		anystr_strip_whitespace = True


class Strain(SimulationBaseModel):
	name: str


s1 = Strain(name="I")
s2 = Strain(name="II")
s3 = Strain(name="IIIA")
s4 = Strain(name="IIIB")
s5 = Strain(name="IV")
s6 = Strain(name="V")
strains = [s1, s2, s3, s4, s5, s6]


class Genotype(SimulationBaseModel):
	name: str
	n: float
	b: float


g1 = Genotype(n=100.0, b=0.2, name="AA")
g2 = Genotype(n=100.0, b=0.2, name="Aa")
g3 = Genotype(n=100.0, b=0.2, name="aa")
genotypes = [g1, g2, g3]


class InfectionInteraction(SimulationBaseModel):
	n: float
	beta: float
	alpha: float
	gamma: float
	strain: Strain
	genotype: Genotype


interactions = [
	InfectionInteraction(
		strain=s1, genotype=g1, n=100.0, beta=1.0, alpha=0.2, gamma=0.0001
	),
	InfectionInteraction(
		strain=s1, genotype=g2, n=100.0, beta=1.0, alpha=0.2, gamma=0.0001
	),
	InfectionInteraction(
		strain=s1, genotype=g3, n=100.0, beta=1.0, alpha=0.2, gamma=0.0001
	),
	InfectionInteraction(
		strain=s2, genotype=g1, n=100.0, beta=1.0, alpha=0.2, gamma=0.0001
	),
	InfectionInteraction(
		strain=s2, genotype=g2, n=100.0, beta=1.0, alpha=0.2, gamma=0.0001
	),
	InfectionInteraction(
		strain=s2, genotype=g3, n=100.0, beta=1.0, alpha=0.2, gamma=0.0001
	),
	InfectionInteraction(
		strain=s3, genotype=g1, n=100.0, beta=1.0, alpha=0.2, gamma=0.0001
	),
	InfectionInteraction(
		strain=s3, genotype=g2, n=100.0, beta=1.0, alpha=0.2, gamma=0.0001
	),
	InfectionInteraction(
		strain=s3, genotype=g3, n=100.0, beta=1.0, alpha=0.2, gamma=0.0001
	),
	InfectionInteraction(
		strain=s4, genotype=g1, n=100.0, beta=1.0, alpha=0.2, gamma=0.0001
	),
	InfectionInteraction(
		strain=s4, genotype=g2, n=100.0, beta=1.0, alpha=0.2, gamma=0.0001
	),
	InfectionInteraction(
		strain=s4, genotype=g3, n=100.0, beta=1.0, alpha=0.2, gamma=0.0001
	),
	InfectionInteraction(
		strain=s5, genotype=g1, n=100.0, beta=1.0, alpha=0.2, gamma=0.0001
	),
	InfectionInteraction(
		strain=s5, genotype=g2, n=100.0, beta=1.0, alpha=0.2, gamma=0.0001
	),
	InfectionInteraction(
		strain=s5, genotype=g3, n=100.0, beta=1.0, alpha=0.2, gamma=0.0001
	),
	InfectionInteraction(
		strain=s6, genotype=g1, n=100.0, beta=1.0, alpha=0.2, gamma=0.0001
	),
	InfectionInteraction(
		strain=s6, genotype=g2, n=100.0, beta=1.0, alpha=0.2, gamma=0.0001
	),
	InfectionInteraction(
		strain=s6, genotype=g3, n=100.0, beta=1.0, alpha=0.2, gamma=0.0001
	),
]


import dataclasses, abc


@dataclasses.dataclass
class Validator(abc.ABC):
	"""Validator super class for `dataclasses.dataclass` type coercion."""

	def __post_init__(self):
		"""
		Post initialisation validation.

		After a `dataclasses.dataclass` instance is initialised and required attributes
		are assigned, this function will assert strict type checking without the use of
		libraries such as `pydantic`. This will only look at generics depth of 2.

		Raises:
		    TypeError: Parameter is given an incorrect type.
		    TypeError: Parameter is given an incorrect first-level generic.
		    TypeError: Parameter is given an incorrect second-level type.
		"""
		# Cycle through all the class type hints to validate types
		for pname, ptype in self.__annotations__.items():
			parameter_value = dataclasses.asdict(self)[pname]
			# Capture first-level generics
			if "__origin__" in dir(ptype):
				# Ensure strict typing of the first-level generic
				if not isinstance(parameter_value, ptype.__origin__):
					raise TypeError(f"`{pname}` requires `{ptype}`")
				# Ensure all internal generic values are correct as well
				if not all([isinstance(x, ptype.__args__) for x in parameter_value]):
					raise TypeError(f"`{pname}` requires `{ptype}`")
			else:
				# Ensure normal parameters have correct type
				if not isinstance(parameter_value, ptype):
					raise TypeError(f"`{pname}` requires `{ptype}`")


@dataclasses.dataclass
class Infection(Validator):
	"""Abstraction for SEIR model infections."""

	n: float  # Number of infected
	beta: float  # Infection rate
	alpha: float  # Virulence
	gamma: float  # Recovery rate

	def __post_init__(self):
		super().__post_init__()
		# Check all values are non-negative
		for pname, ptype in self.__annotations__.items():
			if dataclasses.asdict(self)[pname] < 0:
				raise ValueError(f"`{pname}` must be greater than 0")


STRAINS = (
	"Strain I",
	"Strain II",
	"Strain IIIA",
	"Strain IIIB",
	"Strain IV",
	"Strain V",
)


@dataclasses.dataclass
class SEIR:
	"""SEIR simulation abstraction for running models."""

	a: float  # Reproduction rate
	b: float  # Death rate
	K: float  # Maximum population
	N: float  # Number of days
	dt: float  # Delta step
	S0: float  # Number of susceptible
	I0: list[Infection]  # Infection list
	R0: float  # Number of recovered

	def __post_init__(self):
		# Check all floats are non-negative
		for pname, ptype in self.__annotations__.items():
			parameter_value = dataclasses.asdict(self)[pname]
			if isinstance(parameter_value, float) and parameter_value < 0:
				raise ValueError(f"`{pname}` must be greater than 0")

	@staticmethod
	def dSdt(b, a, K, S, beta, I, size):
		return a * size * (1 - size / K) - b * S - np.sum(beta * I * S / size)

	@staticmethod
	def dIdt(b, gamma, alpha, S, beta, I, size):
		return -b * I - alpha * I + beta * I * S / size - gamma * I

	@staticmethod
	def dRdt(b, R, gamma, I):
		return -b * R + np.sum(gamma * I)

	def simulation(self):
		# Set simulation constants
		steps = int(self.N / self.dt)
		strains = len(self.I0)

		# Initialise parameter wrappers
		beta = np.array([i.beta for i in self.I0], dtype=float)
		alpha = np.array([i.alpha for i in self.I0], dtype=float)
		gamma = np.array([i.gamma for i in self.I0], dtype=float)

		# Initialise result arrays
		S = np.zeros(steps, dtype=float)
		I = np.zeros((steps, strains), dtype=float)
		R = np.zeros(steps, dtype=float)
		T = np.zeros(steps, dtype=float)

		# Set starting values
		S[0] = self.S0
		I[0] = [i.n for i in self.I0]
		R[0] = self.R0

		# Cycle through each time step to iterate
		for ind in range(1, steps):
			# Calculate the derivatives
			size = S[ind - 1] + np.sum(I[ind - 1]) + R[ind - 1]
			dS = SEIR.dSdt(self.b, self.a, self.K, S[ind - 1], beta, I[ind - 1], size)
			dI = SEIR.dIdt(self.b, gamma, alpha, S[ind - 1], beta, I[ind - 1], size)
			dR = SEIR.dRdt(self.b, R[ind - 1], gamma, I[ind - 1])
			# Update each results group
			S[ind] = S[ind - 1] + self.dt * dS
			I[ind] = I[ind - 1] + self.dt * dI
			R[ind] = R[ind - 1] + self.dt * dR
			T[ind] = T[ind - 1] + self.dt

		# Create a dataframe to return
		df = pd.DataFrame()
		df["Susceptible"] = S
		df["Recovered"] = R
		df["Time"] = T
		for i in range(strains):
			df[f"{STRAINS[i]}"] = I.T[i]
		df["Total"] = df.loc[:, df.columns != "Time"].sum(axis=1)
		return df


infections = [Infection(100e3, 0.2, 0.091, 0.0001)]
R0 = 0.2 / (0.091 + 0.0001 + 0.011)
df1 = SEIR(
	a=0.03,
	b=0.011,
	N=365.0,
	K=200e6,
	dt=1.0,
	S0=200e6 - np.sum([x.n for x in infections]) - R0,
	I0=infections,
	R0=(0.2 / (0.091 + 0.0001 + 0.011)),
).simulation()

infections: list[Infection] = [
	Infection(100e3, 0.2, 0.091, 0.0001),
	Infection(1.0, 0.2, 0.069, 0.0022),
	Infection(1.0, 0.2, 0.051, 0.0042),
	Infection(1.0, 0.2, 0.039, 0.01),
	Infection(1.0, 0.2, 0.025, 0.0169),
	Infection(1.0, 0.2, 0.008, 0.0301),
]

betas = np.array([i.beta for i in infections])
alphas = np.array([i.alpha for i in infections])
gammas = np.array([i.gamma for i in infections])

R0 = np.sum(betas / (alphas + gammas + 0.011))
df2 = SEIR(
	a=0.03,
	b=0.011,
	N=365.0,
	K=200e6,
	dt=1.0,
	S0=200e6 - np.sum([x.n for x in infections]) - R0,
	I0=infections,
	R0=R0,
).simulation()
