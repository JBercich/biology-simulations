#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from dataclasses import dataclass


@dataclass
class Strain:
	infection_rate: float
	virulence: float
	recovery_rate: float


@dataclass
class Infection:
	strain: Strain
	num_infected: float


@dataclass
class Environment:
	reproduction_rate: float
	fatality_rate: float
	maximum_population: float


@dataclass
class SEIR:
	infections: list[Infection]
	num_susceptible: float
	num_recovered: float
