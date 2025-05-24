#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from biosimpy.exceptions import DependencyConflictException

variable_counter: int = 0


class Variable(ABC):
	@classmethod
	def _generate_name(cls, name: str) -> str:
		global variable_counter
		variable_counter += 1
		return f"{name if name else cls.__name__}:{variable_counter - 1}"

	def __init__(self, init: Any = 0, name: str = ""):
		self.value: Any = init
		self._type: Any = type(init)
		self._name: str = self._generate_name(name)
		self.history: List[Any] = [self.value]
		self.depends: Dict[str, Variable] = {}

	def register_variables(self, **dependencies):
		for name, variable in dependencies.items():
			if name in self.depends.keys():
				raise DependencyConflictException(name)
			self.depends[name] = variable

	def step(self, *args, **kwargs):
		self.value = self.update(*args, **kwargs, **self.depends)
		if not isinstance(self.value, self._type_value):
			raise TypeError(f"Expected type: {self._type_value}")
		self.history.append(self.value)

	@abstractmethod
	def update(self) -> Any:
		pass
