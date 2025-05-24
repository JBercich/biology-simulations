#!/usr/bin/env python3
# -*- coding:utf-8 -*-


class GraphException(Exception):
	pass


class DependencyConflictException(GraphException):
	def __init__(self, name: str):
		self.name: str = name

	def __str__(self) -> str:
		return f"Variable already registered dependency: '{self.name}'"
