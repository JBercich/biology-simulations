#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging


def setup_logging(verbose: bool = False) -> None:
	"""Configures the root logger with a custom setup.

	Args:
	    verbose (bool): Verbosity flag for the runtime environment will
	        determine the log-level showing INFO and DEBUG or WARNING
			and above.
	"""
	log_format = "%(asctime)s [%(levelname)-8s] %(filename)16s:%(lineno)d - %(message)s"
	formatter = logging.Formatter(log_format)
	stream_handler: logging.StreamHandler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger: logging.Logger = logging.getLogger()
	logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
	logger.addHandler(stream_handler)
