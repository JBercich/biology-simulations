#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from dash import Dash, html

app = Dash()
app.layout = [html.Div(children="Hello Worlds")]


def run() -> None:
	app.run(debug=True)
