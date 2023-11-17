#!/usr/bin/env bash

# flake8 . || exit 1 # lint check
black . --preview || exit 1 # format all documents
