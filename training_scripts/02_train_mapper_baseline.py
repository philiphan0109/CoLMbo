#!/usr/bin/env python3
"""Uniform-sampling CoLMbo mapper fine-tuning entry point."""

from mapper_trainer import main


if __name__ == "__main__":
    main(default_sampling="uniform")
