#!/usr/bin/env python3
"""Task+label weighted CoLMbo mapper fine-tuning entry point."""

from mapper_trainer import main


if __name__ == "__main__":
    main(default_sampling="weighted")
