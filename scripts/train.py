#!/usr/bin/env python
# pylint:disable=missing-docstring
from ray.rllib.train import create_parser, run

import raylab


def main():
    raylab.register_all_agents()
    raylab.register_all_environments()
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)


if __name__ == "__main__":
    main()
