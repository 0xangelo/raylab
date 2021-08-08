#!/usr/bin/env python
# pylint:disable=missing-docstring
from contextlib import suppress

from ray.rllib.rollout import create_parser, run

import raylab


def main():
    raylab.register_all_agents()
    raylab.register_all_environments()
    with suppress(KeyboardInterrupt):
        parser = create_parser()
        args = parser.parse_args()
        run(args, parser)


if __name__ == "__main__":
    main()
