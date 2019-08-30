#!/usr/bin/env python
from contextlib import suppress

from ray.rllib.rollout import create_parser, run

import raylab  # pylint: disable=unused-import


def main():
    with suppress(KeyboardInterrupt):
        parser = create_parser()
        args = parser.parse_args()
        run(args, parser)


if __name__ == "__main__":
    main()
