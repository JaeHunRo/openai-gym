"""Demo Learning Agents for OpenAI Gym Problems.

To run:
    python main.py

"""


import argparse
import sys


def process_args():
    """Process command line args."""
    message = 'OpenAI Gym Implementations.'
    parser = argparse.ArgumentParser(description=message)

    default_env = 'cartpole'

    parser.add_argument('-e', '--environment',
                        dest='env',
                        help='Environment choice.',
                        default=default_env)

    options = parser.parse_args()
    return options


def run_cartpole():
    """Train and run cart pole agent."""
    print('cartpole')


def main():
    """Entry point."""
    options = process_args()
    if options.env == 'cartpole':
        run_cartpole()
    else:
        print('Environment choice {} is currently unsupported.'
              .format(options.env))
        sys.exit(1)


if __name__ == '__main__':
    main()
