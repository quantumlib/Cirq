"""Creates and simulates a simple circuit.
"""

import cirq

def main():

    print(cirq.google.Bristlecone)

    print(cirq.line.placement.line._line_placement_on_device(
        cirq.google.Foxtail, 10))

    print()

    print(cirq.line.placement.line._line_placement_on_device(
        cirq.google.Bristlecone, 10))


if __name__ == '__main__':
    main()
