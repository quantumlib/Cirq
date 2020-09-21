"""The script that produces bristlecone_heatmap_example.png."""

import cirq


def main():
    title = 'Two Qubit Sycamore Gate Xeb Cycle Total Error'
    value_map = {
        (cirq.GridQubit(3, 2), cirq.GridQubit(4, 2)):[0.004619111460557768],
        (cirq.GridQubit(4, 1), cirq.GridQubit(4, 2)):[0.0076079162393482835],
        (cirq.GridQubit(4, 1), cirq.GridQubit(5, 1)):[0.010323903068646778],
        (cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)):[0.00729037246947839],
        (cirq.GridQubit(4, 2), cirq.GridQubit(5, 2)):[0.008226663382640803],
        (cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)):[0.01504682356081491],
        (cirq.GridQubit(5, 0), cirq.GridQubit(5, 1)):[0.00673880216745637],
        (cirq.GridQubit(5, 1), cirq.GridQubit(5, 2)):[0.01020380985719993],
        (cirq.GridQubit(5, 1), cirq.GridQubit(6, 1)):[0.005713058677283056],
        (cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)):[0.006431698844451689],
        (cirq.GridQubit(5, 2), cirq.GridQubit(6, 2)):[0.004676551878404933],
        (cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)):[0.009471810549265769],
        (cirq.GridQubit(5, 3), cirq.GridQubit(6, 3)):[0.003834724159559072],
        (cirq.GridQubit(5, 4), cirq.GridQubit(6, 4)):[0.010423354216218345],
        (cirq.GridQubit(6, 1), cirq.GridQubit(6, 2)):[0.0062515002303844824],
        (cirq.GridQubit(6, 2), cirq.GridQubit(6, 3)):[0.005419247075412775],
        (cirq.GridQubit(6, 2), cirq.GridQubit(7, 2)):[0.02236774155039517],
        (cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)):[0.006116965562115412],
        (cirq.GridQubit(6, 3), cirq.GridQubit(7, 3)):[0.005300336755683754],
        (cirq.GridQubit(6, 4), cirq.GridQubit(6, 5)):[0.012849356290539266],
        (cirq.GridQubit(6, 4), cirq.GridQubit(7, 4)):[0.007785990142364307],
        (cirq.GridQubit(6, 5), cirq.GridQubit(7, 5)):[0.008790971346696541],
        (cirq.GridQubit(7, 2), cirq.GridQubit(7, 3)):[0.004104719338404117],
        (cirq.GridQubit(7, 3), cirq.GridQubit(7, 4)):[0.009236765681133435],
        (cirq.GridQubit(7, 3), cirq.GridQubit(8, 3)):[0.024921853294157192],
        (cirq.GridQubit(7, 4), cirq.GridQubit(7, 5)):[0.0059072812181635015],
        (cirq.GridQubit(7, 4), cirq.GridQubit(8, 4)):[0.004990546867455203],
        (cirq.GridQubit(7, 5), cirq.GridQubit(7, 6)):[0.007852170748540305],
        (cirq.GridQubit(7, 5), cirq.GridQubit(8, 5)):[0.006424831182351348],
        (cirq.GridQubit(8, 3), cirq.GridQubit(8, 4)):[0.005248674988741292],
        (cirq.GridQubit(8, 4), cirq.GridQubit(8, 5)):[0.014301577907262525],
        (cirq.GridQubit(8, 4), cirq.GridQubit(9, 4)):[0.0038720100369923904]
    }
    heatmap = cirq.InterHeatmap(value_map, title)
    heatmap.plot()


if __name__ == '__main__':
    main()
