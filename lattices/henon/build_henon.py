"""The Hénon maps as Xsuite lines (.json): one turn is a sextupole kick and a rotation.

    kick: px -> px + (x^2 - y^2),  py -> py - 2 x y      (thin multipole knl = [0, 0, -2])
    rot : rotation by 2 pi qx in (x, px) and 2 pi qy in (y, py)   (beta = 1, alpha = 0: normalized coordinates)

The 2D lines are the 4D line with y = py = 0 (their qy only keeps the twiss defined). The rotation is a
FirstOrderTaylorMap because a LineSegmentMap does not survive saving and loading in xtrack 0.104.3.
"""
import math
import os
import numpy as np
import xtrack as xt

HERE = os.path.dirname(os.path.abspath(__file__))


def henon_line(qx, qy):
    rotation = np.eye(6)                                    # identity in the longitudinal plane
    for i, q in ((0, qx), (2, qy)):
        c, s = math.cos(2*math.pi*q), math.sin(2*math.pi*q)
        rotation[i:i+2, i:i+2] = [[c, s], [-s, c]]
    line = xt.Line(elements=[xt.Multipole(knl=[0., 0., -2.]), xt.FirstOrderTaylorMap(m1=rotation)],
                   element_names=['kick', 'rot'])
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, p0c=450e9)   # any reference works
    return line


if __name__ == '__main__':
    out = os.environ.get('OUT', HERE)
    for name, qx, qy in (('henon_2d_q0.2071', .2071, .28), ('henon_2d_q0.218', .218, .28),
                         ('henon_4d_q0.2071_0.28', .2071, .28)):
        henon_line(qx, qy).to_json(os.path.join(out, name+'.json'))
        print('wrote', name)
