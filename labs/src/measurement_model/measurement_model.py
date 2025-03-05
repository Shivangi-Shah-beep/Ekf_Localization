import sympy as sp
import numpy as np

    
def measurement_model():
        # Define state variables
        x, y, theta = sp.symbols('x, y, theta')
        
        # Define camera transformation parameters
        tcx, tcy, tcz = sp.symbols('tcx, tcy, tcz')
        
        # Define landmark parameters
        xl, yl, hl, rl = sp.symbols('xl, yl, hl, rl')
        
        # Define camera intrinsic parameters
        fx, fy, cx, cy = sp.symbols('fx, fy, cx, cy')

        xc = x + sp.cos(theta) * tcx - sp.sin(theta) * tcy
        yc = y + sp.sin(theta) * tcx + sp.cos(theta) * tcy

        # Bearing angle psi
        delta_x = xl - xc
        delta_y = yl - yc

        psi = sp.atan2(delta_y, delta_x)

        x1 = xl - rl * sp.sin(psi)
        y1 = yl + rl * sp.cos(psi)
        x2 = xl + rl * sp.sin(psi)
        y2 = yl - rl * sp.cos(psi)

        p1g = sp.Matrix([[x1], [y1], [0], [1]])
        p2g = sp.Matrix([[x2], [y2], [0], [1]])
        p3g = sp.Matrix([[x2], [y2], [hl], [1]])
        p4g = sp.Matrix([[x1], [y1], [hl], [1]])

        Tmr = sp.Matrix([
            [sp.cos(theta), -sp.sin(theta), 0, x],
            [sp.sin(theta),  sp.cos(theta), 0, y],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        Tro = sp.Matrix([
            [0, 0, 1, tcx],
            [-1, 0, 0, tcy],
            [0, -1, 0, tcz],
            [0, 0, 0, 1]
        ])

        Tmo = Tmr * Tro
        Tom = sp.simplify(Tmo.inv())

        p1o = (Tom * p1g)[0:3, 0]
        p2o = (Tom * p2g)[0:3, 0]
        p3o = (Tom * p3g)[0:3, 0]
        p4o = (Tom * p4g)[0:3, 0]

        P = sp.Matrix([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])


        def project_point(p_o):
            u_v_w = P * p_o
            u = u_v_w[0] / u_v_w[2]
            v = u_v_w[1] / u_v_w[2]
            return sp.Matrix([u, v])


        p1_p = project_point(p1o)
        p2_p = project_point(p2o)
        p3_p = project_point(p3o)
        p4_p = project_point(p4o)

        # Measurement vector (z_pred)
        g = sp.Matrix([
            p1_p[0], p1_p[1],
            p2_p[0], p2_p[1],
            p3_p[0], p3_p[1],
            p4_p[0], p4_p[1]
        ])

        Jacobian_g = g.jacobian(sp.Matrix([x, y, theta]))

        g_func = sp.lambdify((x, y, theta, tcx, tcy, tcz, xl, yl, hl, rl, fx, fy, cx, cy), g)
        Jacobian_g_func = sp.lambdify((x, y, theta, tcx, tcy, tcz, xl, yl, hl, rl, fx, fy, cx, cy), Jacobian_g)
        
        return g_func, Jacobian_g_func