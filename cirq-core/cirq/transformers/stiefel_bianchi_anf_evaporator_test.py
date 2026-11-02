# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");

import numpy as np
try:
    from cirq.transformers.stiefel_bianchi_anf_evaporator import (
        fast_zeta_xor_transform,
        integer_rota_mobius_transform,
        smooth_witt_bitplane_lift,
        solve_one_shot_stiefel_bianchi_gauge,
    )
except ImportError:
    try:
        from qualtran_dropin.stiefel_bianchi_anf_evaporator import (
            fast_zeta_xor_transform,
            integer_rota_mobius_transform,
            smooth_witt_bitplane_lift,
            solve_one_shot_stiefel_bianchi_gauge,
        )
    except ImportError:
        from stiefel_bianchi_anf_evaporator import (
            fast_zeta_xor_transform,
            integer_rota_mobius_transform,
            smooth_witt_bitplane_lift,
            solve_one_shot_stiefel_bianchi_gauge,
        )


def test_one_shot_stiefel_bianchi_100_percent_gate_evaporation():
    k_bits = 6
    n_rows = 1 << k_bits
    n_copies = 16
    high_masks = [3, 5, 6, 9, 10, 12, 17, 18, 20, 24, 33, 34, 36, 40, 48, 7]

    w_clean = np.zeros((n_rows, n_copies), dtype=np.float64)
    for c_i in range(n_copies):
        b_i = c_i % k_bits
        m_hi = high_masks[c_i]
        for a in range(n_rows):
            bit_val = float((a >> b_i) & 1)
            and_val = 1.0 if ((a & m_hi) == m_hi) else 0.0
            w_clean[a, c_i] = 4.0 * bit_val + 16.0 * and_val

    q_init = np.eye(n_copies, dtype=np.float64)
    for p in range(n_copies):
        for q in range(p + 1, n_copies):
            ang = 0.25 * np.sin(p * 3.0 + q * 7.0 + 1.0)
            co, si = np.cos(ang), np.sin(ang)
            g_rot = np.eye(n_copies, dtype=np.float64)
            g_rot[p, p], g_rot[q, q] = co, co
            g_rot[p, q], g_rot[q, p] = -si, si
            q_init = q_init @ g_rot
    w_entangled = w_clean @ q_init

    f_scrambled = np.round(w_entangled).astype(np.int64)
    nl_mask = np.array([int(m).bit_count() >= 2 for m in range(n_rows)], dtype=bool)
    nl_gates_before = 0
    for b in range(4):
        anf_b = fast_zeta_xor_transform((f_scrambled >> b) & 1)
        nl_gates_before += int(np.sum(anf_b[nl_mask]))
    assert nl_gates_before == 1754

    q_opt, info = solve_one_shot_stiefel_bianchi_gauge(w_entangled)
    assert info["sym2_dim"] == 136.0
    assert info["max_rank1_residual"] < 1e-12
    assert info["slice_invariance_error"] < 1e-14

    w_recovered = np.round(w_entangled @ q_opt).astype(np.int64)
    nl_gates_after = 0
    for b in range(4):
        anf_b = fast_zeta_xor_transform((w_recovered >> b) & 1)
        nl_gates_after += int(np.sum(anf_b[nl_mask]))
    assert nl_gates_after == 0


def test_smooth_witt_bitplane_lift_and_2adic_carry_trap():
    ints = np.arange(0, 16, dtype=np.float64)
    for b in (0, 1):
        lifted = smooth_witt_bitplane_lift(ints, b)
        exact_bits = ((ints.astype(np.int64) >> b) & 1).astype(np.float64)
        np.testing.assert_allclose(lifted, exact_bits, atol=1e-14)

    # Verify the 2-adic inter-plane carry trap witness at m=3:
    f0 = np.array([0.0, 1.0, 1.0, 0.0])
    f1 = np.array([0.0, 0.0, 0.0, 1.0])
    f_comb = f0 + 2.0 * f1
    mu_f0 = integer_rota_mobius_transform(f0)
    mu_f1 = integer_rota_mobius_transform(f1)
    mu_comb = integer_rota_mobius_transform(f_comb)
    assert mu_comb[3] == 0.0 and mu_f0[3] == -2.0 and mu_f1[3] == 1.0
