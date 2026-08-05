import brute_force_solver
import xy_model


def compare_observables(param_sets):
    print("Comparing observables between brute-force and XY model")
    print("-" * 90)

    for Jx, Jy, h, n in param_sets:
        bf_energy = brute_force_solver.energy_per_site(Jx, Jy, h, n)
        xy_energy = xy_model.energy_per_site(Jx, Jy, h, n)

        bf_mag = brute_force_solver.magnetization(Jx, Jy, h, n)
        xy_mag = xy_model.magnetization(Jx, Jy, h, n)

        bf_ent = brute_force_solver.entanglement_entropy(Jx, Jy, h, n)
        xy_ent = xy_model.entanglement_entropy(Jx, Jy, h, n)

        print(f"params: Jx={Jx}, Jy={Jy}, h={h}, n={n}")
        print(f"  energy_per_site: brute={bf_energy:.6f}, xy={xy_energy:.6f}, abs_err={abs(bf_energy - xy_energy):.6e}")
        print(f"  magnetization:   brute={bf_mag:.6f}, xy={xy_mag:.6f}, abs_err={abs(bf_mag - xy_mag):.6e}")
        print(f"  entanglement_entropy: brute={bf_ent:.6f}, xy={xy_ent:.6f}, abs_err={abs(bf_ent - xy_ent):.6e}")
        print()


if __name__ == "__main__":
    param_sets = [
        (1.0, 1.0, 0.5, 2),
        (1.2, 0.4, 0.0, 2),
        (0.7, 1.3, 1.0, 3),
        (0.5, 0.5, 1.5, 4),
        (1.5, 0.2, 0.2, 4),
        (0.8, 1.8, 0.0, 5),
        (1.0, 0.0, 2.0, 6),
        (0.3, 0.9, 0.7, 7),
        (1.1, 1.1, 1.2, 8),
        (0.6, 0.4, 0.3, 10),
    ]
    compare_observables(param_sets)
