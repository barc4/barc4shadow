from __future__ import annotations

import numpy as np


def s4_beamline_to_layout(beamline) -> dict:
    """
    Convert a SHADOW4 S4Beamline into a normalized beamline layout dictionary.
    """
    from shadow4.beamline.optical_elements.absorbers.s4_screen import S4Screen
    from shadow4.beamline.optical_elements.compound.s4_compound import S4Compound
    from shadow4.beamline.optical_elements.crystals.s4_crystal import S4Crystal
    from shadow4.beamline.optical_elements.gratings.s4_grating import S4Grating
    from shadow4.beamline.optical_elements.ideal_elements.s4_empty import S4Empty
    from shadow4.beamline.optical_elements.ideal_elements.s4_ideal_lens import S4IdealLens
    from shadow4.beamline.optical_elements.mirrors.s4_mirror import S4Mirror
    from shadow4.beamline.optical_elements.multilayers.s4_multilayer import S4Multilayer
    from shadow4.beamline.optical_elements.refractors.s4_crl import S4CRL
    from shadow4.beamline.optical_elements.refractors.s4_lens import S4Lens
    from shadow4.beamline.optical_elements.refractors.s4_transfocator import S4Transfocator

    def _kind_from_oe(oe) -> str:
        if isinstance(oe, S4Empty):
            return "E"
        if isinstance(oe, S4Multilayer):
            return "ML"
        if isinstance(oe, S4Mirror):
            return "M"
        if isinstance(oe, S4Crystal):
            return "C"
        if isinstance(oe, S4Grating):
            return "G"
        if isinstance(oe, S4Screen):
            boundary_shape = oe.get_boundary_shape()
            if oe._i_abs > 0:
                return "F"      # filter
            if boundary_shape is None:
                return "O"      # screen
            if oe._i_stop == 0:
                return "SL"     # slit/aperture
            if oe._i_stop == 1:
                return "BS"     # obstruction
            return "O"
        if isinstance(oe, (S4CRL, S4Transfocator)):
            return "CRL"
        if isinstance(oe, (S4Lens,S4IdealLens)):
            return "L"
        if isinstance(oe, S4Compound):
            return "CMP"
        return "O"

    positions = beamline.syspositions()
    n_oe = beamline.get_beamline_elements_number()

    if n_oe == 0:
        return {
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "elements": {
                "labels": ["Source"],
                "kinds": ["SRC"],
            },
            "meta": {
                "source": "shadow4",
                "n_oe": 0,
                "has_final_image": False,
            },
        }

    labels = ["Source"]
    kinds = ["SRC"]

    for i, element in enumerate(beamline.get_beamline_elements()):
        oe = element.get_optical_element()
        # print(f"OE {i + 1}: {oe.get_name()} ({_kind_from_oe(oe)})")

        try:
            label = str(oe.get_name()).strip()
        except Exception:
            label = f"OE {i + 1}"

        labels.append(label if label else f"OE {i + 1}")
        kinds.append(_kind_from_oe(oe))

    x = np.asarray(positions["optical_axis_x"], dtype=float)
    y = np.asarray(positions["optical_axis_y"], dtype=float)
    z = np.asarray(positions["optical_axis_z"], dtype=float)

    mirr = np.asarray(positions["mirr"], dtype=float)
    star = np.asarray(positions["star"], dtype=float)

    last_mirr = mirr[:, -1]
    last_star = star[:, -1]

    append_final_image = not np.allclose(last_mirr, last_star)

    if append_final_image:
        labels.append("Final image")
        kinds.append("O")
    else:
        x = x[:-1]
        y = y[:-1]
        z = z[:-1]

    return {
        "x": x,
        "y": y,
        "z": z,
        "elements": {
            "labels": labels,
            "kinds": kinds,
        },
        "meta": {
            "source": "shadow4",
            "n_oe": n_oe,
            "has_final_image": append_final_image,
        },
    }