"""Explicit public identities for the common river column arithmetic."""
TOP_METHODS = {
    "column_top.recompute_porosity": 1,
    "column_top.mix_added_porosity": 2,
}
BASE_METHODS = {
    "column_base.finite_independent": 1,
    "column_base.fixed_composition": 2,
    "column_base.copy_upper_composition": 3,
    "column_base.fixed_composition_and_thickness": 4,
}
POROSITY_METHODS = {
    "porosity.dry_bulk_density": 0,
    "porosity.frings": 1,
    "porosity.beard_weyl": 2,
    "porosity.constant": 3,
    "porosity.sand_mud_weighted": 4,
    "porosity.flow_cdryb_zero": 5,
}


def method_ids(columns):
    """Internal numeric wires never substitute for the public named choices."""
    def name(mapping, number):
        return next(k for k, v in mapping.items() if v == number)
    return dict(
        top=name(TOP_METHODS, columns.top_update),
        base=name(BASE_METHODS, columns.base_update),
        porosity=name(POROSITY_METHODS, columns.porosity),
        exchange="column_exchange.constant" if columns.exchange_thickness_m > 0 else "off")


def validate_identity(value, columns):
    expected = method_ids(columns)
    if value != expected:
        raise ValueError("Explicit column method identities differ from their numerical settings; "
                         "old identities are not migrated")
    if columns.bed == "mixed" and (columns.exchange_thickness_m or columns.lagrangian_underlayers):
        raise ValueError("A fully mixed bed cannot contain exchange or substrate layers")
    if columns.bed == "mixed" and columns.base_update != 1:
        raise ValueError("A fully mixed finite bed has no separate basal layer; "
                         "choose the finite independent rule or a layered bed")
    return expected


def topology(columns, count, *, coarse=False):
    """Ordered roles shared by admission, runtime identity and UI labels."""
    if columns.bed == "mixed":
        if count != 1 or coarse:
            raise ValueError("A fully mixed bed has exactly one layer")
        return ["mixed"]
    roles = ["active"] + (["coarse"] if coarse else [])
    if columns.exchange_thickness_m > 0:
        roles.append("exchange")
    roles.extend(f"lagrangian_{i+1}" for i in range(columns.lagrangian_underlayers))
    if len(roles) >= count:
        raise ValueError("Layer topology must retain an independent basal layer")
    roles.extend(f"eulerian_{i+1}" for i in range(count-len(roles)-1))
    return roles+["base"]
