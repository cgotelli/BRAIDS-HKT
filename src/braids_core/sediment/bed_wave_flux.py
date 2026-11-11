"""Bed-only numerical dissipation for the split hydraulic/Exner update.

The augmented f-wave solver already evolves the hydraulic jumps. Applying
the complete coupled sign(A) flux row to Exner alone can turn a hydraulic
pressure jump into a spurious sediment source, even across a planar bed.
Keep the existing second-order physical capacity flux, adding only the bed
diagonal of |A| times the reconstructed bed jump. The shared face is still
gathered conservatively, with the existing finite-stock and wet/dry guards.

This is a BRAIDS numerical correction, not a copied Delft3D transport law.
Qualification: reports/desktop_wave164 in river_morphodynamics.
"""

METHOD = "split_capacity_true_face_bed_diagonal_v2"


def source(code):
    """Transform the resolved mean-state wave source; fail on ABI drift.

    State order is h, hu_n, hu_t, z. Only A[1,3] = gh is nonzero in the
    bed column, hence |A|[3,3] = sign(A)[3,1]*gh. Its nonnegative part is
    a numerical viscosity, not an extra physical transport coefficient.
    The existing coupled spectral radius remains the conservative step bound.
    """
    start = "  double jump[4]={"
    end = "  double old=face[i]*nx+face[nf+i]*ny;"
    marker = "  double next=(.5*(qs[0]+qs[1]-diss))*law_params[13]/acceleration;"
    if code.count(start) != 1 or code.count(end) != 1 or code.count(marker) != 1:
        raise ValueError("Unexpected resolved sediment-wave source layout")
    a, b = code.index(start), code.index(end)
    if not a < code.index(marker) < b:
        raise ValueError("Unexpected resolved sediment-wave source order")
    replacement = """  // Preserve the physical MUSCL flux; do not inject independent hydraulic jumps.
  const double bed_speed=fmax(0.,row[1]*gh);
  double next=face[i]*nx+face[nf+i]*ny-.5*bed_speed*bed_jump[i]*law_params[13]/acceleration;
"""
    import re
    result=code[:a]+replacement+code[b:]
    pattern=r'(ENTRY void matrix_bed_faces\([^{}]*)(\)\{)'
    result,count=re.subn(pattern,r'\1,const double*bed_jump\2',result)
    if count!=1:raise ValueError('Unexpected bed-wave entry signature')
    return "// "+METHOD+"\n"+result
