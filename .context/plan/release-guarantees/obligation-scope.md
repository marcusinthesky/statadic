# Integrated proof scope

No required all-method obligation is declared closed by this integration.
The claim registry deliberately leaves these obligation arrays empty; the
relationships below are partial evidence, not closure mappings.

| Obligation | New evidence | Remaining scope |
| --- | --- | --- |
| permutation-002 | RANDOMIZATION-002 | Finite invariant law, independent uniform group draws with replacement only; other schedules, conditioning, selection and executable generation unverified |
| mmd-001 | MMD-003 through MMD-006 | Finite weighted V algebra, PSD/Gram sign, negative U witness and zero padding; characteristic separation, iid moment derivation and numerical refinement remain open |
| energy-002 | ENERGY-003 | Ordinary Euclidean distance, exponent 1, finite laws only; general moments, other powers and separation remain open |
| linear-002 | LINEAR-006 | Positive diagonal known precision, finite-law full-rank WLS and Aitken contrasts; general estimation and numerical QR correspondence remain open |
| linear-covariance-001 | LINEAR-007 and LINEAR-008 | Supplied finite error moments, sandwich and leverage algebra; robust consistency, interval coverage and numerical correspondence remain open |

LINEAR-005 repairs its existing coefficient-contrast statement. It does not add
a new coverage or implementation claim. Existing implementation pins bind source
fragments; they do not prove floating-point refinement.
