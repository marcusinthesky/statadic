# Geometry contracts without the terminology trap

A method's name does not establish its mathematical laws. STATADIC distinguishes the objects being compared, the ground comparison, the transformation, the resulting quantity and the authority for its declared properties.

## Start with the objects and equality

Points, directions, empirical measures and fitted parameters are different objects. A normalization that maps a nonzero vector onto its direction deliberately removes positive scale. A zero distance in that representation need not mean equality of the original vectors or distributions.

Angular comparisons therefore need their nonzero domain stated beside the first example. Euclidean scale sensitivity is a different choice, not merely a numerical tuning option.

## Keep the output convention visible

Raw V-energy and rooted V-energy differ. MMD squared and rooted MMD differ. A signed U-statistic estimate can be negative; clipping it and taking a square root does not make it a legitimately constructed distance.

The high-level `Energy` and `MMD` representations require rooted V-statistics in STATADIC. The scalar energy function is a different door. Consult the [reference card](../reference/energy.md) before comparing reported numbers.

## Match the consumer's actual requirement

A metric alone does not automatically supply every property needed by an embedding or kernel method. PCoA requires the relevant squared-distance negative-type structure; DISCO requires the corresponding law of its input distance. A kernel consumer needs a justified PSD kernel. General Wasserstein metric status is not a universal license to compose all of these.

The host `HilbertianMeasureMatrix` records a producer's explicit construction warrant. It is not a proof inferred from one observed spectrum, a Boolean flag or erased runtime type parameters. Finite sliced comparisons also retain the limits of their shared finite projection design.

## Progressive explanation, exact authority
