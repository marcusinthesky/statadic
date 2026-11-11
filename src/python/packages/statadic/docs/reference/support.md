# Environment and transformation support

This is a scoped summary of supported capabilities. The existing capability guide remains the detailed authority.

## Environment

| Item | Contract |
|---|---|
| Python | 3.13 only, `~=3.13.0` |
| JAX dependency | CPU extra, `>=0.11.0,<0.12` |
| Precision | Caller-owned; no library-side global x64 enablement |
| Package identity | Installed distribution metadata; see the package version |
| Table/plot dependencies | Optional `data` and `plot` extras |
| Measured backend claims | CPU; no GPU performance or qualification claim inferred |

## Transformation scope is per operation

| Operation | Supported scope | Do not infer |
|---|---|---|
| Native OLS `LeastSquares` | Fixed-shape JIT/vmap; coefficient derivatives checked away from rank/validity changes | Smoothness at rank changes |
| Native WLS/prediction | Fixed-shape JIT/vmap in checked domains | A blanket fitted-WLS derivative qualification |
| Fixed/iterated affine GMM | Fixed-shape numerical workflows and actual-weight diagnostics | A derivative of the entire iterated fitting procedure |
| Fixed/affine GEL host solve | Eager | JIT merely because a problem is a pytree |
| Host `statadic.api` workflows | Named preparation, reporting and artifact operations are eager | End-to-end JIT over tables or Python reports |
| Scheduled resampling | Fixed structures and explicit schedule/callback contracts | Statistical bootstrap validity or smooth discrete draws |
| Spatial/survival workflows | Operation-specific numerical contracts; host alignment/refinement eager | Blanket fitted-solution derivatives or calibration |

Float32 and float64 numerical witnesses apply to their recorded operations and configurations. One passing dtype case is not an all-dtype guarantee. Backend-specific support must be tested rather than inferred from a JAX import.

## Performance reporting

Separate cold compilation, synchronized warm execution, preparation and transfers. Distinguish compiler allocation estimates from measured peak memory. Report shapes, dtype, backend/device, software versions and thread settings. Do not turn a small synthetic benchmark or the old notebook runtime estimates into a GPU speed promise.
