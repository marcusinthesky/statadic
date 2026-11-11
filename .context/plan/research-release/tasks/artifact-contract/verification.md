# Artifact-contract verification

Implementation and cross-lane review are complete. The coordinator established
acceptance and marked this task done from passing checkpoint 1.

The final focused artifact suite passed 51 tests. Package lint and strict typing
passed; the final integrated checkpoint repeated typing after the last
test-only additions. Existing corruption tests were not modified.

A narrow fresh-wheel smoke check passed live/reloaded Poisson and logistic
predictions for every supported kind, corrupt and schema-2 manifests, a fresh
x64-off refusal of 64-bit state, and snapshot TeX trust. The first smoke attempt
used explicit float64 host inputs in its x64-off context; its inputs were
corrected to float32 before the passing rerun.

`just python::statadic::test-artifacts` subsequently passed both wheel routes,
the data-extra walkthrough lane and standalone sdist development. All commands
used `devenv shell --` and locked projects.

The additional whole-file x64-off experiment reported 46 passed and two failures
in existing tests that assume x64 inputs. It was not treated as a passing gate;
the required fresh-process x64-off rejection check passed. The normal default
lane and explicit precision-boundary tests remain the supported checks.

Review required schema rejection before writer-field validation because old
schema-2 manifests lack the new field. Writer versions are diagnostic and never
compared. The post-release compatibility promise still requires owner confirmation.

Checkpoint 1 passed and establishes acceptance; see the [integrated results](../../verification.md).
