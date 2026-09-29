# GLM-5.3 experiments

These files are excluded from the SGLang package's applied patch directory.

`glm53-rmsnorm.patch` retains native Torch FP32 statistics and fuses the final
normalization multiplies. Its check script requires a runtime built with that
patch. All 298 reference cases per GPU pass, including bitwise native equality,
and a paired full-model probe finds no norm-output differences across all 113
norms on four ranks. Nevertheless, the changed execution/allocation path alters
full-model probabilities, including when the paired probe selects native output.
The candidate remains unqualified and is not enabled in the package.

See [the investigation results](../results/glm53-rmsnorm-investigation-2026-09-29.json)
and [performance audit](../../../docs/glm53-performance-audit.md).
