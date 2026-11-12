# Track 1 dependence-aware state experiment

Open development candidate. Frozen baseline inputs are preserved. This run tests the prepared native contexts and analytic motion; independent physical references remain absent.

Native RGB uses a forward/backward checked LK tracker initialized by a manual mask. Point IDs are permanent and lost points stay unsupported. This is a measured classical baseline; no neural point-tracker training or multi-neural-teacher distillation was performed.

| Reference | Source | Method | Horizon | Proposed | Supported | Participant RMSE px |
|---|---|---|---:|---:|---:|---:|
| Exact synthetic | anchor | adaptive | 0.0 | 360 | 357 | 1.4680444008132985 |
| Exact synthetic | anchor | adaptive | 0.25 | 360 | 355 | 22.029909984955474 |
| Exact synthetic | anchor | adaptive | 0.5 | 360 | 325 | 45.7880482647689 |
| Exact synthetic | anchor | cv | 0.0 | 360 | 360 | 3.2953984192890076 |
| Exact synthetic | anchor | cv | 0.25 | 360 | 358 | 17.452747431979052 |
| Exact synthetic | anchor | cv | 0.5 | 360 | 328 | 34.20564287228024 |
| Exact synthetic | anchor | dependent | 0.0 | 360 | 357 | 3.3966363516873286 |
| Exact synthetic | anchor | dependent | 0.25 | 360 | 355 | 17.05135889769482 |
| Exact synthetic | anchor | dependent | 0.5 | 360 | 325 | 33.712072277749805 |
| Exact synthetic | anchor | independent | 0.0 | 360 | 357 | 4.0350545403265885 |
| Exact synthetic | anchor | independent | 0.25 | 360 | 355 | 16.1598555440654 |
| Exact synthetic | anchor | independent | 0.5 | 360 | 325 | 31.203472586863803 |
| Exact synthetic | anchor | nearest | 0.0 | 360 | 360 | 2.629227283958392 |
| Exact synthetic | anchor | nearest | 0.25 | 360 | 358 | 18.31686234209493 |
| Exact synthetic | anchor | nearest | 0.5 | 360 | 328 | 30.614123782501487 |
| Exact synthetic | centroid | adaptive | 0.0 | 360 | 357 | 14.58537780628193 |
| Exact synthetic | centroid | adaptive | 0.25 | 360 | 355 | 25.68460861011328 |
| Exact synthetic | centroid | adaptive | 0.5 | 360 | 325 | 40.94300964828106 |
| Exact synthetic | centroid | cv | 0.0 | 360 | 360 | 11.253619068342624 |
| Exact synthetic | centroid | cv | 0.25 | 360 | 358 | 25.514946961675868 |
| Exact synthetic | centroid | cv | 0.5 | 360 | 328 | 45.200336259926246 |
| Exact synthetic | centroid | dependent | 0.0 | 360 | 357 | 13.746797390611164 |
| Exact synthetic | centroid | dependent | 0.25 | 360 | 355 | 26.238326105815396 |
| Exact synthetic | centroid | dependent | 0.5 | 360 | 325 | 43.698528770932086 |
| Exact synthetic | centroid | independent | 0.0 | 360 | 357 | 12.860895378513032 |
| Exact synthetic | centroid | independent | 0.25 | 360 | 355 | 25.917426092499728 |
| Exact synthetic | centroid | independent | 0.5 | 360 | 325 | 43.40428453773054 |
| Exact synthetic | centroid | nearest | 0.0 | 360 | 360 | 11.847633762579898 |
| Exact synthetic | centroid | nearest | 0.25 | 360 | 358 | 21.292629972129863 |
| Exact synthetic | centroid | nearest | 0.5 | 360 | 328 | 32.878436364418555 |
| RGB consistency proxy | anchor | adaptive | 0.0 | 95 | 4 | 23.49338703364465 |
| RGB consistency proxy | anchor | adaptive | 0.25 | 95 | 2 | 50.85982844129789 |
| RGB consistency proxy | anchor | adaptive | 0.5 | 95 | 1 | 101.80812478499476 |
| RGB consistency proxy | anchor | cv | 0.0 | 95 | 4 | 16.341126347362177 |
| RGB consistency proxy | anchor | cv | 0.25 | 95 | 2 | 32.70603091153822 |
| RGB consistency proxy | anchor | cv | 0.5 | 95 | 1 | 49.22047054595452 |
| RGB consistency proxy | anchor | dependent | 0.0 | 95 | 4 | 10.27190128440378 |
| RGB consistency proxy | anchor | dependent | 0.25 | 95 | 2 | 31.971508119983575 |
| RGB consistency proxy | anchor | dependent | 0.5 | 95 | 1 | 57.282917142151014 |
| RGB consistency proxy | anchor | independent | 0.0 | 95 | 4 | 13.620014239878707 |
| RGB consistency proxy | anchor | independent | 0.25 | 95 | 2 | 35.743973093772134 |
| RGB consistency proxy | anchor | independent | 0.5 | 95 | 1 | 61.44296977477849 |
| RGB consistency proxy | anchor | nearest | 0.0 | 95 | 4 | 12.745276120224567 |
| RGB consistency proxy | anchor | nearest | 0.25 | 95 | 2 | 24.03614819796821 |
| RGB consistency proxy | anchor | nearest | 0.5 | 95 | 1 | 34.31214923690102 |
| RGB consistency proxy | centroid | adaptive | 0.0 | 95 | 5 | 148.4208845219898 |
| RGB consistency proxy | centroid | adaptive | 0.25 | 95 | 4 | 138.7150600444963 |
| RGB consistency proxy | centroid | adaptive | 0.5 | 95 | 2 | 134.5564755346066 |
| RGB consistency proxy | centroid | cv | 0.0 | 95 | 5 | 147.42138282583556 |
| RGB consistency proxy | centroid | cv | 0.25 | 95 | 4 | 155.30855590316565 |
| RGB consistency proxy | centroid | cv | 0.5 | 95 | 2 | 175.3867521237591 |
| RGB consistency proxy | centroid | dependent | 0.0 | 95 | 5 | 146.95166517063703 |
| RGB consistency proxy | centroid | dependent | 0.25 | 95 | 4 | 140.48654247207506 |
| RGB consistency proxy | centroid | dependent | 0.5 | 95 | 2 | 134.32486659563307 |
| RGB consistency proxy | centroid | independent | 0.0 | 95 | 5 | 146.8880589311384 |
| RGB consistency proxy | centroid | independent | 0.25 | 95 | 4 | 137.649952384271 |
| RGB consistency proxy | centroid | independent | 0.5 | 95 | 2 | 128.52761858168424 |
| RGB consistency proxy | centroid | nearest | 0.0 | 95 | 5 | 149.08744396328206 |
| RGB consistency proxy | centroid | nearest | 0.25 | 95 | 4 | 136.05451551688324 |
| RGB consistency proxy | centroid | nearest | 0.5 | 95 | 2 | 122.6738703526186 |

Both unconditioned support and common-support tables are stored in JSON. Disjoint RGB points still share the same seed, image sequence and algorithm. They are not human references. Retrospective centroid observations are explicitly separated from past-only RGB recursion.

Implemented: correlated observation filtering, robust measurements/coherent jumps, joint exponential memories, endpoint-constrained history fitting, full gauge/mixture covariance, bounded static all-ring geometry allocation, task-weighted projection and compact linear probes.

Remaining scope: neural point-tracker adaptation/re-detection, multi-neural-teacher verification/distillation, local adaptive spline knots, temporal geometry continuity and independently labeled motion/contact/identity evaluation. These are not completed by the numerical or proxy checks.

See synthetic.json, real.json, memory.json, geometry.json and verification.json for measured evidence. No candidate is promoted automatically.
