# Qwen resource audit

The current review uses the Apple M4 Max GPU through MLX. Thirty one-second samples recorded device utilization between 99% and 100%, with median 100.0%. Every sample was at least 95%. This establishes sustained GPU saturation in that interval, not proof of maximum theoretical throughput or clock frequency.

Minimum available system memory was 12.48 GiB. No new swap-outs occurred. About 3.81 MiB was swapped in. Peak driver-reported GPU memory in use was 18.81 GiB; this counter covers the device, not just Qwen. System power diagnostics reported no recorded thermal/performance warning. Clock/power telemetry required unavailable administrator authentication, so no maximum-clock claim is made.

The last 12 completed batches had median elapsed time 210.1 seconds. Reasoning accounted for 86.6% of generated tokens. The review's large reasoning workload explains much of its elapsed time; low CPU occupancy during GPU generation is not evidence of idle inference capacity.

Quality configuration remains unchanged: installed 4-bit Qwen3.8 27B, native image inputs, 16,384-token context, 8,192 output-token allowance, two-case batches, close-up rechecks, strict response validation and conservative abstention. Memory guards remain active. Concurrency was not increased without matched evidence that it preserves review quality; these measurements alone do not establish a safe throughput benefit from more simultaneous requests.

The worker continues running. Raw samples are in `work/track1_vlm/performance/telemetry.json`; the structured report is `performance_audit.json`. No model or dataset settings were modified during this audit.
