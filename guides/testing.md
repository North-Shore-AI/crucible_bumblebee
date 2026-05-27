# Testing

The default test suite is offline. Live tests are tagged with
`:live_cpu_heavy`, `:live_gpu`, or related tags and are excluded by default.

Run `mix ci` before publishing. It checks formatting, warnings-as-errors,
tests, and docs.
