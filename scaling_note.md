## Scaling Note (≤ ½ page)

**Cloud deployment.** Package the agent as a container (Docker) and run it as a scheduled job (e.g., a nightly Cloud Run job / AWS ECS task triggered by EventBridge or Cloud Scheduler) after the day's sales close. It reads sales, stock and parameters from a warehouse or object store (BigQuery/S3), writes recommended POs and the decision log back, and sends POs to a human approval queue before they go to suppliers. Per-SKU logic is independent, so it parallelises across SKUs (one task per batch of SKUs).

**Monitoring.** Alert if rolling fill rate drops below target (e.g., 7-day fill rate < service level − 3 pts), if forecast error (WAPE/bias) drifts above its baseline, if stockout days spike, or if the job misses its schedule. Dashboards show fill rate, forecast error, stockouts, orders placed and the daily decision log per SKU.

**Reliability.** Retry reads/writes with exponential backoff; make the job idempotent (one run per SKU-date, keyed so reruns do not duplicate POs). Validate inputs (schema, negative quantities, missing days, unknown SKUs); on missing data fall back to the last good forecast and flag it, and on repeated failure freeze orders and page a human rather than guess.

**Cost controls.** Per-SKU and portfolio inventory caps (days of cover and $ value), a maximum PO size and daily PO budget requiring approval above a threshold, and working-capital tracking (average and peak inventory value) reported next to fill rate.
