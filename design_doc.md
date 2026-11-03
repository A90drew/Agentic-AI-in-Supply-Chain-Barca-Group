## Design Document (1 page)

**Goal.** Each day, for each SKU, decide *how many units to order, or whether to wait*, so that the business keeps shelves stocked (few stockouts, high fill rate) without tying up cash in excess stock. The agent minimises **total cost = holding + stockout + order cost** subject to a per-SKU target **service level**.

**Inputs (per SKU, per day).** Daily sales history through yesterday (`sales.csv`, aggregated per day per SKU) → EWMA demand forecast `f` and running forecast-error `σ`; current on-hand stock and units already on order (`inventory.csv` opening stock + simulation state); policy and cost inputs from `params.csv`: `lead_time_days`, `min_order_qty`, `service_level`, `holding_cost_per_day`, `stockout_cost`, `unit_cost`.

**Outputs.** One decision per SKU per day: **ORDER *q* units** (arrives after the lead time) or **WAIT**, plus a written rationale with the numbers behind it (the daily log).

**Decision rule.** Protection interval `H = lead time + review period`. Safety stock `SS = z(service level) · σ · √H`. Projected inventory `= on-hand + on-order − f·H`. If projected ≥ SS → wait. Otherwise order up to `f·H + SS + cover_days·f`, with at least `min_order_qty`.

**Guardrails & business rules.**
* Respects `min_order_qty` (raises small orders to the MOQ) and accounts for lead time (orders count as on-order until they arrive, so the agent does not double-order).
* Uses the service level to size safety stock, so riskier/more important SKUs hold more buffer.
* Prevents unrealistic over-ordering: the inventory position can never exceed `max_cover_days · f + SS` (and optionally a $ cap); if the cap leaves no room for a legal PO, the agent waits and logs why.
* Holding vs stockout trade-off: waiting saves holding cost but risks lost sales; every order adds holding cost and a fixed PO cost. The service level is the dial between the two.

**Metrics.** *Stockout* = a day on which demand exceeded stock (we also count units short). *Fill rate* = 1 − lost units / demanded units. *Total cost* = holding + stockout + order cost. We also track average and peak inventory value (working capital).

**Ethics & trust.** Stockouts hurt customers and the people who rely on the product (here lost demand, in a real setting possibly essential goods), so the agent is not allowed to "save money" by silently under-stocking: the service-level target and stockout cost make that trade-off explicit. Over-ordering wastes cash and goods and can hide risk, so it is capped. Every decision is written to a **daily log with its rationale** (numbers, rule that fired, guardrail that applied), so a planner can audit, question and override the agent; the agent recommends, a human remains accountable for policy parameters. Forecast error is tracked and reported rather than hidden.

**Assumptions.** Unmet demand is lost (no backorders); SKUs are independent; demand is stationary enough for EWMA (no explicit seasonality model); forecast errors on different days are independent when scaling `σ` by `√H`; data is synthetic.
