# Inventory Replenishment Agent - Barca Group

A daily agent that forecasts demand per SKU (EWMA), sizes safety stock from forecast error and a target service level,
projects inventory over *lead time + review period*, and either places a purchase order (respecting `min_order_qty` and
an over-order cap) or waits. It is simulated day by day and compared to a static `(s, Q)` baseline.

## Run it
```bash
pip install numpy pandas scipy matplotlib        # Python 3.10+
python inventory_agent.py                        # prints the daily log + results, saves charts to outputs/
# or open InventoryReplenishmentAgent.ipynb in Jupyter / Colab and "Run all"
```
If `data/sales.csv`, `data/inventory.csv`, `data/params.csv` are missing they are regenerated (seeded, 3 SKUs x 120 days).

## Files
| File | What it is |
|---|---|
| `InventoryReplenishmentAgent.ipynb` | Main deliverable: design doc, code, daily log, evaluation, trade-offs, scaling note (executed, outputs embedded) |
| `inventory_agent.py` | Same code as a plain script (jupytext "percent" format) |
| `data/sales.csv`, `inventory.csv`, `params.csv` | Input data in the required schemas |
| `design_doc.md`, `scaling_note.md`, `tradeoffs.md` | Design document (<= 1 page), scaling note (<= 1/2 page), results reflection |
| `outputs/` | `agent_daily_log.csv`, charts, `sample_run_output.txt` (console output of a full run) |
| `build_notebook.py` | Rebuilds the notebook from the script + markdown files |

## Key results (default settings, seed 7 data)
| Policy | Fill rate | Stockout days | Total cost | Avg inventory value |
|---|---|---|---|---|
| Static (s,Q) baseline | 93.2% | 21 | $9,498 | $3,932 |
| Naive-forecast agent | 99.5% | 4 | $7,897 | $14,585 |
| **EWMA agent** | **99.5%** | **2** | **$4,058** | $6,631 |

Over 30 other seeds the EWMA agent is 29.8% cheaper than the baseline on average (lower cost in 90% of seeds), but a
baseline hand-tuned to a 0.999 service level is ~4% cheaper than the agent. See section 9 and 11 of the notebook.
