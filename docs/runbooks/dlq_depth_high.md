# Runbook: DLQ Depth High

Alert: `tech_redis_queue_depth{queue="dlq:audit:tasks"} > 100`

## Triage
- Check DLQ depth: `python scripts/requeue_dlq.py --queue audit:tasks --action stats`
- Inspect recent incidents: `ls -lt reports/incidents | head`

## Actions
- Try processing batch (no-op handler to keep):
  `python scripts/requeue_dlq.py --queue audit:tasks --action process --max 100 --handler requeue`
- Requeue to original (if the source is healthy):
  `python scripts/requeue_dlq.py --queue audit:tasks --action requeue --max 50`

## Verify
- Depth trending down in Grafana.
- No new P1 errors for 15 minutes.

## Rollback/Containment
- Increase DLQ threshold alert temporarily if noisy.
- Park messages by leaving in DLQ; open incident and escalate to on-call.
