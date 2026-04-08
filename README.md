---
title: Hospital Bed Environment
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - hospital
  - reinforcement-learning
  - meta-hackathon
pinned: false
---

# Hospital Bed & Resource Allocation Environment

A fully-compliant OpenEnv environment for the Meta AI Hackathon.

## Quick Start

The server runs automatically on port 7860. Test it with:

```bash
# Health check
curl https://YOUR_SPACE_URL/health

# Reset environment
curl -X POST https://YOUR_SPACE_URL/reset -H "Content-Type: application/json" -d '{"task_id": "steady_state"}'

# Step
curl -X POST https://YOUR_SPACE_URL/step -H "Content-Type: application/json" -d '{"action": {"decisions": []}}'
```

## Tasks
- `steady_state` — Easy: 1 ward, 10 steps
- `surge_event` — Medium: 3 wards, 20 steps, surge at step 9
- `mass_casualty` — Hard: 3 wards, 30 steps, mass casualty + staff shortage
