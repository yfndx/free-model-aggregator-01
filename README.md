# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/yfndx/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/yfndx/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A free AI model aggregator gateway: one Node 22 server that serves the OpenAI Chat Completions API and the Anthropic Messages API (JSON and streaming) over the trial's ten free-model providers. Each request is answered with the highest-rated free model that meets its requirements (tools, vision, JSON mode, context), with automatic failover to the next candidate when an upstream provider fails.

## Run it

```sh
node main.ts --list
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=g4q2q9hb&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
