"""Live prediction service: reads the collector's data, scores upcoming connections every
minute with the same feature code as training, and serves them over HTTP (FastAPI)."""
