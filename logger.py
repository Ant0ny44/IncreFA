import logging
import json
import os


class MixLogger:
    def __init__(self, experiment: str):
        os.makedirs("logs", exist_ok=True)
        self.metrics_path = os.path.join("logs", f"{experiment}.jsonl")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s] => %(message)s",
            handlers=[logging.StreamHandler()],
        )

    def log_scalars(self, values, step):
        payload = {"step": step, **values}
        with open(self.metrics_path, "a") as stream:
            stream.write(json.dumps(payload) + "\n")
