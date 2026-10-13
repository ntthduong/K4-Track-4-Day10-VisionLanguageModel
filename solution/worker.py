"""Isolated GPU worker: exiting releases CUDA allocations and CPU model loaders."""
import argparse
from pathlib import Path

from solution.storage import read_json, write_json


def execute(request):
    op, cfg = request["operation"], request["config"]
    data, work = request["data_dir"], request["work_dir"]
    if op == "infer":
        from solution.experiment import infer
        path = infer(cfg, request.get("checkpoint"), request["split"], data, work)
        result = {"predictions": str(path)}
    else:
        from solution.training import train_grpo, train_sft
        if op == "train":
            checkpoint, state = train_sft(cfg, data, work)
        elif op == "grpo":
            checkpoint, state = train_grpo(cfg, request["checkpoint"], request["e5_config"], data, work)
        else:
            raise ValueError(op)
        result = {"checkpoint": str(checkpoint), "state": state}
    write_json(request["result_path"], result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("request")
    args = ap.parse_args()
    execute(read_json(Path(args.request)))


if __name__ == "__main__":
    main()
