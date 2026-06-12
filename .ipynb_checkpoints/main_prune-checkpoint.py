import sys
from runners.prune_runner import PruneRunner

# modify prune_config.yaml to adjust to your model and pruning setting
def main():
    config_path = "config/prune_config.yaml"
    overrides = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            overrides[k] = v
        else:
            config_path = arg

    runner = PruneRunner(config_path)
    for k, v in overrides.items():
        old = getattr(runner.cfg, k)
        setattr(runner.cfg, k, type(old)(v))
    runner.run_name = runner.build_run_name()
    print(f"[LOG] running {runner.run_name}")
    runner.run()

if __name__ == "__main__":
    main()