from runners.eval_runner import EvalRunner

# modify prune_config.yaml to adjust to your model and pruning setting
def main():
    runner = EvalRunner("config/eval_config.yaml")
    runner.run()

if __name__ == "__main__":
    main()