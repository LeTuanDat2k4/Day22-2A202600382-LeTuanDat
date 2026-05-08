import subprocess
import argparse
import sys

def run_step(step_num, script_name):
    print("\n" + "=" * 60)
    print(f"  Running Step {step_num}: {script_name}")
    print("=" * 60)
    try:
        # Using sys.executable to ensure we use the same python environment
        subprocess.run([sys.executable, script_name], check=True)
        print(f"\n[Step {step_num}] Completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n[Step {step_num}] Failed with error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Day 22 Lab steps.")
    parser.add_argument("--step", type=int, help="Run a specific step (1, 2, 3, or 4)")
    args = parser.parse_args()

    steps = {
        1: "01_langsmith_rag_pipeline.py",
        2: "02_prompt_hub_ab_routing.py",
        3: "03_ragas_evaluation.py",
        4: "04_guardrails_validator.py"
    }

    if args.step:
        if args.step in steps:
            run_step(args.step, steps[args.step])
        else:
            print(f"Error: Step {args.step} not recognized.")
            sys.exit(1)
    else:
        # Run all steps sequentially
        for step_num, script_name in steps.items():
            run_step(step_num, script_name)
        print("\nAll lab steps completed!")

if __name__ == "__main__":
    main()
