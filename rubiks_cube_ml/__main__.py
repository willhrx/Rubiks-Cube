"""
Main entry point for the Rubik's Cube ML project.

Usage:
    python -m rubiks_cube_ml <command> [options]
    
Commands:
    train       Train the model
    demo        Demonstrate the model
    evaluate    Evaluate the model
"""

import argparse
import sys

def main():
    """Parse command line arguments and dispatch to the appropriate module."""
    parser = argparse.ArgumentParser(
        description="Rubik's Cube ML solver",
        usage="python -m rubiks_cube_ml <command> [options]"
    )
    parser.add_argument("command", choices=["train", "demo", "evaluate"], 
                       help="Command to run")
    
    # Parse just the command
    args, remaining = parser.parse_known_args()
    
    # Import the appropriate module based on the command
    if args.command == "train":
        from rubiks_cube_ml.train import main
    elif args.command == "demo":
        from rubiks_cube_ml.demo import main
    elif args.command == "evaluate":
        # Import a simple evaluation runner
        from rubiks_cube_ml.evaluation.evaluator import CubeEvaluator
        from rubiks_cube_ml.model.policy import CubePolicy
        import torch
        import os
        
        def main():
            # Parse arguments
            eval_parser = argparse.ArgumentParser(description="Evaluate Rubik's Cube solver")
            eval_parser.add_argument("--model", type=str, default="checkpoints/model_best.pt",
                                  help="Path to the trained model checkpoint")
            eval_parser.add_argument("--difficulty", type=int, default=20,
                                  help="Maximum scramble difficulty to evaluate")
            eval_parser.add_argument("--episodes", type=int, default=20,
                                  help="Number of episodes per difficulty level")
            eval_parser.add_argument("--save_plot", type=str, default="evaluation_results.png",
                                  help="Path to save the evaluation plot")
            eval_args = eval_parser.parse_args(remaining)
            
            # Create policy
            state_dim = 6 * 3 * 3 * 6
            policy = CubePolicy(state_dim=state_dim)
            
            # Load model if it exists
            if os.path.exists(eval_args.model):
                checkpoint = torch.load(eval_args.model, map_location="cpu")
                policy.load_state_dict(checkpoint['policy_state_dict'])
                print(f"Loaded model from {eval_args.model}")
            else:
                print(f"Warning: Model {eval_args.model} not found. Using untrained model.")
            
            # Create evaluator
            evaluator = CubeEvaluator(policy)
            
            # Evaluate by difficulty
            results = evaluator.evaluate_by_difficulty(
                max_difficulty=eval_args.difficulty,
                episodes_per_level=eval_args.episodes
            )
            
            # Plot results
            evaluator.plot_difficulty_results(results, save_path=eval_args.save_plot)
            
            print(f"\nEvaluation complete. Results saved to {eval_args.save_plot}")
    
    # Run the command with the remaining arguments
    sys.argv = [sys.argv[0]] + remaining
    main()


if __name__ == "__main__":
    main()