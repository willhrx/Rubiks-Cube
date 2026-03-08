import torch
checkpoint = torch.load("checkpoints/best_model.pt", weights_only=False)
print(checkpoint.keys())  # See what's saved
print(checkpoint.get('total_iterations'))
print(checkpoint.get('best_solve_rate'))
