import numpy as np
import matplotlib.pyplot as plt

def visualize_alignment(score_matrix_file, output_file):
    # Read the score matrix from file
    score_matrix = np.loadtxt(score_matrix_file)

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.imshow(score_matrix, cmap='hot', interpolation='nearest')
    plt.title('Alignment Score Matrix')
    plt.xlabel('Query Sequence')
    plt.ylabel('Reference Sequence')
    plt.colorbar()

    # Save the plot
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    score_matrix_file = "alignment_score.txt"  # This should match the output file from the CUDA program
    output_file = "alignment_visualization.png"
    visualize_alignment(score_matrix_file, output_file)