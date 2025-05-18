import matplotlib.pyplot as plt
import numpy as np

# visualization trend when increasing the number of original example when doing fewshort learning on the speech dataset
# Data for group 1
def original_visualization():
    labels_1 = ['obama', 'trump', 'bush']
    values_1 = [0.42, 0.16, 0.14]

    # Data for group 2
    labels_2 = ['obama', 'trump', 'bush']
    values_2 = [0.60, 0.35, 0.27]

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot the first group
    axes[0].bar(labels_1, values_1, color=['blue', 'red', 'green'])
    axes[0].set_title('AV with mimicking')
    axes[0].set_ylabel('Accuracy')

    # Plot the second group
    axes[1].bar(labels_2, values_2, color=['blue', 'red', 'green'])
    axes[1].set_title('AV with original')
    axes[0].set_ylim(0, 0.7)
    axes[1].set_ylim(0, 0.7)
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig('comparison_figure.png', dpi=300)
    # Show the plot
    plt.show()

def authorship_verification_trend():
    # Given data
    # data = {
    #     "0_original": {'obama': 0.3257, 'trump': 0.1758, 'bush': 0.2453},
    #     "2_original": {'obama': 0.4157, 'trump': 0.25, 'bush': 0.2813},
    #     "5_original": {'obama': 0.4804, 'trump': 0.2530, 'bush': 0.3014},
    #     "8_original": {'obama': 0.61, 'trump': 0.26, 'bush': 0.34},
    #     "10_original": {'obama': 0.69, 'trump': 0.31, 'bush': 0.34},
    # }

    # without user metadata
    data = {
        "0_original": {'obama': 0.13871559633027523, 'trump': 0.0390625, 'bush': 0.230188679245283},
        "2_original": {'trump': 0.0546875, 'bush': 0.2226415094339623, 'obama': 0.13844036697247707},
        "5_original": {'obama': 0.16394495412844038, 'trump': 0.06640625, 'bush': 0.2360377358490566},
        "8_original": {'obama': 0.211467889908257, 'trump': 0.10546875, 'bush': 0.25754716981132076},
        "10_original": {'trump': 0.0625, 'bush': 0.258773584905660377, 'obama': 0.1726605504587156},
    }

    # Extract x-axis labels
    x_labels = list(data.keys())
    x = range(len(x_labels))  # Numeric x-axis positions

    # Extract y-values for each political figure
    obama_values = [d['obama'] for d in data.values()]
    trump_values = [d['trump'] for d in data.values()]
    bush_values = [d['bush'] for d in data.values()]

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(x, obama_values, marker='o', linestyle='-', color='blue', label='Obama')
    plt.plot(x, trump_values, marker='s', linestyle='--', color='red', label='Trump')
    plt.plot(x, bush_values, marker='^', linestyle='-.', color='green', label='Bush')

    # Formatting
    plt.xticks(x, x_labels, rotation=20, ha='right')  # Rotate x-axis labels
    plt.ylabel("Value")
    plt.title("Trend of Obama, Trump, and Bush across Conditions")
    plt.legend()
    plt.grid(True)
    plt.savefig('mimicking_attribution.png', dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()

authorship_verification_trend()