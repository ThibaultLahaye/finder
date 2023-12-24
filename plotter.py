import pandas as pd
import matplotlib.pyplot as plt

def load_and_process_csv(file_path):
    # Load CSV file into a DataFrame
    df = pd.read_csv(file_path, comment='#')

    iteration = df.iloc[:, 0]
    elapsed_time = df.iloc[:, 1]
    mean_value = df.iloc[:, 2]
    best_value = df.iloc[:, 3]

    print(iteration)
    print(elapsed_time)
    print(mean_value)

    return iteration, elapsed_time, mean_value, best_value

def create_plots(iteration, elapsed_time, mean_value, best_value):
    # Plot 1: Elapsed time vs Mean and Best values
    plt.figure(figsize=(10, 6))
    plt.plot(elapsed_time, mean_value, label='Mean Value')
    plt.plot(elapsed_time, best_value, label='Best Value')
    plt.xlabel('Elapsed Time')
    plt.ylabel('Values')
    plt.title('Elapsed Time vs Mean and Best Values')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot 2: Iteration vs Mean and Best values
    plt.figure(figsize=(10, 6))
    plt.plot(iteration, mean_value, label='Mean Value')
    plt.plot(iteration, best_value, label='Best Value')
    plt.xlabel('Iteration')
    plt.ylabel('Values')
    plt.title('Iteration vs Mean and Best Values')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_path = './r0713047.csv'

    iteration, elapsed_time, mean_value, best_value = load_and_process_csv(file_path)
    create_plots(iteration, elapsed_time, mean_value, best_value)