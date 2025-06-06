import os
import pandas as pd
import matplotlib.pyplot as plt

data_folder = "tests"

def main() -> None:
    for file in os.listdir(data_folder):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(data_folder, file)
        df = pd.read_csv(file_path)

        # Win rate pie chart
        labels = df['name']
        wins = df['wins']
        plt.figure(figsize=(9, 6))
        wedges, texts, autotexts = plt.pie(
            wins,
            labels=None,
            autopct='%1.1f%%',
            textprops={'color':"black", 'bbox':{'facecolor':'white', 'edgecolor':'none', 'pad':2}},
            startangle=90
        )
        plt.title(f"Win distribution: {file.replace('.csv', '').replace('_', ' ')}")
        plt.axis('equal')

        plt.legend(wedges, labels, title="Model", loc="center left", bbox_to_anchor=(1, 0.5))

        output_folder = "plots/win_rate"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, file.replace(".csv", ".png"))

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        # Avg turn chart
        labels = df['name']
        avg_turns = df['avg_turns']

        bars = plt.bar(labels, avg_turns, color=['skyblue', 'orange'], width=0.3)
        plt.xlabel('Model')
        plt.ylabel('Average Turns')
        plt.title('Average Turns per Winning Model')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        output_folder = "plots/avg_turns"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, file.replace(".csv", ".png"))

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom'
            )

        plt.savefig(output_path)
        plt.close()


if __name__ == "__main__":
    main()