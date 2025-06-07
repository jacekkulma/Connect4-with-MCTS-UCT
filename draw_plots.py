import os
import pandas as pd
import matplotlib.pyplot as plt
from run_comparison_tests import seeds, models

data_folder = "tests"
game_count = len(seeds)

def format_pct_and_count(pct, allvals):
            absolute = int(round(pct / 100. * sum(allvals)))
            return f"{pct:.1f}%\n({absolute})"

def main() -> None:
    for file in os.listdir(data_folder):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(data_folder, file)
        df = pd.read_csv(file_path)

        # Win rate pie chart
        labels = df['name'].tolist()
        wins = df['wins'].tolist()

        total_recorded_wins = sum(wins)
        draw_count = game_count - total_recorded_wins

        if draw_count > 0:
            labels.append("Draw")
            wins.append(draw_count)

        plt.figure(figsize=(9, 6))
        wedges, texts, autotexts = plt.pie(
            wins,
            labels=None,
            autopct=lambda pct: format_pct_and_count(pct, wins),
            textprops={'color':"black", 'bbox':{'facecolor':'white', 'edgecolor':'none', 'pad':2}},
            startangle=90
        )
        plt.title(f"Win distribution: {file.replace('.csv', '').replace('_', ' ')}")
        plt.axis('equal')

        plt.legend(wedges, labels, title="Model", loc="center left", bbox_to_anchor=(1, 0.5))

        output_folder = os.path.join("plots", "win_rate")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, file.replace(".csv", ".png"))

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        # Avg turn chart - one base model compared with all opponents
        output_folder = os.path.join("plots", "avg_turns")
        os.makedirs(output_folder, exist_ok=True)

        # print(models)
        for BASE_MODEL in models:
            base_avg_turns = []
            opponent_avg_turns = []
            opponent_names = []

            for file in os.listdir(data_folder):
                if not file.endswith(".csv"):
                    continue

                file_path = os.path.join(data_folder, file)
                df = pd.read_csv(file_path)

                if BASE_MODEL not in df['name'].values or len(df) != 2:
                    continue

                base_row = df[df['name'] == BASE_MODEL].iloc[0]
                opp_row = df[df['name'] != BASE_MODEL].iloc[0]

                opponent_names.append(opp_row['name'])
                base_avg_turns.append(base_row['avg_turns'])
                opponent_avg_turns.append(opp_row['avg_turns'])

            if not opponent_names:
                return 

            import numpy as np

            x = np.arange(len(opponent_names))
            width = 0.35

            fig, ax = plt.subplots(figsize=(10, 6))
            bars1 = ax.bar(x - width/2, opponent_avg_turns, width, label='Opponent', color='orange')
            bars2 = ax.bar(x + width/2, base_avg_turns, width, label=BASE_MODEL, color='skyblue')

            ax.set_xlabel('Opponent Model')
            ax.set_ylabel('Average Turns')
            ax.set_title(f'Avg Turns (in wins) per Model vs {BASE_MODEL}\n(lower is better)')
            ax.set_xticks(x)
            ax.set_xticklabels(opponent_names, rotation=45, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.legend()

            for bars in (bars1, bars2):
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f'{height:.2f}',
                        ha='center',
                        va='bottom'
                    )

            
            output_path = os.path.join(output_folder, f"{BASE_MODEL}_comparison.png")

            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()


if __name__ == "__main__":
    main()