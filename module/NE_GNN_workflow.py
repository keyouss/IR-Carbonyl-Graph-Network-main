"""Command-line workflow for NE-GNN carbonyl IR prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ne_gnn import CarbonylIRDataset, GNNPredictor, load_carbonyl_table, random_split_dataframe
from ne_gnn.training import evaluate_predictions, predict, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train NE-GNN for carbonyl IR peak prediction.")
    parser.add_argument("--data", default="dataset/CIAC_carbonyl_group_extended.xlsx")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="model/ne_gnn_latest.pt")
    return parser


def dataframe_to_dataset(df):
    doi = df["DOI"] if "DOI" in df.columns else None
    return CarbonylIRDataset(df["Canonical_SMILES"], df["IR_Characteristic_Peak"], doi)


def main() -> None:
    args = build_parser().parse_args()
    df = load_carbonyl_table(args.data)
    train_df, valid_df, test_df = random_split_dataframe(df)

    train_dataset = dataframe_to_dataset(train_df)
    valid_dataset = dataframe_to_dataset(valid_df)
    test_dataset = dataframe_to_dataset(test_df)

    model = GNNPredictor(node_dim=20, edge_dim=5, hidden_dim=128, num_layers=4, num_nodes=2)
    history = train_model(
        model,
        train_dataset,
        valid_dataset=valid_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    y_true, y_pred = predict(model, test_dataset, device=args.device, batch_size=args.batch_size)
    metrics = evaluate_predictions(y_true, y_pred)
    print(f"Final epoch: {history[-1]}")
    print(f"Test metrics: {metrics}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_mean": train_dataset.mean,
            "train_std": train_dataset.std,
            "metrics": metrics,
            "args": vars(args),
        },
        output,
    )
    print(f"Saved checkpoint to {output}")


if __name__ == "__main__":
    main()
