from flask import Flask, render_template, request, send_file, jsonify, make_response
from flask.wrappers import Response
import os
import yaml
from .backtest import Backtester
import matplotlib.pyplot as plt
from typing import Any, Dict

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_files() -> Response:
    if "config_file" not in request.files or "data_file" not in request.files:
        return make_response(
            jsonify({"error": "Both configuration and data files are required."}),
            400,
        )

    config_file = request.files["config_file"]
    data_file = request.files["data_file"]

    config_path: str = os.path.join(app.config["UPLOAD_FOLDER"], config_file.filename)
    data_path: str = os.path.join(app.config["UPLOAD_FOLDER"], data_file.filename)

    config_file.save(config_path)
    data_file.save(data_path)

    # Load config
    with open(config_path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    config["data"]["file_path"] = data_path
    config["output"]["file_path"] = os.path.join(
        app.config["UPLOAD_FOLDER"], "results.csv"
    )

    # Run backtest
    backtester = Backtester(config)
    results = backtester.run()

    # Save results
    results_path: str = config["output"]["file_path"]
    results.to_csv(results_path)

    # Collect metrics
    metrics: Dict[str, Any] = backtester.calculate_performance_metrics(results)

    # Generate and save plot
    plot_path: str = os.path.join(app.config["UPLOAD_FOLDER"], "backtest_plot.png")
    plot_results(results, plot_path)

    return make_response(
        jsonify(
            {
                "metrics": metrics,
                "results_file": "results.csv",
                "plot_file": "backtest_plot.png",
            }
        ),
        200,
    )


@app.route("/download/<filename>")
def download_file(filename: str) -> Response:
    file_path: str = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return make_response(jsonify({"error": "File not found."}), 404)


@app.route("/plot/<filename>")
def serve_plot(filename: str) -> Response:
    file_path: str = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="image/png")
    return make_response(jsonify({"error": "File not found."}), 404)


def plot_results(data: Any, plot_path: str) -> None:
    """Generate and save a plot of cumulative returns."""
    plt.figure(figsize=(12, 6))
    plt.plot(data["Cumulative_Market_Return"], label="Market Return", color="blue")
    plt.plot(
        data["Cumulative_Strategy_Return"], label="Strategy Return", color="orange"
    )
    plt.title("Backtest Results: Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid()
    plt.savefig(plot_path)
    plt.close()


if __name__ == "__main__":
    app.run(debug=True)
