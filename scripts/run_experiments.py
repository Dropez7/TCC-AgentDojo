#!/usr/bin/env python3
"""
Script de automação para execução de benchmarks e coleta de métricas no AgentDojo.
Projetado para avaliação de resiliência a Indirect Prompt Injection (IPI).

Funcionalidades:
- Execução em lote de cenários (Baseline, Important Instructions, Tool Knowledge, Neutral Noise)
- Suporte a modelos locais (via Ollama/OpenAI compatível) ou modelos via API
- Coleta e agregação de métricas: Utility (%), Security/ASR (%), Duração
- Exportação dos resultados em formatos CSV, JSON e Tabela LaTeX pronta para o TCC.
"""

import argparse
import datetime
import json
import logging
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.attacks.attack_registry import load_attack

# Importações do framework AgentDojo
from agentdojo.benchmark import (
    SuiteResults,
    benchmark_suite_with_injections,
    benchmark_suite_without_injections,
)
from agentdojo.logging import OutputLogger
from agentdojo.task_suite.load_suites import get_suite

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_single_benchmark(
    suite_name: str,
    model_name: str,
    attack: str | None,
    defense: str | None,
    user_tasks: list[str] | None,
    injection_tasks: list[str] | None,
    logdir: Path,
    benchmark_version: str = "v1.1.1",
    force_rerun: bool = False,
) -> SuiteResults:
    """Executa uma rodada única de benchmark no AgentDojo."""
    suite = get_suite(benchmark_version, suite_name)

    pipeline_config = PipelineConfig(
        llm=model_name,
        defense=defense,
    )
    pipeline = AgentPipeline.from_config(pipeline_config)

    with OutputLogger(str(logdir)):
        if attack is None or attack.lower() in ("none", "baseline"):
            console.log(
                f"[bold cyan]Executando Baseline (Sem Ataque)[/bold cyan] para {suite_name} | Modelo: {model_name}"
            )
            results = benchmark_suite_without_injections(
                pipeline=pipeline,
                suite=suite,
                user_tasks=tuple(user_tasks) if user_tasks else None,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=benchmark_version,
            )
        else:
            console.log(
                f"[bold yellow]Executando Ataque '{attack}'[/bold yellow] para {suite_name} | Modelo: {model_name}"
            )
            attacker = load_attack(attack, suite, pipeline)
            results = benchmark_suite_with_injections(
                pipeline=pipeline,
                suite=suite,
                attack=attacker,
                user_tasks=tuple(user_tasks) if user_tasks else None,
                injection_tasks=tuple(injection_tasks) if injection_tasks else None,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=benchmark_version,
            )

    return results


def calculate_metrics(results: SuiteResults, is_attack: bool) -> dict[str, float]:
    """Calcula a Utilidade média (%) e o ASR (%)."""
    utility_list = list(results["utility_results"].values())
    avg_utility = (sum(utility_list) / len(utility_list) * 100.0) if utility_list else 0.0

    asr = 0.0
    if is_attack and "injection_tasks_utility_results" in results:
        inj_results = list(results["injection_tasks_utility_results"].values())
        if inj_results:
            asr = sum(inj_results) / len(inj_results) * 100.0

    total_samples = len(utility_list)
    return {
        "samples": total_samples,
        "utility": avg_utility,
        "asr": asr,
    }


def generate_latex_table(df_summary: pd.DataFrame, output_file: Path) -> str:
    """Gera código LaTeX formatado para inclusão direta no TCC."""
    latex_code = [
        "\\begin{table}[htb]",
        "\\centering",
        "\\caption{Resultados comparativos automatizados de Utilidade e ASR}",
        "\\label{tab:resultados_automatizados}",
        "\\begin{tabular}{|l|c|c|c|}",
        "\\hline",
        "\\textbf{Cenário / Vetor} & \\textbf{Amostras ($N$)} & \\textbf{Utilidade Média (\\%)} & \\textbf{ASR (\\%)} \\\\ \\hline",
    ]

    for _, row in df_summary.iterrows():
        cenario = str(row["Cenário"]).replace("_", "\\_")
        samples = int(row["Amostras ($N$)"])
        utility = f"{row['Utilidade (%)']:.1f}\\%"
        asr = f"{row['ASR (%)']:.1f}\\%"
        latex_code.append(f"{cenario} & {samples} & {utility} & {asr} \\\\ \\hline")

    latex_code.extend(["\\end{tabular}", "\\end{table}"])

    code_str = "\n".join(latex_code)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(code_str)
    return code_str


def main():
    parser = argparse.ArgumentParser(description="Automação de Testes e Coleta de Métricas - AgentDojo")
    parser.add_argument("--suite", type=str, default="workspace", help="Nome da suíte (padrão: workspace)")
    parser.add_argument("--model", type=str, default="qwen2.5:7b-instruct", help="Nome do modelo no pipeline")
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["none", "important_instructions", "tool_knowledge", "neutral_noise"],
        help="Lista de ataques a executar em sequência",
    )
    parser.add_argument("--user-tasks", nargs="*", default=None, help="Lista específica de user_tasks")
    parser.add_argument("--injection-tasks", nargs="*", default=None, help="Lista específica de injection_tasks")
    parser.add_argument("--logdir", type=str, default="./runs", help="Diretório para salvar logs de execução")
    parser.add_argument("--outdir", type=str, default="./results_summary", help="Diretório para relatórios finais")
    parser.add_argument("--force-rerun", action="store_true", help="Forçar reexecução mesmo se houver cache")
    parser.add_argument("--version", type=str, default="v1.1.1", help="Versão do benchmark")

    args = parser.parse_args()

    logdir = Path(args.logdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_rows = []

    console.rule("[bold green]Iniciando Automação de Experimentos - TCC[/bold green]")

    for attack in args.attacks:
        is_attack = attack.lower() not in ("none", "baseline")
        attack_name = None if not is_attack else attack

        try:
            results = run_single_benchmark(
                suite_name=args.suite,
                model_name=args.model,
                attack=attack_name,
                defense=None,
                user_tasks=args.user_tasks,
                injection_tasks=args.injection_tasks,
                logdir=logdir,
                benchmark_version=args.version,
                force_rerun=args.force_rerun,
            )

            metrics = calculate_metrics(results, is_attack=is_attack)
            summary_rows.append(
                {
                    "Cenário": "Baseline (Sem Ataque)" if not is_attack else attack,
                    "Amostras ($N$)": metrics["samples"],
                    "Utilidade (%)": round(metrics["utility"], 2),
                    "ASR (%)": round(metrics["asr"], 2),
                }
            )

        except Exception as e:
            console.log(f"[bold red]Erro ao executar cenário '{attack}': {e}[/bold red]")

    df_summary = pd.DataFrame(summary_rows)

    # Exibição rica no terminal
    rich_table = Table(title=f"Resultados Consolidados ({args.model} - {args.suite})")
    rich_table.add_column("Cenário / Vetor", style="cyan", no_wrap=True)
    rich_table.add_column("Amostras (N)", justify="center", style="magenta")
    rich_table.add_column("Utilidade Média (%)", justify="right", style="green")
    rich_table.add_column("ASR (%)", justify="right", style="red")

    for _, row in df_summary.iterrows():
        rich_table.add_row(
            str(row["Cenário"]), str(row["Amostras ($N$)"]), f"{row['Utilidade (%)']:.2f}%", f"{row['ASR (%)']:.2f}%"
        )

    console.print(rich_table)

    # Exportações
    csv_path = outdir / f"resumo_experimentos_{timestamp}.csv"
    latex_path = outdir / f"tabela_tcc_{timestamp}.tex"
    json_path = outdir / f"metricas_{timestamp}.json"

    df_summary.to_csv(csv_path, index=False)
    generate_latex_table(df_summary, latex_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    console.log(f"[bold green]✓ CSV salvo em:[/bold green] {csv_path}")
    console.log(f"[bold green]✓ Tabela LaTeX salva em:[/bold green] {latex_path}")
    console.log(f"[bold green]✓ JSON salvo em:[/bold green] {json_path}")
    console.rule("[bold green]Concluído com Sucesso[/bold green]")


if __name__ == "__main__":
    main()
