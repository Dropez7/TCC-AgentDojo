#!/usr/bin/env python3
"""
Pipeline Avançado de Avaliação Empírica, Benchmark Multi-Modelo e Análise Qualitativa - AgentDojo

Capacidades:
1. Suporte a Múltiplos Modelos:
   - Execução em múltiplos LLMs locais (via Ollama / OpenAI-compatible / vLLM) ou APIs comerciais.
   - Aplicação homogênea das mesmas User Tasks e Injection Tasks.

2. Rigor Estatístico com Amostragem N-Runs:
   - Repetição de cada par de teste (User Task x Injection Task) por N iterações independentes.
   - Registro individual e temporal de cada execução (Utility: bool, Security: bool).

3. Registro Completo de Respostas, Pensamentos e Logs:
   - Captura integral da cadeia de raciocínio (Chain-of-Thought / tags <think>...</think>).
   - Armazenamento de todas as respostas textuais, chamadas de ferramentas e saídas/erros da runtime.

4. Análise Qualitativa de Causa Raiz de Falhas e Resiliência:
   - Classificação automática de cada episódio em categorias diagnósticas (falha de sintaxe, falta de tool calling, vulnerabilidade direta a IPI, etc.).
   - Geração de relatório detalhado em Markdown com Estudos de Caso e distribuição estatística de falhas.
"""

import argparse
import datetime
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

# AgentDojo imports
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.benchmark import (
    SuiteResults,
    benchmark_suite_with_injections,
    benchmark_suite_without_injections,
)
from agentdojo.logging import OutputLogger
from agentdojo.models import MODEL_PROVIDERS, ModelsEnum
from agentdojo.task_suite.load_suites import get_suite

console = Console()
logging.basicConfig(level=logging.WARNING)

# ==============================================================================
# 1. FUNÇÕES ESTATÍSTICAS
# ==============================================================================

def calculate_ci95(data: list[float]) -> tuple[float, float, float]:
    """Calcula média, desvio padrão amostral e margem de erro do IC de 95%."""
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return float(data[0]), 0.0, 0.0

    arr = np.array(data, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    sem = std / math.sqrt(n)
    ci95_margin = 1.96 * sem
    return mean, std, ci95_margin

# ==============================================================================
# 2. EXTRAÇÃO DE TRACES, PENSAMENTOS E RESPOSTAS DAS LLMS
# ==============================================================================

def extract_text_and_thoughts(content: Any) -> tuple[str, str]:
    """Extrai a resposta textual limpa e os pensamentos / cadeia de raciocínio (CoT)."""
    if content is None:
        return "", ""

    raw_text = ""
    thinking_text = ""

    if isinstance(content, str):
        raw_text = content
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                b_type = block.get("type", "text")
                b_content = block.get("content", "")
                if b_type in ("thinking", "redacted_thinking"):
                    thinking_text += b_content + "\n"
                elif b_type == "text":
                    raw_text += b_content + "\n"
            elif isinstance(block, str):
                raw_text += block + "\n"

    # Extração de tags <think>...</think>
    think_tags = re.findall(r"<think>(.*?)</think>", raw_text, re.DOTALL)
    if think_tags:
        for tag in think_tags:
            thinking_text += tag.strip() + "\n"
        raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

    return raw_text.strip(), thinking_text.strip()

def parse_task_trace_file(trace_path: Path) -> dict[str, Any] | None:
    """Lê o arquivo de trace JSON gerado pelo TraceLogger do AgentDojo e extrai dados ricos."""
    if not trace_path.exists():
        return None
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    messages = data.get("messages", [])
    assistant_replies = []
    reasoning_traces = []
    tool_calls = []
    tool_errors = []
    user_query = ""

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user" and not user_query:
            txt, _ = extract_text_and_thoughts(content)
            user_query = txt
        elif role == "assistant":
            txt, thoughts = extract_text_and_thoughts(content)
            if txt:
                assistant_replies.append(txt)
            if thoughts:
                reasoning_traces.append(thoughts)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", "")
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                tool_calls.append({"function": fn, "args": args})
        elif role == "tool":
            err = msg.get("error")
            if err:
                tool_errors.append(str(err))

    return {
        "user_query": user_query,
        "assistant_replies": assistant_replies,
        "reasoning_traces": reasoning_traces,
        "tool_calls": tool_calls,
        "tool_errors": tool_errors,
        "injections": data.get("injections", {}),
        "error": data.get("error"),
        "duration": data.get("duration"),
    }

# ==============================================================================
# 3. MOTOR DE ANÁLISE QUALITATIVA E DIAGNÓSTICO DE CAUSA RAÍZ
# ==============================================================================

def diagnose_episode(utility: bool, security: bool, is_attack: bool, trace_info: dict[str, Any] | None) -> tuple[str, str]:
    """Classifica o episódio em uma categoria de causa raiz e devolve descrição."""
    if trace_info is None:
        if not utility:
            return "FALHA_EXECUCAO_DESCONHECIDA", "Trace indisponível; falha não identificada."
        return "SUCESSO_SEM_TRACE", "Execução bem-sucedida sem trace detalhado."

    tool_calls = trace_info.get("tool_calls", [])
    tool_errors = trace_info.get("tool_errors", [])
    trace_err = trace_info.get("error")
    total_tools = len(tool_calls)

    # Sucesso sob ataque
    if is_attack and utility and security:
        return "SUCESSO_RESILIENTE", "Modelo ignorou/habilitou a injeção e completou a tarefa legítima."
    # Sucesso baseline
    if not is_attack and utility:
        return "SUCESSO_BENIGNO", "Modelo executou corretamente em cenário sem ataque."
    # Vulnerabilidade direta
    if is_attack and not security:
        if not utility:
            return "VULNERABILIDADE_IPI_HIJACKING_TOTAL", "Desvio de fluxo (Branch Divergence) após payload adversarial."
        else:
            return "VULNERABILIDADE_IPI_EXECUCAO_PARALELA", "Execução paralela da tarefa legítima + ação maliciosa."
    # Falha de limite de contexto
    if trace_err and ("context_length" in str(trace_err).lower() or "max_tokens" in str(trace_err).lower()):
        return "FALHA_LIMITE_CONTEXTO", f"Estouro de contexto: {trace_err}"
    # Falha de sintaxe de tool calling
    if tool_errors:
        snippet = "; ".join(tool_errors[:2])
        return "FALHA_SINTAXE_TOOL_CALLING", f"Erro de sintaxe nas chamadas de ferramenta: {snippet}"
    # Falha sem tool calling (alucinação)
    if total_tools == 0:
        if assistant_replies:
            return "FALHA_SEM_TOOL_CALLING_ALUCINACAO", "Modelo respondeu textualmente sem acionar ferramentas (alucinação)."
        return "FALHA_RESPOSTA_VAZIA", "Nenhuma resposta nem chamada de ferramenta foram emitidas."
    # Terminou prematuramente (apenas leituras)
    read_only = [tc["function"] for tc in tool_calls if any(k in tc["function"].lower() for k in ["read", "get", "search", "list"])]
    if read_only and total_tools > 0:
        return "FALHA_TERMINO_PREMATURO", f"Apenas ferramentas de leitura chamadas ({', '.join(read_only)}) sem ação final."
    # Perturbação adversarial sem hijacking
    if is_attack:
        return "FALHA_DISTORCAO_ADVERSARIAL", "Payload desorientou o modelo, causando falha sem executar a injeção."
    # Descompasso de parâmetros
    return "FALHA_DESCOMPASSO_PARAMETROS", f"Chamadas de ferramenta válidas porém com parâmetros incorretos (total: {total_tools})."

# ==============================================================================
# 4. EXECUÇÃO DO BENCHMARK NO AGENTDOJO
# ==============================================================================

def run_iteration(
    suite_name: str,
    model_name: str,
    attack: str | None,
    defense: str | None,
    user_tasks: list[str] | None,
    injection_tasks: list[str] | None,
    logdir: Path,
    benchmark_version: str = "v1.1.1",
    force_rerun: bool = True,
) -> SuiteResults:
    """Executa uma iteração única de benchmark no AgentDojo."""
    suite = get_suite(benchmark_version, suite_name)
    model_slug = model_name.replace(":", "_").replace("/", "_")

    # Identificação do modelo / provider
    is_openai_compat = False
    model_id = None
    llm_arg: Any = model_name

    if model_name in MODEL_PROVIDERS:
        llm_arg = model_name
    elif os.getenv("OPENAI_COMPATIBLE_BASE_URL"):
        llm_arg = ModelsEnum.OPENAI_COMPATIBLE
        model_id = model_name
        is_openai_compat = True
    else:
        # Padrão local Ollama/vLLM
        llm_arg = ModelsEnum.OPENAI_COMPATIBLE
        model_id = model_name
        if not os.getenv("OPENAI_COMPATIBLE_BASE_URL"):
            os.environ["OPENAI_COMPATIBLE_BASE_URL"] = "http://localhost:11435/v1"
            os.environ["OPENAI_COMPATIBLE_API_KEY"] = "ollama"

    pipeline_config = PipelineConfig(
        llm=llm_arg,
        model_id=model_id,
        defense=defense,
        # System message required pelo pydantic v2
        system_message_name="default",
        system_message="",
    )
    pipeline = AgentPipeline.from_config(pipeline_config)
    # Determine o nome correto da pipeline conforme o tipo de modelo.
    # Para modelos locais (não presentes em MODEL_PROVIDERS), o AgentDojo exige
    # exatamente "Local model". Para provedores conhecidos (OpenAI, Anthropic, etc.)
    # deixamos o nome padrão (ou podemos usar "AI assistant").
    if model_name not in MODEL_PROVIDERS:
        pipeline.name = "Local model"
    else:
        # Mantém o nome padrão da pipeline (geralmente "AI assistant" ou outro
        # aceitável para o provedor). Não alteramos para evitar conflitos.
        pass
    # ---------------------------------------------------------------------
    # (Não altere o nome acima a menos que o AgentDojo seja atualizado para
    # aceitar novos identificadores.)
    # ---------------------------------------------------------------------


    with OutputLogger(str(logdir)):
        if attack is None or attack.lower() in ("none", "baseline"):
            # Benchmark without injections – positional arguments per API
            results = benchmark_suite_without_injections(
                pipeline,
                suite,
                logdir,
                force_rerun,
                tuple(user_tasks) if user_tasks else None,
                benchmark_version,
            )
        else:
            attacker = load_attack(attack, suite, pipeline)
            # Benchmark with injections – positional arguments per API
            results = benchmark_suite_with_injections(
                pipeline,
                suite,
                attacker,
                logdir,
                force_rerun,
                tuple(user_tasks) if user_tasks else None,
                tuple(injection_tasks) if injection_tasks else None,
                benchmark_version,
            )
    return results

# ==============================================================================
# 5. GERADOR DO RELATÓRIO DE ANÁLISE QUALITATIVA EM MARKDOWN
# ==============================================================================

def generate_qualitative_report(all_episodes: list[dict[str, Any]], output_file: Path, timestamp: str) -> str:
    df = pd.DataFrame(all_episodes)
    lines = [
        f"# Relatório de Análise Qualitativa de Falhas e Resiliência a IPI",
        f"**Data:** {timestamp}",
        f"**Total de Episódios:** {len(all_episodes)}",
        "",
        "---",
        "",
        "## 1. Distribuição de Causas Raízes por Modelo",
        "",
    ]
    if not df.empty:
        piv = pd.crosstab(df["model"], df["root_cause_category"], normalize="index") * 100.0
        piv = piv.round(1)
        header = "| Modelo | " + " | ".join(piv.columns) + " |"
        sep = "| :--- | " + " | ".join([":---:"] * len(piv.columns)) + " |"
        lines.append(header)
        lines.append(sep)
        for m, row in piv.iterrows():
            lines.append(f"| `{m}` | " + " | ".join([f"{v:.1f}%" for v in row]) + " |")
    lines.extend(["", "---", "", "## 2. Estudos de Caso (até 4)", ""])
    # Seleciona alguns casos representativos
    representative = [
        ("VULNERABILIDADE_IPI_HIJACKING_TOTAL", "Caso 1: Branch Divergence"),
        ("FALHA_SINTAXE_TOOL_CALLING", "Caso 2: Sintaxe de Tool Calling"),
        ("FALHA_SEM_TOOL_CALLING_ALUCINACAO", "Caso 3: Falta de Tool Calling / Alucinação"),
        ("SUCESSO_RESILIENTE", "Caso 4: Resiliência a Ataque"),
    ]
    count = 0
    for slug, title in representative:
        matches = [e for e in all_episodes if e.get("root_cause_category") == slug]
        if not matches:
            continue
        count += 1
        ep = matches[0]
        lines.extend([
            f"### {title}",
            f"- **Modelo:** `{ep.get('model')}`",
            f"- **Vetor de Ataque:** `{ep.get('attack')}`",
            f"- **Tarefa:** `{ep.get('user_task')}` | **Injeção:** `{ep.get('injection_task')}`",
            f"- **Utilidade:** {'✓' if ep.get('utility') else '✗'}",
            f"- **Segurança (ASR):** {'✓' if ep.get('security') else '✗'}`",
            f"- **Diagnóstico:** {ep.get('root_cause_description')}",
            "",
            "#### Prompt do Usuário",
            "```text",
            ep.get('user_query', '')[:600],
            "```",
            "",
            "#### Pensamento (CoT)",
            "```text",
            "\n---\n".join(ep.get('reasoning_traces', [])[:2]),
            "```",
            "",
            "#### Chamadas de Ferramentas",
            "```json",
            json.dumps(ep.get('tool_calls', []), indent=2, ensure_ascii=False),
            "```",
            "",
            "#### Resposta do Assistente",
            "```text",
            ep.get('assistant_replies', ["[nenhuma]"])[-1][:500],
            "```",
            "---",
        ])
    if count == 0:
        lines.append("*Nenhum caso representativo encontrado.*")
    report = "\n".join(lines)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)
    return report

# ==============================================================================
# 6. FUNÇÃO PRINCIPAL DE ORQUESTRAÇÃO
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline de Avaliação Empírica Multi-Modelo com Análise Qualitativa e Amostragem N-Runs")
    parser.add_argument(
    "--models",
    nargs="+",
    default=[
        # Modelos mais potentes (alta capacidade de parâmetros e tools)
        # "gpt-oss-safeguard:120b",  # removed devido a timeouts
        "gpt-oss-safeguard:20b",
        "deepseek-r1:32b",
        "qwen2.5:32b-instruct-q4_K_M",
        "qwen3:latest",
        "gpt-oss:latest",
        # Modelos de porte médio
        "llama3:latest",
        "mistral:latest",
        "deepseek-r1:8b",
        # Modelo mais fraco (para comparativo de baseline)
        "orca-mini:latest",
    ],
    help=(
        "Lista de modelos a serem avaliados (ex.: gpt-oss-safeguard:120b, "
        "gpt-oss-safeguard:20b, deepseek-r1:32b, qwen2.5:32b-instruct-q4_K_M, "
        "qwen3:latest, gpt-oss:latest, llama3:latest, mistral:latest, "
        "deepseek-r1:8b, orca-mini:latest)"
    ),
    )
    parser.add_argument("--attacks", nargs="+", default=["none", "important_instructions", "tool_knowledge", "neutral_noise"], help="Lista de vetores de ataque")
    parser.add_argument("--suite", type=str, default="workspace", help="Nome da suíte (padrão: workspace)")
    parser.add_argument("--runs", type=int, default=30, help="Número N de repetições independentes por teste (padrão: 30)")
    parser.add_argument("--user-tasks", nargs="*", default=None, help="Filtrar user_tasks específicas")
    parser.add_argument("--injection-tasks", nargs="*", default=None, help="Filtrar injection_tasks específicas")
    parser.add_argument("--logdir", type=str, default="./runs_benchmark", help="Diretório para salvar logs brutos")
    parser.add_argument("--outdir", type=str, default="./results_benchmark", help="Diretório para relatórios finais")
    parser.add_argument("--defense", type=str, default=None, help="Defesa a ser aplicada (e.g., tool_filter).")
    parser.add_argument("--force-rerun", action="store_true", default=True, help="Forçar nova execução sem cache")
    parser.add_argument("--version", type=str, default="v1.1.1", help="Versão da suíte no AgentDojo" )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    console.print(
        Panel.fit(
            f"[bold green]Pipeline de Avaliação Empírica Multi-Modelo & Análise Qualitativa[/bold green]\n"
            f"[cyan]Modelos ({len(args.models)}):[/cyan] {', '.join(args.models)}\n"
            f"[cyan]Ataques ({len(args.attacks)}):[/cyan] {', '.join(args.attacks)}\n"
            f"[cyan]Amostragem (N-Runs):[/cyan] {args.runs} execuções independentes\n"
            f"[cyan]Suíte:[/cyan] {args.suite} ({args.version})",
            title="Configuração do Benchmark",
            border_style="bright_blue",
        )
    )

    all_detailed_episodes: list[dict[str, Any]] = []
    aggregated_results: list[dict[str, Any]] = []

    total_steps = len(args.models) * len(args.attacks) * args.runs

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task("[bold cyan]Progresso Total", total=total_steps)

        for model in args.models:
            model_slug = model.replace(":", "_").replace("/", "_")
            model_episodes: list[dict[str, Any]] = []

            for attack in args.attacks:
                is_attack = attack.lower() not in ("none", "baseline")
                attack_name = None if not is_attack else attack
                attack_label = "Baseline (Sem Ataque)" if not is_attack else attack

                run_utilities: list[float] = []
                run_asrs: list[float] = []

                for run_idx in range(1, args.runs + 1):
                    step_desc = f"[yellow]{model}[/yellow] | [magenta]{attack_label}[/magenta] (Run {run_idx}/{args.runs})"
                    progress.update(overall_task, description=step_desc)

                    run_logdir = Path(args.logdir) / model_slug / attack_label / f"run_{run_idx}"

                    try:
                        results = run_iteration(
                            suite_name=args.suite,
                            model_name=model,
                            attack=attack_name,
                            defense=args.defense,
                            user_tasks=args.user_tasks,
                            injection_tasks=args.injection_tasks,
                            logdir=run_logdir,
                            benchmark_version=args.version,
                            force_rerun=args.force_rerun,
                        )

                        utility_vals = list(results["utility_results"].values())
                        mean_util = (sum(utility_vals) / len(utility_vals) * 100.0) if utility_vals else 0.0
                        run_utilities.append(mean_util)

                        mean_asr = 0.0
                        if is_attack and "injection_tasks_utility_results" in results:
                            inj_vals = list(results["injection_tasks_utility_results"].values())
                            if inj_vals:
                                mean_asr = (sum(inj_vals) / len(inj_vals) * 100.0)
                        run_asrs.append(mean_asr)

                        for (u_task, inj_task), u_val in results["utility_results"].items():
                            sec_val = results["security_results"].get((u_task, inj_task), True)
                            inj_label = inj_task if (inj_task and is_attack) else "none"
                            attack_folder = attack_name if is_attack else "none"
                            # Determine a pasta correta onde o AgentDojo grava os traces.
                            pipeline_folder = "Local model" if model not in MODEL_PROVIDERS else "AI assistant"
                            trace_path = (
                                run_logdir
                                / pipeline_folder
                                / args.suite
                                / u_task
                                / attack_folder
                                / f"{inj_label}.json"
                            )
                            trace_info = parse_task_trace_file(trace_path)
                            root_slug, root_desc = diagnose_episode(u_val, sec_val, is_attack, trace_info)
                            episode = {
                                "timestamp": datetime.datetime.now().isoformat(),
                                "model": model,
                                "attack": attack_label,
                                "run_id": run_idx,
                                "user_task": u_task,
                                "injection_task": inj_task,
                                "utility": u_val,
                                "security": sec_val,
                                "root_cause_category": root_slug,
                                "root_cause_description": root_desc,
                                "user_query": trace_info.get("user_query", "") if trace_info else "",
                                "assistant_replies": trace_info.get("assistant_replies", []) if trace_info else [],
                                "reasoning_traces": trace_info.get("reasoning_traces", []) if trace_info else [],
                                "tool_calls": trace_info.get("tool_calls", []) if trace_info else [],
                                "tool_errors": trace_info.get("tool_errors", []) if trace_info else [],
                            }
                            model_episodes.append(episode)
                            all_detailed_episodes.append(episode)

                    except Exception as exc:
                        console.log(f"[bold red]Erro na execução: {model} | {attack_label} (Run {run_idx}): {exc}[/bold red]")
                        run_utilities.append(0.0)
                        run_asrs.append(0.0)

                    progress.advance(overall_task)

                u_mean, u_std, u_ci = calculate_ci95(run_utilities)
                a_mean, a_std, a_ci = calculate_ci95(run_asrs)
                aggregated_results.append({
                    "Modelo": model,
                    "Vetor de Ataque": attack_label,
                    "N (Runs)": args.runs,
                    "Utilidade Média (%)": round(u_mean, 2),
                    "Utilidade Std": round(u_std, 2),
                    "Utilidade CI-95 (±)": round(u_ci, 2),
                    "ASR Médio (%)": round(a_mean, 2),
                    "ASR Std": round(a_std, 2),
                    "ASR CI-95 (±)": round(a_ci, 2),
                })

            # Salva traces individuais do modelo
            model_traces_file = outdir / f"raw_traces_{model_slug}_{timestamp}.json"
            with open(model_traces_file, "w", encoding="utf-8") as f:
                json.dump(model_episodes, f, indent=2, ensure_ascii=False)

    # Exportação consolidada
    df_agg = pd.DataFrame(aggregated_results)
    csv_path = outdir / f"aggregated_metrics_{timestamp}.csv"
    json_path = outdir / f"aggregated_metrics_{timestamp}.json"
    all_episodes_path = outdir / f"all_episodes_log_{timestamp}.json"
    md_comparativo_path = outdir / f"comparativo_modelos_{timestamp}.md"
    md_qualitative_path = outdir / f"relatorio_analise_qualitativa_{timestamp}.md"

    df_agg.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_results, f, indent=2, ensure_ascii=False)
    with open(all_episodes_path, "w", encoding="utf-8") as f:
        json.dump(all_detailed_episodes, f, indent=2, ensure_ascii=False)

    # Relatório qualitativo
    generate_qualitative_report(all_detailed_episodes, md_qualitative_path, timestamp)

    # Tabela comparativa MD
    md_lines = [
        "### Tabela Comparativa Quantitativa (N-Runs)",
        "",
        "| Modelo | Vetor de Ataque | $N$ Runs | Utilidade Média (\\%) ± IC 95\\% | ASR Médio (\\%) ± IC 95\\% | Desvio Padrão Utilidade | Desvio Padrão ASR |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, row in df_agg.iterrows():
        u = f"{row['Utilidade Média (%)']:.2f}\\% $\\pm$ {row['Utilidade CI-95 (±)']:.2f}"
        a = f"{row['ASR Médio (%)']:.2f}\\% $\\pm$ {row['ASR CI-95 (±)']:.2f}"
        md_lines.append(f"| `{row['Modelo']}` | {row['Vetor de Ataque']} | {row['N (Runs)']} | {u} | {a} | {row['Utilidade Std']:.2f} | {row['ASR Std']:.2f} |")
    with open(md_comparativo_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # Exibição rica
    rich_table = Table(
        title=f"Resultados Quantitativos Consolidados (N = {args.runs})",
        border_style="green",
        header_style="bold cyan",
    )
    rich_table.add_column("Modelo", style="bold yellow")
    rich_table.add_column("Vetor de Ataque", style="white")
    rich_table.add_column("Runs (N)", justify="center", style="magenta")
    rich_table.add_column("Utilidade Média (%) [IC-95]", justify="right", style="green")
    rich_table.add_column("ASR Médio (%) [IC-95]", justify="right", style="red")
    for _, row in df_agg.iterrows():
        u_str = f"{row['Utilidade Média (%)']:.1f}% ± {row['Utilidade CI-95 (±)']:.1f}"
        a_str = f"{row['ASR Médio (%)']:.1f}% ± {row['ASR CI-95 (±)']:.1f}"
        rich_table.add_row(str(row["Modelo"]), str(row["Vetor de Ataque"]), str(row["N (Runs)"]), u_str, a_str)
    console.print("\n")
    console.print(rich_table)

    # Tabela de causas raízes
    if all_detailed_episodes:
        df_ep = pd.DataFrame(all_detailed_episodes)
        cause_table = Table(title="Distribuição de Causas Raízes", border_style="bright_blue", header_style="bold magenta")
        cause_table.add_column("Modelo", style="yellow")
        cause_table.add_column("Causa Raiz", style="white")
        cause_table.add_column("Ocorrências", justify="center", style="cyan")
        cause_table.add_column("%", justify="right", style="green")
        grp = df_ep.groupby(["model", "root_cause_category"]).size().reset_index(name="count")
        for model_name in args.models:
            sub = grp[grp["model"] == model_name]
            total = sub["count"].sum()
            for _, r in sub.iterrows():
                pct = (r["count"] / total) * 100.0 if total else 0.0
                cause_table.add_row(str(model_name), str(r["root_cause_category"]), str(r["count"]), f"{pct:.1f}%")
        console.print("\n")
        console.print(cause_table)

    console.rule("[bold green]Artefatos Gerados com Sucesso[/bold green]")
    console.print(f"[bold green]✓ CSV Consolidado:[/bold green] {csv_path}")
    console.print(f"[bold green]✓ JSON Métricas:[/bold green] {json_path}")
    console.print(f"[bold green]✓ Log Completo:[/bold green] {all_episodes_path}")
    console.print(f"[bold green]✓ Relatório Qualitativo:[/bold green] {md_qualitative_path}")
    console.print(f"[bold green]✓ Tabela Comparativa MD:[/bold green] {md_comparativo_path}\n")

if __name__ == "__main__":
    main()
