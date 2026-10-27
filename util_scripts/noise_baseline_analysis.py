#!/usr/bin/env python3
"""
Análise Comparativa de Ruído para o Experimento de Controle de Ruído (Noise Baseline).

Este script compara os resultados de:
  1. Baseline (sem ruído/injeção): utilidade = 100% (10/10 tarefas)
  2. Ataque tool_knowledge: utilidade = 67.9%, ASR = 0.7%
  3. Ruído neutro (neutral_noise): a ser executado

Autor: Pedro Henrique Silva de Albuquerque
TCC - Análise da Corrupção de Lógica Condicional em Agentes Baseados em LLM Sob Ataques de IPI
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def load_results(logdir: Path, is_baseline: bool = False) -> dict:
    """Carrega resultados de um diretório de logs do AgentDojo."""
    results = defaultdict(list)

    if not logdir.exists():
        print(f"AVISO: Diretório {logdir} não encontrado.")
        return results

    clean_tasks = {
        "user_task_1", "user_task_4", "user_task_5", "user_task_6",
        "user_task_10", "user_task_11", "user_task_22", "user_task_24",
        "user_task_28", "user_task_29",
    }

    for json_file in logdir.rglob("*.json"):
        # Ignorar tarefas de injeção isoladas
        if "injection_task_" in json_file.parent.parent.name and not is_baseline:
            continue

        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            user_task = data.get("user_task_id")
            if not user_task or user_task not in clean_tasks:
                continue

            utility = data.get("utility", False)
            security = data.get("security", False)
            duration = data.get("duration", 0)
            attack_type = data.get("attack_type") or "baseline"

            results[user_task].append({
                "attack": attack_type,
                "utility": utility,
                "security": security,
                "duration": duration,
                "file": str(json_file.relative_to(logdir)),
            })
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"AVISO: Erro ao processar {json_file}: {e}")

    return dict(results)


def compute_aggregates(results: dict) -> dict:
    """Calcula métricas agregadas por tipo de ataque."""
    by_attack = defaultdict(lambda: {"utility": [], "security": [], "durations": []})

    for task, runs in results.items():
        for run in runs:
            attack = run["attack"]
            by_attack[attack]["utility"].append(run["utility"])
            by_attack[attack]["security"].append(run["security"])
            by_attack[attack]["durations"].append(run["duration"])

    aggregates = {}
    for attack, metrics in by_attack.items():
        n = len(metrics["utility"])
        aggregates[attack] = {
            "n": n,
            "utility_rate": sum(metrics["utility"]) / n if n > 0 else 0,
            "security_rate": sum(metrics["security"]) / n if n > 0 else 0,  # ASR
            "avg_duration": sum(metrics["durations"]) / n if n > 0 else 0,
        }

    return aggregates


def main():
    base_dir = Path(__file__).parent.parent / "runs"

    print("=" * 80)
    print("ANÁLISE COMPARATIVA DE RUÍDO - EXPERIMENTO DE CONTROLE DE RUÍDO")
    print("=" * 80)
    print()

    # Definir os cenários e caminhos
    scenarios = {
        "Baseline (limpo)": {
            "path": base_dir / "baseline_7b_full",
            "is_baseline": True,
            "filter_attack": None,
        },
        "important_instructions": {
            "path": base_dir / "attack_important_instructions",
            "is_baseline": False,
            "filter_attack": "important_instructions",
        },
        "tool_knowledge": {
            "path": base_dir / "attack_important_instructions",
            "is_baseline": False,
            "filter_attack": "tool_knowledge",
        },
        "neutral_noise (controle)": {
            "path": base_dir / "baseline_7b_noise",
            "is_baseline": False,
            "filter_attack": "neutral_noise",
        },
    }

    all_results = {}

    for name, config in scenarios.items():
        path = config["path"]
        is_baseline = config["is_baseline"]
        filter_attack = config["filter_attack"]

        print(f"Carregando: {name} ({path})")
        raw = load_results(path, is_baseline=is_baseline)

        if filter_attack is not None:
            filtered = {}
            for task, runs in raw.items():
                matching = [r for r in runs if r["attack"] == filter_attack]
                if matching:
                    filtered[task] = matching
            all_results[name] = filtered
        else:
            all_results[name] = raw

    print()

    # Calcular agregados
    aggregates = {}
    for name, results in all_results.items():
        agg = compute_aggregates(results)
        if agg:
            # Pegar o primeiro (e geralmente único) ataque por cenário
            first_attack = list(agg.keys())[0]
            aggregates[name] = agg[first_attack]

    # Imprimir tabela comparativa
    print("-" * 80)
    print("RESULTADOS COMPARATIVOS DE UTILIDADE")
    print("-" * 80)
    print(f"{'Cenário':<30} {'N':>6} {'Utilidade':>12} {'ASR':>12} {'Duração Média':>15}")
    print("-" * 80)

    for name, agg in aggregates.items():
        util_pct = agg["utility_rate"] * 100
        asr_pct = agg["security_rate"] * 100
        dur = agg["avg_duration"]
        print(f"{name:<30} {agg['n']:>6} {util_pct:>11.1f}% {asr_pct:>11.1f}% {dur:>14.1f}s")

    print("-" * 80)

    # Análise interpretativa
    print()
    print("ANÁLISE INTERPRETATIVA:")
    print("=" * 80)

    baseline_util = aggregates.get("Baseline (limpo)", {}).get("utility_rate", 1.0)
    noise_util = aggregates.get("neutral_noise", {}).get("utility_rate", None)
    tk_util = aggregates.get("tool_knowledge", {}).get("utility_rate", None)

    if noise_util is not None and tk_util is not None:
        noise_degradation = (baseline_util - noise_util) * 100
        tk_degradation = (baseline_util - tk_util) * 100

        print(f"Utilidade baseline (sem ruído):    {baseline_util*100:.1f}%")
        print(f"Utilidade com ruído neutro:        {noise_util*100:.1f}% (degradação: {noise_degradation:.1f}pp)")
        print(f"Utilidade com tool_knowledge:      {tk_util*100:.1f}% (degradação: {tk_degradation:.1f}pp)")
        print()

        if noise_degradation < 5.0:
            print("CONCLUSÃO: O ruído neutro causa degradação insignificante (<5pp) na utilidade.")
            print("Isso indica que o modelo local consegue processar contexto extenso sem perda funcional.")
        elif noise_degradation < tk_degradation:
            print("CONCLUSÃO: O ruído neutro causa degradação moderada, mas MENOR que tool_knowledge.")
            print("A diferença (tool_knowledge - ruído) é atribuída ao conteúdo adversarial.")
        else:
            print("CONCLUSÃO: O ruído neutro causa degradação similar ou MAIOR que tool_knowledge.")
            print("Isso sugere que o tamanho do contexto (e não o conteúdo adversarial) é o fator principal.")

        if tk_util < noise_util:
            diff = (noise_util - tk_util) * 100
            print(f"\nA diferença entre ruído neutro e tool_knowledge é de {diff:.1f}pp.")
            print("Essa diferença pode ser atribuída ao efeito adversarial específico do ataque.")
        elif tk_util >= noise_util:
            print("\nAVISO: tool_knowledge mantém/utiliza utilidade igual ou superior ao ruído neutro.")
            print("Isso pode indicar que o ataque não causa degradação adicional além do ruído.")

    print()
    print("=" * 80)
    print("FIM DA ANÁLISE")
    print("=" * 80)


if __name__ == "__main__":
    main()
