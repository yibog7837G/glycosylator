from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import shutil

import numpy as np
import pandas as pd
import glycosylator as gl


@dataclass(frozen=True)
class OptimizationConfig:
    n_runs: int = 12
    n_ancestors: int = 10
    neighbor_cutoff: float = 15.0
    radius: float = 15.0
    pushback: float = 4.0
    n_split: int = 4
    repulsion_distance: float = 1.0
    repulsion_weight: float = 25.0
    repulsion_power: float = 2.0
    overlap_distance: float | None = None
    overlap_weight: float = 1.0
    hollow_out_cutoff: float = 0.75
    early_stop_score: int = 0


def score_clashes(glycoprotein) -> int:
    score = 0
    glycans = getattr(glycoprotein, "get_glycans", lambda: {})()
    for glycan in glycans.values():
        if hasattr(glycan, "count_clashes"):
            score += int(glycan.count_clashes())
        elif hasattr(glycan, "clashes_with_scaffold") and glycan.clashes_with_scaffold():
            score += 1
    return score


def ensure_asn_nd2_bonds(protein, residue) -> None:
    nd2 = residue.get_atom("ND2")
    if not nd2:
        raise ValueError("目标残基上找不到 ND2 原子！(请确认该残基是 ASN)")

    neighbors = protein.get_neighbors(nd2)
    for h_name in ["HD21", "HD22", "HD2", "1HD2", "2HD2"]:
        try:
            h_atom = residue.get_atom(h_name)
            if h_atom and h_atom not in neighbors:
                protein.add_bond(nd2, h_atom)
        except Exception:
            continue


def optimize_once(glycoprotein, config: OptimizationConfig):
    graph, edges = gl.optimizers.make_scaffold_graph(
        glycoprotein,
        only_clashing_glycans=False,
        include_root=True,
        include_n_ancestors=config.n_ancestors,
        neighbor_cutoff=config.neighbor_cutoff,
    )

    env = gl.optimizers.DistanceRotatron(
        graph,
        edges,
        pushback=config.pushback,
        radius=config.radius,
    )

    glycoprotein = gl.optimizers.optimize(glycoprotein, env)

    split = gl.optimizers.split_environment(env, config.n_split)
    split = [
        gl.optimizers.ScaffoldRotatron(
            rot,
            repulsion_distance=config.repulsion_distance,
            repulsion_weight=config.repulsion_weight,
            repulsion_power=config.repulsion_power,
            overlap_distance=config.overlap_distance,
            overlap_weight=config.overlap_weight,
        )
        for rot in split
    ]

    return gl.optimizers.parallel_optimize(glycoprotein, split)


def optimize_glycoprotein(glycoprotein, config: OptimizationConfig):
    best_gp = None
    best_score = None
    best_run = None

    base_seed = random.randint(1, 10_000_000)
    for run_idx in range(1, config.n_runs + 1):
        seed = base_seed + run_idx
        random.seed(seed)
        np.random.seed(seed)

        gp_try = glycoprotein.copy()
        gp_try.hollow_out(cutoff=config.hollow_out_cutoff)
        gp_try = optimize_once(gp_try, config)
        gp_try.fill()
        score = score_clashes(gp_try)
        print(f"      - run {run_idx}/{config.n_runs}: clash_score = {score}")

        if best_score is None or score < best_score:
            best_score = score
            best_gp = gp_try
            best_run = run_idx

        if score <= config.early_stop_score:
            break

    if best_gp is None:
        raise RuntimeError("所有优化尝试都失败，无法得到结果。")

    print(f"    ✅ 最终选择 run {best_run}，clash_score={best_score}")
    return best_gp


def find_pdb(input_dir: Path, base_name: str) -> Path:
    matches = sorted(input_dir.glob(f"{base_name}_*.pdb"))
    if not matches:
        raise FileNotFoundError(f"找不到对应的 PDB 文件 ({base_name})")
    return matches[0]


def get_target_residue(protein, chain_id: str, site_int: int):
    residue = protein.get_residue(site_int, chain=chain_id)
    if residue:
        return residue

    sites = protein.find_n_linked_sites()
    chain = protein.get_chain(chain_id)
    chain_sites = sites.get(chain, []) if chain else []
    for site in chain_sites:
        if getattr(site, "seqid", None) == site_int:
            return site
    raise ValueError(f"在 {chain_id} 链找不到第 {site_int} 号残基！")


def is_skip_glycan(glycan_type: str, iupac_seq: str, glycan_site) -> bool:
    return glycan_type.lower() == "no" or iupac_seq == "N/A" or pd.isna(glycan_site)


def process_row(row, input_pdb_dir: Path, output_pdb_dir: Path, indi_pdb_dir: Path, config: OptimizationConfig):
    glycan_type = str(row.get("glycan", "")).strip()
    iupac_seq = str(row.get("IUPAC", "")).strip()
    yaml_filename = str(row.get("yaml_filename", "")).strip()
    glycan_site = row.get("glycan_site")

    base_name = yaml_filename.replace(".yaml", "")
    pdb_path = find_pdb(input_pdb_dir, base_name)

    if is_skip_glycan(glycan_type, iupac_seq, glycan_site):
        output_pdb_path = output_pdb_dir / f"{base_name}_unmodified.pdb"
        output_pdb_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdb_path, output_pdb_path)
        indi_path = indi_pdb_dir / base_name
        indi_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_pdb_path, indi_path / "origin.pdb")
        print(f"⏩ 迁移: {base_name} (无糖链结构，已原样复制)")
        return True

    site_int = int(float(glycan_site))
    print(f"🔄 正在处理: {base_name} (位点: C链 {site_int})")

    complex_glycan = gl.glycan(iupac_seq)
    protein = gl.Protein.from_pdb(str(pdb_path))

    target_residue = get_target_residue(protein, "C", site_int)
    ensure_asn_nd2_bonds(protein, target_residue)

    glycoprotein = gl.glycosylate(protein, complex_glycan, residues=[target_residue])

    print("    ⚙️ 正在启动遗传算法进行构象优化 (消除空间位阻，约需几分钟)...")
    try:
        glycoprotein = optimize_glycoprotein(glycoprotein, config)
    except Exception as opt_err:
        print(f"    ⚠️ 构象优化过程中出现问题: {opt_err}，将回退使用初始构象。")

    output_pdb_path = output_pdb_dir / f"{base_name}_glycosylated.pdb"
    output_pdb_path.parent.mkdir(parents=True, exist_ok=True)
    glycoprotein.to_pdb(str(output_pdb_path))

    indi_path = indi_pdb_dir / base_name
    indi_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_pdb_path, indi_path / "origin.pdb")
    print(f"   ✅ 成功！已保存至 -> {output_pdb_path.name}")
    return True


def main():
    excel_path = Path("/data/gyb/glycosylator/use/MHCI/MHCI_Enriched_Results.xlsx")
    input_pdb_dir = Path("/data/gyb/glycosylator/use/MHCI/AF_test_fixed_pdb")
    output_pdb_dir = Path("/data/gyb/glycosylator/use/MHCI/best_models_glycan_af_test")
    indi_pdb_dir = Path("/data/gyb/glycosylator/use/MHCI/best_models_indi_glycan_af_test")

    output_pdb_dir.mkdir(parents=True, exist_ok=True)
    indi_pdb_dir.mkdir(parents=True, exist_ok=True)

    config = OptimizationConfig(
        n_runs=16,
        repulsion_distance=1.2,
        repulsion_weight=50.0,
        repulsion_power=2.0,
        overlap_distance=0.9,
        overlap_weight=10000.0,
        hollow_out_cutoff=0.75,
    )

    print(f"正在读取花名册: {excel_path}...")
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"❌ 读取 Excel 失败: {e}")
        return

    success_count = 0
    fail_count = 0
    print(f"\n🚀 开始批量加糖任务，总计 {len(df)} 条记录...\n")

    for _, row in df.iterrows():
        try:
            if process_row(row, input_pdb_dir, output_pdb_dir, indi_pdb_dir, config):
                success_count += 1
        except Exception as e:
            print(f"   ❌ 失败 ({row.get('yaml_filename', 'unknown')}): {e}")
            fail_count += 1

    print("\n" + "=" * 40)
    print("🎉 批量加糖任务完成！")
    print(f"总计处理成功: {success_count} 个")
    print(f"总计处理失败: {fail_count} 个")
    print(f"输出文件夹: {output_pdb_dir}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
