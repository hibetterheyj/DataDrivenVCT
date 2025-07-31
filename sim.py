"""
python sim.py
python sim.py --multi --num_simulations 100000
python sim.py --multi --num_simulations 10000 --region pacific
"""
import os
import json
import random
import argparse
import concurrent.futures
import time
from pathlib import Path
from collections import defaultdict
from math import comb
from typing import List, Dict, Tuple, Optional, Any, Set, Union, Callable
import yaml
import requests
import graphviz
import tqdm

# 全局 debug 变量
debug: bool = False

def load_yaml(file_path: str) -> Any:
    """
    加载YAML文件

    参数:
        file_path: YAML文件路径

    返回:
        YAML文件解析后的内容（通常为字典或列表）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_group_teams(yaml_folder: str, region: str = 'cn') -> Tuple[List[str], List[str]]:
    """
    返回预设的两个小组

    参数:
        yaml_folder: YAML文件所在文件夹
        region: 赛区名称，默认'cn'

    返回:
        包含两个小组队伍列表的元组 (Alpha组, Omega组)
    """
    file_path = Path(yaml_folder) / region / 'groups.yaml'
    groups = load_yaml(str(file_path))
    if debug:
        print(f"加载分组文件: {file_path}")
        print(groups)
    return groups['Alpha'], groups['Omega']

def load_initial_pts(yaml_folder: str, region: str = 'cn') -> Dict[str, int]:
    """
    加载初始积分

    参数:
        yaml_folder: YAML文件所在文件夹
        region: 赛区名称，默认'cn'

    返回:
        包含各队初始积分的字典 {队伍名: 积分}
    """
    file_path = Path(yaml_folder) / region / 'initial_pts.yaml'
    return load_yaml(str(file_path))

def load_real_results(
    source: str = "local",
    results_file: str = "results.yaml",
    yaml_folder: str = "./yaml",
    region: str = 'cn'
) -> Dict[str, List[Any]]:
    """
    从本地文件或网络API加载真实比赛结果

    参数:
        source: 数据来源，'local'或'online'
        results_file: 本地结果文件名
        yaml_folder: YAML文件所在文件夹
        region: 赛区名称

    返回:
        包含常规赛和季后赛结果的字典，格式为:
        {
            'regular_season': [...],  # 常规赛结果
            'playoffs': [...]        # 季后赛结果
        }
    """
    if source == "local":
        try:
            data = load_yaml(os.path.join(yaml_folder, region, results_file))
            if 'playoffs' not in data:
                data['playoffs'] = []
            return data
        except FileNotFoundError:
            if debug:
                print("本地数据文件未找到，将使用模拟数据")
            return {'regular_season': [], 'playoffs': []}
        except Exception as e:
            if debug:
                print(f"加载本地数据失败: {e}")
            return {'regular_season': [], 'playoffs': []}
    else:  # 从网络加载
        try:
            response = requests.get("https://api.example.com/vct_results") # 示例URL
            data = response.json()
            if 'playoffs' not in data:
                data['playoffs'] = []
            return data
        except Exception as e:
            if debug:
                print(f"加载网络数据失败: {e}")
            return {'regular_season': [], 'playoffs': []}

def get_bo_score_probs(bo: int, p: float) -> Dict[str, float]:
    """
    计算不同BO赛制下各比分的理论概率分布

    参数:
        bo: 比赛局数，3或5
        p: 队伍A每局获胜的概率

    返回:
        字典，包含各比分的概率分布，键为比分字符串如"2:0"，值为概率
    """
    if bo not in {3, 5}:
        raise ValueError("bo参数必须是3或5")
    required_wins = (bo + 1) // 2
    results: Dict[str, float] = {}
    for total_games in range(required_wins, bo + 1):
        losses = total_games - required_wins
        if losses >= required_wins:
            continue
        comb_val = comb(total_games - 1, required_wins - 1)
        prob_a = comb_val * (p ** required_wins) * ((1 - p) ** losses)
        a_score = f"{required_wins}:{losses}"
        results[a_score] = prob_a
        prob_b = comb_val * ((1 - p) ** required_wins) * (p ** losses)
        b_score = f"{losses}:{required_wins}"
        results[b_score] = prob_b
    total = sum(results.values())
    return {k: v/total for k, v in results.items()}

def simulate_match(
    team1: str,
    team2: str,
    bo: int = 3,
    team1_win_rate: float = 0.5
) -> Tuple[str, Tuple[int, int]]:
    """
    模拟一场BO3或BO5比赛，返回比分结果

    参数:
        team1: 队伍A的名称
        team2: 队伍B的名称
        bo: 比赛局数，3或5（默认3）
        team1_win_rate: 队伍A每局获胜的概率（默认0.5）

    返回:
        元组 (胜者, 比分)，比分格式为(胜场, 负场)
    """
    if bo not in {3, 5}:
        raise ValueError("bo参数必须是3或5")
    if not (0 <= team1_win_rate <= 1):
        raise ValueError("team1_win_rate必须在0到1之间")
    theoretical_probs = get_bo_score_probs(bo, team1_win_rate)
    rand = random.random()
    cumulative = 0.0
    for score, prob in theoretical_probs.items():
        cumulative += prob
        if rand < cumulative:
            wins1, wins2 = map(int, score.split(':'))
            winner = team1 if wins1 > wins2 else team2
            return winner, (wins1, wins2)
    # 默认返回 (理论上不会执行到这里)
    required_wins = (bo + 1) // 2
    losses = random.randint(0, required_wins-1)
    winner = team1 if random.random() < team1_win_rate else team2
    if winner == team1:
        return winner, (required_wins, losses)
    else:
        return winner, (losses, required_wins)

def load_map_pool(yaml_folder: str, region: str = 'cn') -> List[str]:
    """
    加载地图池配置，返回地图名称列表（兼容旧格式）

    参数:
        yaml_folder: YAML文件所在文件夹
        region: 赛区名称，默认'cn'

    返回:
        地图名称列表
    """
    file_path = Path(yaml_folder) / 'map_pool.yaml' # 注意：移除了region子目录
    try:
        data = load_yaml(str(file_path))
        if data and isinstance(data, list) and isinstance(data[0], dict):
            return [item["name"] for item in data]
        elif data and isinstance(data, list) and isinstance(data[0], str):
            return data
        else:
            raise ValueError("Invalid map pool format")
    except (FileNotFoundError, ValueError):
        return [
            "Ascent",
            "Bind",
            "Corrode",
            "Haven",
            "Icebox",
            "Lotus",
            "Sunset"
        ]

def _parse_score_from_result(result_data: Any) -> Tuple[int, int]:
    """
    从真实数据的结果部分解析出地图比分

    参数:
        result_data: 比赛结果数据，可以是列表或元组

    返回:
        元组 (队伍1胜场, 队伍2胜场)
    """
    if isinstance(result_data, list) and len(result_data) > 0 and isinstance(result_data[0], (list, tuple)):
        # 地图比分格式：[[13,11], [8,13], [13,10]]
        team1_maps = sum(1 for r in result_data if r[0] > r[1])
        team2_maps = sum(1 for r in result_data if r[1] > r[0])
        return team1_maps, team2_maps
    else:
        # 传统比分格式：[2, 0] 或 (2, 1)
        return result_data[0], result_data[1]

def play_regular_season(
    group: List[str],
    use_real_data: bool = False,
    map_based: bool = False,
    map_pool: Optional[List[str]] = None,
    real_results: Optional[Dict[str, List[Any]]] = None
) -> Tuple[
    Dict[str, int],
    Dict[str, str],
    Dict[str, str],
    Dict[str, Dict[str, int]],
    Dict[str, Dict[str, int]],
    List[Tuple[str, str, str, Tuple[int, int]]]
]:
    """
    进行常规赛：每组内每支队伍与同组其他队伍各打一场比赛

    参数:
        group: 小组内的队伍列表
        use_real_data: 是否使用真实比赛数据
        map_based: 是否基于地图进行模拟
        map_pool: 地图池列表
        real_results: 真实比赛结果数据

    返回:
        元组包含以下内容:
        - 各队积分 {队伍: 积分}
        - 各队胜负记录 {队伍: "X胜-Y负"}
        - 各队地图差 {队伍: "+X"或"-X"}
        - 各队详细统计 {队伍: {统计项: 值}}
        - 队伍间交锋记录 {队伍: {对手: 胜场}}
        - 比赛记录列表 [(队1, 队2, 胜者, (比分1, 比分2)), ...]
    """
    pts: Dict[str, int] = {team: 0 for team in group}
    win_loss: Dict[str, List[int]] = {team: [0, 0] for team in group}
    map_diff: Dict[str, int] = {team: 0 for team in group}
    head_to_head: Dict[str, Dict[str, int]] = {team: {} for team in group}
    match_records: List[Tuple[str, str, str, Tuple[int, int]]] = []
    team_stats: Dict[str, Dict[str, int]] = {
        team: {
            'wins': 0,
            'losses': 0,
            'maps_won': 0,
            'maps_lost': 0
        } for team in group
    }
    played_matches: Set[Tuple[str, str]] = set()

    if use_real_data and real_results and real_results['regular_season']:
        if debug:
            print("\n使用真实常规赛数据")
        for match in real_results['regular_season']:
            team1, team2, result = match
            if team1 in group and team2 in group and result is not None:
                played_matches.add(tuple(sorted([team1, team2])))
                team1_maps, team2_maps = _parse_score_from_result(result)
                winner = team1 if team1_maps > team2_maps else team2
                pts[winner] += 1
                win_loss[winner][0] += 1
                win_loss[team2 if winner == team1 else team1][1] += 1
                team_stats[team1]['maps_won'] += team1_maps
                team_stats[team1]['maps_lost'] += team2_maps
                team_stats[team2]['maps_won'] += team2_maps
                team_stats[team2]['maps_lost'] += team1_maps
                if winner == team1:
                    team_stats[team1]['wins'] += 1
                    team_stats[team2]['losses'] += 1
                    head_to_head[team1][team2] = 1
                    head_to_head[team2][team1] = 0
                else:
                    team_stats[team2]['wins'] += 1
                    team_stats[team1]['losses'] += 1
                    head_to_head[team2][team1] = 1
                    head_to_head[team1][team2] = 0
                if debug:
                    print(f"{team1} vs {team2} -> 比分: {team1_maps}:{team2_maps} 胜者: {winner}")
                match_records.append((team1, team2, winner, (team1_maps, team2_maps)))

    if debug:
        print("\n模拟常规赛")
    for i, team1 in enumerate(group):
        for team2 in group[i + 1:]:
            match = tuple(sorted([team1, team2]))
            if match not in played_matches:
                if map_based:
                    winner, (team1_maps, team2_maps) = simulate_match(
                        team1, team2, bo=3, team1_win_rate=0.5
                    )
                else:
                    winner = team1 if random.choice([True, False]) else team2
                    if winner == team1:
                        team1_maps = 2
                        team2_maps = random.choice([0, 1])
                    else:
                        team2_maps = 2
                        team1_maps = random.choice([0, 1])

                loser = team2 if winner == team1 else team1
                pts[winner] += 1
                win_loss[winner][0] += 1
                win_loss[loser][1] += 1
                team_stats[team1]['maps_won'] += team1_maps
                team_stats[team1]['maps_lost'] += team2_maps
                team_stats[team2]['maps_won'] += team2_maps
                team_stats[team2]['maps_lost'] += team1_maps
                if winner == team1:
                    team_stats[team1]['wins'] += 1
                    team_stats[team2]['losses'] += 1
                    head_to_head[team1][team2] = 1
                    head_to_head[team2][team1] = 0
                else:
                    team_stats[team2]['wins'] += 1
                    team_stats[team1]['losses'] += 1
                    head_to_head[team2][team1] = 1
                    head_to_head[team1][team2] = 0
                if debug:
                    print(f"{team1} vs {team2} -> 比分: {team1_maps}:{team2_maps} 胜者: {winner}")
                match_records.append((team1, team2, winner, (team1_maps, team2_maps)))
                played_matches.add(match)

    for team in group:
        map_diff[team] = team_stats[team]['maps_won'] - team_stats[team]['maps_lost']

    win_loss_dict: Dict[str, str] = {team: f"{win_loss[team][0]}胜-{win_loss[team][1]}负" for team in group}
    map_diff_dict: Dict[str, str] = {team: f"+{map_diff[team]}" if map_diff[team] > 0 else str(map_diff[team]) for team in group}
    return pts, win_loss_dict, map_diff_dict, team_stats, head_to_head, match_records

def compare_teams(
    team1: str,
    team2: str,
    team_stats: Dict[str, Dict[str, int]],
    head_to_head: Dict[str, Dict[str, int]]
) -> bool:
    """
    比较两支队伍的排名优先级

    参数:
        team1: 队伍1名称
        team2: 队伍2名称
        team_stats: 队伍统计数据
        head_to_head: 队伍间交锋记录

    返回:
        如果队伍1排名应在队伍2之前，返回True，否则返回False
    """
    wins1, wins2 = team_stats[team1]['wins'], team_stats[team2]['wins']
    if wins1 != wins2:
        return wins1 > wins2

    diff1 = team_stats[team1]['maps_won'] - team_stats[team1]['maps_lost']
    diff2 = team_stats[team2]['maps_won'] - team_stats[team2]['maps_lost']
    if diff1 != diff2:
        return diff1 > diff2

    if team2 in head_to_head[team1]:
        h2h1 = head_to_head[team1][team2]
        h2h2 = head_to_head[team2][team1]
        if h2h1 != h2h2:
            return h2h1 > h2h2

    return team_stats[team1]['maps_won'] > team_stats[team2]['maps_won']

def get_qualified(
    group: List[str],
    pts: Dict[str, int],
    win_loss_dict: Dict[str, str],
    map_diff_dict: Dict[str, str],
    team_stats: Dict[str, Dict[str, int]],
    head_to_head: Dict[str, Dict[str, int]],
    num_qualify: int = 4
) -> List[str]:
    """
    从小组中选出积分前4的队伍晋级季后赛（支持复杂排名规则）

    参数:
        group: 小组队伍列表
        pts: 各队积分
        win_loss_dict: 各队胜负记录
        map_diff_dict: 各队地图差
        team_stats: 队伍统计数据
        head_to_head: 队伍间交锋记录
        num_qualify: 晋级队伍数量，默认4

    返回:
        按排名排序的晋级队伍列表
    """
    # 创建同分队伍组
    groups = defaultdict(list)
    for team in group:
        key = (
            team_stats[team]['wins'],
            team_stats[team]['maps_won'] - team_stats[team]['maps_lost'],
            team_stats[team]['maps_won']
        )
        groups[key].append(team)

    sorted_groups: List[str] = []
    for key, teams in sorted(groups.items(), key=lambda x: x[0], reverse=True):
        if len(teams) > 1:
            # 使用自定义比较函数进行排序
            teams.sort(key=lambda t: (-team_stats[t]['wins'], -(team_stats[t]['maps_won'] - team_stats[t]['maps_lost']), -head_to_head[t].get(opp, 0) if (opp := next((o for o in teams if o != t), None)) else 0, -team_stats[t]['maps_won']))
        sorted_groups.extend(teams)

    if debug:
        formatted_group = [f"{team}({win_loss_dict[team]}, {map_diff_dict[team]})" for team in sorted_groups]
        print(f"\n{group} 小组最终排名: {formatted_group}")
    return sorted_groups[:num_qualify]

def play_playoffs(
        qualified_teams_a: List[str],
        qualified_teams_b: List[str],
        initial_pts: Dict[str, int],
        regular_pts: Dict[str, int],
        use_real_data: bool = False,
        map_based: bool = False,
        map_pool: Optional[List[str]] = None,
        real_results: Optional[Dict[str, List[Any]]] = None
        ) -> Dict[str, Any]:
    """
    季后赛：M1-M12编号 + 从左到右布局 + 双败淘汰

    参数:
        qualified_teams_a: A组晋级队伍
        qualified_teams_b: B组晋级队伍
        initial_pts: 初始积分
        regular_pts: 常规赛积分
        use_real_data: 是否使用真实数据
        map_based: 是否基于地图模拟
        map_pool: 地图池
        real_results: 真实比赛结果

    返回:
        包含季后赛结果的字典，包括冠军、亚军、季军、殿军、最终积分等
    """
    if debug:
        print("\n=== 季后赛（M1-M12轮次，从左到右布局）===")

    alpha1, alpha2, alpha3, alpha4 = qualified_teams_a
    omega1, omega2, omega3, omega4 = qualified_teams_b

    if debug:
        print("\n分组排名:")
        for group_name, group in [("Alpha组", qualified_teams_a), ("Omega组", qualified_teams_b)]:
            print(group_name, [f"{i + 1}.{team}" for i, team in enumerate(group)])

    left_bracket = [alpha1, omega2, alpha3, omega4]
    right_bracket = [omega1, alpha2, omega3, alpha4]

    if debug:
        print("\n左半区队伍:", left_bracket)
        print("右半区队伍:", right_bracket)

    rounds: Dict[str, Dict[str, Any]] = {}

    def play_round(round_name: str, team1: str, team2: str, bo: int = 3) -> Tuple[str, str]:
        rounds[round_name] = {'teams': [team1, team2], 'winner': None, 'loser': None, 'score': None}
        is_bo5 = round_name in ['M11', 'M12']
        current_bo = 5 if is_bo5 else bo

        winner: Optional[str] = None
        score: Optional[Tuple[int, int]] = None
        if use_real_data and real_results and real_results['playoffs']:
            result = next((r for r in real_results['playoffs'] if
                          (r[0] == team1 and r[1] == team2) or
                          (r[0] == team2 and r[1] == team1)), None)
            if result:
                winner = result[2]
                if len(result) > 3 and result[3] is not None:
                    score = _parse_score_from_result(result[3])
                else:
                     # 如果没有提供比分，则生成合理的地图比分
                    if winner == team1:
                        score = (2, random.choice([0, 1])) if current_bo == 3 else (3, random.choice([0, 1, 2]))
                    else:
                        score = (random.choice([0, 1]), 2) if current_bo == 3 else (random.choice([0, 1, 2]), 3)

        if winner is None: # 没有真实数据或未找到匹配项
            if map_based:
                winner, score = simulate_match(team1, team2, bo=current_bo, team1_win_rate=0.5)
            else:
                winner = team1 if random.choice([True, False]) else team2
                if winner == team1:
                    score = (2, random.choice([0, 1])) if current_bo == 3 else (3, random.choice([0, 1, 2]))
                else:
                    score = (random.choice([0, 1]), 2) if current_bo == 3 else (random.choice([0, 1, 2]), 3)

        loser = team1 if winner == team2 else team2
        rounds[round_name]['winner'] = winner
        rounds[round_name]['loser'] = loser
        rounds[round_name]['score'] = score

        if debug:
            score_str = f"{score[0]}:{score[1]}" if score else ""
            print(f"{round_name}: {team1} vs {team2} -> {winner} 胜 {score_str}")
        return winner, loser

    m1_winner, m1_loser = play_round('M1', left_bracket[1], left_bracket[2])
    m2_winner, m2_loser = play_round('M2', right_bracket[1], right_bracket[2])
    m3_winner, m3_loser = play_round('M3', left_bracket[3], m1_loser)
    m4_winner, m4_loser = play_round('M4', right_bracket[3], m2_loser)
    m5_winner, m5_loser = play_round('M5', left_bracket[0], m1_winner)
    m6_winner, m6_loser = play_round('M6', right_bracket[0], m2_winner)
    m7_winner, m7_loser = play_round('M7', m3_winner, m6_loser)
    m8_winner, m8_loser = play_round('M8', m4_winner, m5_loser)
    m9_winner, m9_loser = play_round('M9', m5_winner, m6_winner)
    m10_winner, m10_loser = play_round('M10', m7_winner, m8_winner)
    m11_winner, m11_loser = play_round('M11', m9_loser, m10_winner, bo=5)
    champion, runner_up = play_round('M12', m9_winner, m11_winner, bo=5)

    third_place = m11_loser
    fourth_place = m10_loser

    if debug:
        print(f"\n最终排名:")
        print(f"1. {champion}（冠军）")
        print(f"2. {runner_up}（亚军）")
        print(f"3. {third_place}（季军 +4分）")
        print(f"4. {fourth_place}（殿军 +3分）")

    updated_pts = {team: initial_pts.get(team, 0) + regular_pts.get(team, 0) for team in set(initial_pts) | set(regular_pts)}
    updated_pts[third_place] += 4
    updated_pts[fourth_place] += 3

    non_champ_runnerup_pts = {team: score for team, score in updated_pts.items() if team not in [champion, runner_up]}
    sorted_non_champ_runnerup = sorted(non_champ_runnerup_pts.items(), key=lambda x: x[1], reverse=True)
    third_seed = sorted_non_champ_runnerup[0][0] if sorted_non_champ_runnerup else None
    fourth_seed = sorted_non_champ_runnerup[1][0] if len(sorted_non_champ_runnerup) > 1 else None

    if debug:
        print("\n冠军赛出征队伍：")
        print(f"一号种子：{champion}")
        print(f"二号种子：{runner_up}")
        print(f"三号种子：{third_seed}")
        print(f"四号种子：{fourth_seed}")

    return {
        'champion': champion,
        'runner_up': runner_up,
        'third_place': third_place,
        'fourth_place': fourth_place,
        'final_pts': updated_pts,
        'rounds': rounds,
        'third_seed': third_seed,
        'fourth_seed': fourth_seed,
        'champions_slots': [champion, runner_up, third_seed, fourth_seed]
    }

def create_playoffs_visualization(playoff_results: Dict[str, Any], region: str = 'cn') -> graphviz.Digraph:
    """
    创建季后赛双败淘汰赛制的可视化图表

    参数:
        playoff_results: play_playoffs函数返回的结果字典
        region: 赛区名称，用于文件名

    返回:
        Graphviz对象
    """
    rounds = playoff_results['rounds']
    dot = graphviz.Digraph(comment='Playoffs Bracket', format='png')
    dot.attr(rankdir='LR', size='15,12', splines='ortho')
    dot.attr('node',
             shape='box',
             style='rounded,filled',
             color='black',
             fontname='Arial',
             width='1.6',
             height='0.9',
             fixedsize='true')
    dot.attr('edge', arrowhead='vee')

    column_config = [
        ("Round 1", ['M1', 'M2', 'M3', 'M4']),
        ("Round 2", ['M5', 'M6', 'M7', 'M8']),
        ("Round 3", ['M9', 'M10']),
        ("Finals", ['M11', 'M12'])
    ]

    for round_name, match_data in rounds.items():
        teams = match_data['teams']
        winner = match_data['winner']
        score = match_data.get('score', (0, 0))
        if winner == teams[1]:
            score_str = f"{score[1]}:{score[0]}" if score else ""
        else:
            score_str = f"{score[0]}:{score[1]}" if score else ""
        label_text = f"{round_name}\n{teams[0]} vs {teams[1]}\nScore: {score_str}\nWinner: {winner}"
        color = "lightblue" if round_name in ['M1', 'M2', 'M5', 'M6', 'M9', 'M12'] else "lightcoral"
        dot.node(round_name, label=label_text, color=color)

    for label, nodes in column_config:
        subgraph_name = f"cluster_{label.lower().replace(' ', '_')}"
        with dot.subgraph(name=subgraph_name, graph_attr={
            'rank': 'same',
            'label': label,
            'fontname': 'Arial',
            'fontsize': '16',
            'fontweight': 'bold',
            'style': 'rounded',
            'color': 'gray25'
        }) as sub:
            for node in nodes:
                if node in rounds:
                    sub.node(node)

    winner_edges = [
        ('M1', 'M5'), ('M2', 'M6'), ('M3', 'M7'), ('M4', 'M8'),
        ('M5', 'M9'), ('M6', 'M9'), ('M7', 'M10'), ('M8', 'M10'),
        ('M9', 'M12'), ('M10', 'M11'), ('M11', 'M12'),
    ]
    for u, v in winner_edges:
        dot.edge(u, v, color="red", penwidth="2")

    dot.node('Champion',
             label=f"🏆 {playoff_results['champion']}",
             color='gold',
             fontsize='20',
             fontweight='bold',
             fontname='Arial',
             width='1.2',
             height='0.9')
    dot.edge('M12', 'Champion', style='invis')

    output_filename = f'playoffs_bracket_{region}'
    dot.render(output_filename, view=True, cleanup=True)
    print(f"季后赛对阵图已保存至 {output_filename}.png")
    return dot

def simulate_single_run(
    alpha: List[str],
    omega: List[str],
    initial_pts: Dict[str, int],
    use_real_data: bool = False,
    map_based: bool = False,
    map_pool: Optional[List[str]] = None,
    real_results: Optional[Dict[str, List[Any]]] = None
) -> Tuple[List[Optional[str]], List[str], Dict[str, Any]]:
    """
    单次模拟运行，支持地图级别模拟

    参数:
        alpha: Alpha组队伍
        omega: Omega组队伍
        initial_pts: 初始积分
        use_real_data: 是否使用真实数据
        map_based: 是否基于地图模拟
        map_pool: 地图池
        real_results: 真实比赛结果

    返回:
        元组包含:
        - 冠军赛参赛队伍
        - 季后赛晋级队伍
        - 完整的季后赛结果
    """
    alpha_pts, alpha_win_loss, alpha_map_diff, alpha_team_stats, alpha_head_to_head, _ = play_regular_season(
        alpha, use_real_data, map_based, map_pool, real_results
    )
    omega_pts, omega_win_loss, omega_map_diff, omega_team_stats, omega_head_to_head, _ = play_regular_season(
        omega, use_real_data, map_based, map_pool, real_results
    )

    team_stats = {**alpha_team_stats, **omega_team_stats}
    head_to_head = {**alpha_head_to_head, **omega_head_to_head}
    map_diff_dict = {**alpha_map_diff, **omega_map_diff}

    alpha_qualified = get_qualified(
        alpha, alpha_pts, alpha_win_loss, alpha_map_diff,
        alpha_team_stats, alpha_head_to_head
    )
    omega_qualified = get_qualified(
        omega, omega_pts, omega_win_loss, omega_map_diff,
        omega_team_stats, omega_head_to_head
    )

    regular_pts = {**alpha_pts, **omega_pts}
    playoff_results = play_playoffs(
        alpha_qualified, omega_qualified, initial_pts, regular_pts,
        use_real_data, map_based, map_pool, real_results
    )
    return playoff_results['champions_slots'], alpha_qualified + omega_qualified, playoff_results # 返回完整结果用于可视化

def simulate_regular_seasons(
    num_simulations: int,
    alpha: List[str],
    omega: List[str],
    use_real_data: bool,
    num_threads: int,
    map_based: bool,
    map_pool: Optional[List[str]],
    real_results: Optional[Dict[str, List[Any]]]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    模拟多次常规赛，计算各队晋级概率

    参数:
        num_simulations: 模拟次数
        alpha: Alpha组队伍
        omega: Omega组队伍
        use_real_data: 是否使用真实数据
        num_threads: 线程数
        map_based: 是否基于地图模拟
        map_pool: 地图池
        real_results: 真实比赛结果

    返回:
        元组包含Alpha组和Omega组各队晋级概率
    """
    alpha_qualify_count: Dict[str, int] = {team: 0 for team in alpha}
    omega_qualify_count: Dict[str, int] = {team: 0 for team in omega}

    def single_simulation() -> None:
        alpha_pts, alpha_win_loss, alpha_map_diff, alpha_team_stats, alpha_head_to_head, _ = play_regular_season(
            alpha, use_real_data, map_based, map_pool, real_results
        )
        omega_pts, omega_win_loss, omega_map_diff, omega_team_stats, omega_head_to_head, _ = play_regular_season(
            omega, use_real_data, map_based, map_pool, real_results
        )

        team_stats = {**alpha_team_stats, **omega_team_stats}
        head_to_head = {**alpha_head_to_head, **omega_head_to_head}
        map_diff_dict = {**alpha_map_diff, **omega_map_diff}

        alpha_qualified = get_qualified(
            alpha, alpha_pts, alpha_win_loss, alpha_map_diff,
            alpha_team_stats, alpha_head_to_head
        )
        omega_qualified = get_qualified(
            omega, omega_pts, omega_win_loss, omega_map_diff,
            omega_team_stats, omega_head_to_head
        )

        for team in alpha_qualified:
            alpha_qualify_count[team] += 1
        for team in omega_qualified:
            omega_qualify_count[team] += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm.tqdm(executor.map(lambda _: single_simulation(), range(num_simulations)), total=num_simulations, desc="模拟常规赛"))

    alpha_probabilities = {team: count / num_simulations for team, count in alpha_qualify_count.items()}
    omega_probabilities = {team: count / num_simulations for team, count in omega_qualify_count.items()}
    return alpha_probabilities, omega_probabilities

def simulate_all_games(
    num_simulations: int,
    alpha: List[str],
    omega: List[str],
    initial_pts: Dict[str, int],
    use_real_data: bool,
    num_threads: int,
    map_based: bool,
    map_pool: Optional[List[str]],
    real_results: Optional[Dict[str, List[Any]]]
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, int]]:
    """
    模拟所有比赛，使用预加载的分组和积分数据

    参数:
        num_simulations: 模拟次数
        alpha: Alpha组队伍
        omega: Omega组队伍
        initial_pts: 初始积分
        use_real_data: 是否使用真实数据
        num_threads: 线程数
        map_based: 是否基于地图模拟
        map_pool: 地图池
        real_results: 真实比赛结果

    返回:
        元组包含:
        - 各队晋级冠军赛的概率
        - 各队未晋级季后赛但进入冠军赛的次数
        - 各队进入冠军赛前两名的次数
    """
    all_teams = alpha + omega
    champions_slots_count: Dict[str, int] = {team: 0 for team in all_teams}
    no_playoffs_but_slot_count: Dict[str, int] = {team: 0 for team in all_teams}
    top2_in_slot_count: Dict[str, int] = {team: 0 for team in all_teams}

    def single_run_wrapper(_: int) -> Tuple[List[Optional[str]], List[str], Dict[str, Any]]:
        return simulate_single_run(alpha, omega, initial_pts, use_real_data, map_based, map_pool, real_results)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm.tqdm(
            executor.map(single_run_wrapper, range(num_simulations)),
            total=num_simulations,
            desc="模拟常规赛+季后赛"
        ))

    for champions_slots, playoff_teams, _ in results: # 忽略第三个返回值（playoff_results）
        for team in champions_slots:
            if team:  # 确保team不是None
                champions_slots_count[team] += 1
                if team in champions_slots[:2]:
                    top2_in_slot_count[team] += 1
        # 检查哪些队伍进入了冠军赛但未进入季后赛
        for team in set(champions_slots) - set(playoff_teams):
            if team:  # 确保team不是None
                no_playoffs_but_slot_count[team] += 1

    probabilities = {team: count / num_simulations for team, count in champions_slots_count.items()}
    return probabilities, no_playoffs_but_slot_count, top2_in_slot_count

def main(args: argparse.Namespace) -> None:
    """
    主函数，执行单次模拟

    参数:
        args: 命令行参数
    """
    global debug
    debug = args.debug # 使用args.debug而不是硬编码True
    yaml_folder = args.yaml_folder
    region = args.region
    random_seed = args.random_seed
    random.seed(random_seed)

    # 预加载配置
    map_pool = load_map_pool(yaml_folder)
    use_real_data = not args.no_real_data # args.no_real_data 为 True 表示不使用真实数据
    map_based = not args.no_map_based     # args.no_map_based 为 True 表示不启用地图模拟

    real_results = None
    if use_real_data:
        real_results = load_real_results(
            args.source,
            args.results_file,
            yaml_folder,
            region
        )

    initial_pts = load_initial_pts(yaml_folder, region)
    group_alpha, group_omega = load_group_teams(yaml_folder, region)

    if debug:
        print("模拟参数：")
        print(f"  模拟赛区: {region}")
        print(f"  启用地图级别模拟: {map_based}") # 直接打印 map_based 的值
        print(f"  使用随机种子: {random_seed}")
        print(f"  使用真实数据: {use_real_data}") # 直接打印 use_real_data 的值
        print(f"  打印详细结果: {debug}")

        print("初始积分：")
        for team, score in initial_pts.items():
            print(f"{team}: {score}")

        print("\n分组：")
        print(f"Alpha组：{group_alpha}")
        print(f"Omega组：{group_omega}")

    # 调用单次模拟函数
    champions_slots, playoff_teams, playoff_results = simulate_single_run(
        group_alpha, group_omega, initial_pts, use_real_data, map_based, map_pool, real_results
    )

    # 可视化
    create_playoffs_visualization(playoff_results, region=region)

    # 最终积分排名
    final_ranking = sorted(playoff_results['final_pts'].items(), key=lambda x: x[1], reverse=True)
    if debug:
        print("\n最终积分排名：")
        for i, (team, score) in enumerate(final_ranking, 1):
            print(f"{i}. {team}: {score}分")

def print_probabilities(
    title: str,
    probs: Union[Dict[str, float], Dict[str, int]],
    show_separator: bool = True,
    threshold: int = 5,
    reverse: bool = True
) -> None:
    """
    打印概率信息的函数

    参数:
        title: 打印的标题
        probs: 概率字典
        show_separator: 是否显示分割线，默认为True
        threshold: 分割线的位置，默认为5
        reverse: 是否降序排序，默认为True
    """
    print(f"\n{title}:")
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=reverse)
    for i, (team, prob) in enumerate(sorted_probs, start=1):
        if show_separator and i == threshold:
            print("------")
        if isinstance(prob, float):
            print(f"{team}: {prob * 100:.2f}%")
        else:
            print(f"{team}: {prob}")

def multi_sim(args: argparse.Namespace) -> Dict[str, Any]:
    """
    多次模拟函数，执行多次模拟并计算概率

    参数:
        args: 命令行参数

    返回:
        包含各类概率统计的字典
    """
    global debug
    debug = args.debug
    yaml_folder = args.yaml_folder
    region = args.region
    num_threads = min(args.num_threads, args.num_simulations)
    random_seed = args.random_seed
    random.seed(random_seed)

    # 预加载配置
    map_pool = load_map_pool(yaml_folder)
    use_real_data = not args.no_real_data # args.no_real_data 为 True 表示不使用真实数据
    map_based = not args.no_map_based     # args.no_map_based 为 True 表示不启用地图模拟

    real_results = None
    if use_real_data:
        real_results = load_real_results(
            args.source,
            args.results_file,
            yaml_folder,
            region
        )

    start_time = time.time()

    # 提前加载分组和初始积分
    alpha, omega = load_group_teams(yaml_folder, region)
    initial_pts = load_initial_pts(yaml_folder, region)

    if debug:
        print("模拟参数：")
        print(f"  模拟赛区: {region}")
        print(f"  模拟次数: {args.num_simulations}")
        print(f"  启用地图级别模拟: {map_based}") # 直接打印 map_based 的值
        print(f"  使用线程数: {num_threads}")
        print(f"  使用随机种子: {random_seed}")
        print(f"  使用真实数据: {use_real_data}") # 直接打印 use_real_data 的值
        print(f"  打印详细结果: {debug}")

    # 模拟常规赛，计算晋级季后赛概率
    alpha_probs, omega_probs = simulate_regular_seasons(
        args.num_simulations, alpha, omega, use_real_data,
        num_threads, map_based, map_pool, real_results
    )

    # 模拟常规赛+季后赛，计算晋级冠军赛概率
    champions_slots_probs, no_playoffs_but_slot_count, top2_in_slot_count = simulate_all_games(
        args.num_simulations, alpha, omega, initial_pts, use_real_data,
        num_threads, map_based, map_pool, real_results
    )

    end_time = time.time()
    simulation_time = end_time - start_time
    print(f"\n模拟总时间: {simulation_time:.2f} 秒")

    all_teams = sorted([k for k, _ in champions_slots_probs.items()])

    champions_slots_no_playoffs_probs = {
        team: count / args.num_simulations
        for team, count in no_playoffs_but_slot_count.items()
    }

    champions_slots_must_top2_probs = {
        team: top2_in_slot_count[team] / (champions_slots_probs[team] * args.num_simulations)
        if champions_slots_probs[team] > 0 else 0
        for team in all_teams
    }

    print_probabilities("Alpha组晋级季后赛概率", alpha_probs)
    print_probabilities("Omega组晋级季后赛概率", omega_probs)
    print_probabilities("晋级冠军赛概率", champions_slots_probs)
    print_probabilities("靠积分，不晋级季后赛进冠军赛概率", champions_slots_no_playoffs_probs)
    print_probabilities("不靠积分，只能以冠亚进占所有进冠军赛可能比例", champions_slots_must_top2_probs, reverse=False)

    summary_dict = {
        'alpha_probs': alpha_probs,
        'omega_probs': omega_probs,
        'champions_slots_probs': champions_slots_probs,
        'champions_slots_no_playoffs_probs': champions_slots_no_playoffs_probs,
        'champions_slots_must_top2_probs': champions_slots_must_top2_probs,
    }
    return summary_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VCT 2025 Stage 2 Simulation')
    parser.add_argument('--no_real_data', action='store_true', help='不使用真实比赛数据（默认使用）')
    parser.add_argument('--source', default='local', choices=['local', 'online'], help='数据加载源 (local/online)')
    parser.add_argument('--results_file', default='results.yaml', help='比赛结果文件路径')
    parser.add_argument('--yaml_folder', default='./yaml', help='YAML文件夹的位置')
    parser.add_argument('--region', type=str, default='cn', help='模拟的VCT赛区（目前支持cn/pacific)')
    parser.add_argument('--no_map_based', action='store_true', help='禁用地图级别模拟（默认启用）')
    parser.add_argument('--multi', action='store_true', default=False, help='是否进行多次模拟实验，默认关闭')
    parser.add_argument('--num_simulations', type=int, default=500, help='模拟实验的次数，默认500')
    parser.add_argument('--debug', action='store_true', help='是否打印内容数据')
    parser.add_argument('--num_threads', type=int, default=8, help='模拟使用的线程数')
    parser.add_argument('--random_seed', type=int, default=2, help='随机种子')
    args = parser.parse_args()

    if args.multi:
        summary_dict = multi_sim(args)
    else:
        main(args)