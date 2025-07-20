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
import random
from collections import defaultdict
from math import comb

import yaml
import requests
import graphviz
import tqdm

# 全局 debug 变量
debug = False

def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:  # 添加编码参数
        return yaml.safe_load(f)

def load_group_teams(yaml_folder, region='cn'):
    """返回预设的两个小组"""
    # 使用Path处理跨平台路径
    file_path = Path(yaml_folder) / region / 'groups.yaml'
    groups = load_yaml(file_path)
    if debug:
        print(f"加载分组文件: {file_path}")  # 显示完整路径
        print(groups)
    return groups['Alpha'], groups['Omega']

def load_initial_pts(yaml_folder, region='cn'):
    """加载初始积分"""
    # 使用Path处理跨平台路径
    file_path = Path(yaml_folder) / region / 'initial_pts.yaml'
    return load_yaml(file_path)

def load_real_results(source="local", results_file="results.yaml", yaml_folder="./yaml", region='cn'):
    """从本地文件或网络API加载真实比赛结果"""
    if source == "local":
        try:
            data = load_yaml(os.path.join(yaml_folder, region, results_file))
            # 检查是否有 'playoffs' 键，如果没有则添加空列表
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
            # 示例URL，实际使用时替换为真实API
            response = requests.get("https://api.example.com/vct_results")
            data = response.json()
            # 检查是否有 'playoffs' 键，如果没有则添加空列表
            if 'playoffs' not in data:
                data['playoffs'] = []
            return data
        except Exception as e:
            if debug:
                print(f"加载网络数据失败: {e}")
            return {'regular_season': [], 'playoffs': []}

# 添加新的地图级别模拟函数
def calculate_win_probability(bo: int, p: float) -> dict:
    """
    计算不同BO赛制下各比分的理论概率分布

    参数:
    bo: 比赛局数，3或5
    p: 队伍A每局获胜的概率

    返回:
    字典，包含各比分的概率分布
    """
    if bo not in {3, 5}:
        raise ValueError("bo参数必须是3或5")

    required_wins = (bo + 1) // 2
    results = {}

    for total_games in range(required_wins, bo + 1):
        losses = total_games - required_wins
        if losses >= required_wins:
            continue  # 不可能的比分

        # A队获胜概率
        comb_val = comb(total_games - 1, required_wins - 1)
        prob_a = comb_val * (p ** required_wins) * ((1 - p) ** losses)
        a_score = f"{required_wins}:{losses}"
        results[a_score] = prob_a

        # B队获胜概率
        prob_b = comb_val * ((1 - p) ** required_wins) * (p ** losses)
        b_score = f"{losses}:{required_wins}"
        results[b_score] = prob_b

    total = sum(results.values())
    return {k: v/total for k, v in results.items()}

def simulate_match(team1: str, team2: str, bo: int = 3, team1_win_rate: float = 0.5) -> tuple:
    """
    模拟一场BO3或BO5比赛，返回比分结果

    参数:
    team1: 队伍A的名称
    team2: 队伍B的名称
    bo: 比赛局数，3或5（默认3）
    team1_win_rate: 队伍A每局获胜的概率（默认0.5）

    返回:
    (胜者, 比分) 元组，比分格式为(胜场, 负场)
    """
    if bo not in {3, 5}:
        raise ValueError("bo参数必须是3或5")
    if not (0 <= team1_win_rate <= 1):
        raise ValueError("team1_win_rate必须在0到1之间")

    required_wins = (bo + 1) // 2
    theoretical_probs = calculate_win_probability(bo, team1_win_rate)

    rand = random.random()
    cumulative = 0
    for score, prob in theoretical_probs.items():
        cumulative += prob
        if rand < cumulative:
            wins1, wins2 = map(int, score.split(':'))
            # 确定胜者
            if wins1 > wins2:
                return team1, (wins1, wins2)
            else:
                return team2, (wins2, wins1)

    # 默认返回
    wins = required_wins
    losses = random.randint(0, required_wins-1)
    winner = team1 if random.random() < team1_win_rate else team2
    if winner == team1:
        return winner, (wins, losses)
    else:
        return winner, (losses, wins)

# 添加地图池加载函数
def load_map_pool(yaml_folder, region='cn'):
    """加载地图池配置，返回地图名称列表（兼容旧格式）"""
    file_path = Path(yaml_folder) / 'map_pool.yaml'  # 注意：移除了region子目录

    try:
        data = load_yaml(file_path)

        # 处理新格式：包含属性的字典列表
        if data and isinstance(data, list) and isinstance(data[0], dict):
            return [item["name"] for item in data]
        # 处理旧格式：纯字符串列表
        elif data and isinstance(data, list) and isinstance(data[0], str):
            return data
        else:
            raise ValueError("Invalid map pool format")

    except (FileNotFoundError, ValueError):
        # 默认地图池
        return [
            "Ascent",
            "Bind",
            "Corrode",
            "Haven",
            "Icebox",
            "Lotus",
            "Sunset"
        ]

# 修改常规赛函数以支持地图级别模拟
def play_regular_season(group, use_real_data=False, map_based=False, map_pool=None):
    """进行常规赛：每组内每支队伍与同组其他队伍各打一场比赛"""
    pts = {team: 0 for team in group}
    win_loss = {team: [0, 0] for team in group}  # 胜场-负场
    map_diff = {team: 0 for team in group}  # 地图净胜分
    head_to_head = {team: {} for team in group}  # 相互胜负关系
    match_records = []  # 存储所有比赛记录

    # 新数据结构用于详细记录
    team_stats = {
        team: {
            'wins': 0,
            'losses': 0,
            'maps_won': 0,
            'maps_lost': 0
        } for team in group
    }

    played_matches = set()

    if use_real_data and real_results['regular_season']:
        if debug:
            print("\n使用真实常规赛数据")
        for match in real_results['regular_season']:
            team1, team2, result = match
            if team1 in group and team2 in group and result is not None:
                played_matches.add(tuple(sorted([team1, team2])))

                # 处理真实数据格式
                if isinstance(result, list) and len(result) > 0 and isinstance(result[0], (list, tuple)):
                    # 地图比分格式：[[13,11], [8,13], [13,10]]
                    team1_maps = sum(1 for r in result if r[0] > r[1])
                    team2_maps = sum(1 for r in result if r[1] > r[0])
                else:
                    # 传统比分格式：[2, 0] 或 (2, 1)
                    team1_maps, team2_maps = result

                winner = team1 if team1_maps > team2_maps else team2
                pts[winner] += 1
                win_loss[winner][0] += 1
                win_loss[team2 if winner == team1 else team1][1] += 1

                # 更新详细统计数据
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

    # 模拟比赛逻辑
    if debug:
        print("\n模拟常规赛")
    for i, team1 in enumerate(group):
        for team2 in group[i + 1:]:
            match = tuple(sorted([team1, team2]))
            if match not in played_matches:
                if map_based:
                    # 地图级别模拟
                    winner, (team1_maps, team2_maps) = simulate_match(
                        team1, team2, bo=3, team1_win_rate=0.5
                    )
                else:
                    # 传统模拟
                    winner = team1 if random.choice([True, False]) else team2
                    # 生成合理的地图比分
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

                # 更新详细统计数据
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

    # 计算地图净胜分
    for team in group:
        map_diff[team] = team_stats[team]['maps_won'] - team_stats[team]['maps_lost']

    win_loss_dict = {team: f"{win_loss[team][0]}胜-{win_loss[team][1]}负" for team in group}
    map_diff_dict = {team: f"+{map_diff[team]}" if map_diff[team] > 0 else str(map_diff[team]) for team in group}

    return pts, win_loss_dict, map_diff_dict, team_stats, head_to_head, match_records

# 实现复杂的排名比较函数
def compare_teams(team1, team2, team_stats, head_to_head):
    """比较两支队伍的排名优先级"""
    # 1. 大场胜率
    if team_stats[team1]['wins'] != team_stats[team2]['wins']:
        return team_stats[team1]['wins'] > team_stats[team2]['wins']

    # 2. 地图净胜分
    diff1 = team_stats[team1]['maps_won'] - team_stats[team1]['maps_lost']
    diff2 = team_stats[team2]['maps_won'] - team_stats[team2]['maps_lost']
    if diff1 != diff2:
        return diff1 > diff2

    # 3. 相互胜负关系
    if team2 in head_to_head[team1]:
        return head_to_head[team1][team2] > head_to_head[team2][team1]

    # 4. 总赢图数
    return team_stats[team1]['maps_won'] > team_stats[team2]['maps_won']

# todo: 修改晋级函数以支持复杂排名规则，需要进一步检查修正
def get_qualified(group, pts, win_loss_dict, map_diff_dict, team_stats, head_to_head, num_qualify=4):
    """从小组中选出积分前4的队伍晋级季后赛（支持复杂排名规则）"""
    # 创建同分队伍组
    groups = defaultdict(list)
    for team in group:
        key = (
            team_stats[team]['wins'],
            team_stats[team]['maps_won'] - team_stats[team]['maps_lost'],
            team_stats[team]['maps_won']
        )
        groups[key].append(team)

    # 对每个同分组内按相互胜负排序
    sorted_groups = []
    for key, teams in sorted(groups.items(), key=lambda x: x[0], reverse=True):
        if len(teams) > 1:
            # 多队同分时，计算相互胜负净胜分
            h2h_stats = {}
            for team in teams:
                wins = 0
                for opp in teams:
                    if team != opp and opp in head_to_head[team]:
                        wins += head_to_head[team][opp]
                h2h_stats[team] = wins

            # 按相互胜场数排序
            teams.sort(key=lambda x: (-h2h_stats[x], -team_stats[x]['maps_won']))
        sorted_groups.extend(teams)

    if debug:
        formatted_group = [f"{team}({win_loss_dict[team]}, {map_diff_dict[team]})" for team in sorted_groups]
        print(f"\n{group} 小组最终排名: {formatted_group}")

    return sorted_groups[:num_qualify]

def play_playoffs(
        qualified_teams_a,
        qualified_teams_b,
        initial_pts,
        regular_pts,
        use_real_data=False,
        map_based=False,
        map_pool=None
        ):
    """季后赛：M1-M12编号 + 从左到右布局 + 双败淘汰可视化"""
    if debug:
        print("\n=== 季后赛（M1-M12轮次，从左到右布局）===")

    # 初始化Graphviz（从左到右布局，PNG格式）
    dot = graphviz.Digraph(comment='Playoffs Bracket', format='png')
    dot.attr(rankdir='LR', size='15,12', splines='ortho')  # 从左到右，正交连线
    dot.attr('node', shape='box', style='rounded,filled', color='black', fontname='Arial')
    dot.attr('edge', arrowhead='vee')

    # 分组排名解析
    alpha1, alpha2, alpha3, alpha4 = qualified_teams_a
    omega1, omega2, omega3, omega4 = qualified_teams_b

    if debug:
        print("\n分组排名:")
        for group_name, group in [("Alpha组", qualified_teams_a), ("Omega组", qualified_teams_b)]:
            print(group_name, [f"{i + 1}.{team}" for i, team in enumerate(group)])

    # 半区队伍定义
    left_bracket = [alpha1, omega2, alpha3, omega4]  # 左1(Alpha1), 左2(Omega2), 左3(Alpha3), 左4(Omega4)
    right_bracket = [omega1, alpha2, omega3, alpha4]  # 右1(Omega1), 右2(Alpha2), 右3(Omega3), 右4(Alpha4)
    if debug:
        print("\n左半区队伍:", left_bracket)
        print("右半区队伍:", right_bracket)

    # 存储各轮次结果（M1-M12）
    rounds = {}
    def play_round(round_name, team1, team2, bo=3):
        rounds[round_name] = {'teams': [team1, team2], 'winner': None, 'loser': None, 'score': None}

        # 确定BO类型
        is_bo5 = round_name in ['M11', 'M12']  # 败者组决赛和总决赛是BO5
        current_bo = 5 if is_bo5 else bo

        if use_real_data and real_results['playoffs']:
            result = next((r for r in real_results['playoffs'] if
                          (r[0] == team1 and r[1] == team2) or
                          (r[0] == team2 and r[1] == team1)), None)
            if result:
                winner = result[2]
                # 处理真实比分
                if len(result) > 3:
                    score_data = result[3]
                    # 判断是否为地图比分格式（列表中的元素也是列表或元组）
                    if isinstance(score_data, list) and len(score_data) > 0 and isinstance(score_data[0], (list, tuple)):
                        # 地图比分格式：[[13,11], [8,13], [13,10]]
                        team1_maps = sum(1 for r in score_data if r[0] > r[1])
                        team2_maps = sum(1 for r in score_data if r[1] > r[0])
                        score = (team1_maps, team2_maps)
                    else:
                        # 传统比分格式：(2, 0) 或 [2, 1]
                        score = tuple(score_data)  # 确保是元组
                else:
                    # 如果没有提供比分，则生成合理的地图比分
                    if winner == team1:
                        score = (2, random.choice([0, 1])) if current_bo == 3 else (3, random.choice([0, 1, 2]))
                    else:
                        score = (random.choice([0, 1]), 2) if current_bo == 3 else (random.choice([0, 1, 2]), 3)
                rounds[round_name]['score'] = score
            else:
                # 没有真实数据时随机模拟
                if map_based:
                    winner, score = simulate_match(team1, team2, bo=current_bo, team1_win_rate=0.5)
                    rounds[round_name]['score'] = score
                else:
                    winner = team1 if random.choice([True, False]) else team2
                    # 生成合理的地图比分
                    if winner == team1:
                        score = (2, random.choice([0, 1])) if current_bo == 3 else (3, random.choice([0, 1, 2]))
                    else:
                        score = (random.choice([0, 1]), 2) if current_bo == 3 else (random.choice([0, 1, 2]), 3)
                    rounds[round_name]['score'] = score
        else:
            # 没有真实数据时模拟
            if map_based:
                winner, score = simulate_match(team1, team2, bo=current_bo, team1_win_rate=0.5)
                rounds[round_name]['score'] = score
            else:
                winner = team1 if random.choice([True, False]) else team2
                # 生成合理的地图比分
                if winner == team1:
                    score = (2, random.choice([0, 1])) if current_bo == 3 else (3, random.choice([0, 1, 2]))
                else:
                    score = (random.choice([0, 1]), 2) if current_bo == 3 else (random.choice([0, 1, 2]), 3)
                rounds[round_name]['score'] = score

        loser = team1 if winner == team2 else team2
        rounds[round_name]['winner'] = winner
        rounds[round_name]['loser'] = loser

        if debug:
            score_str = f"{score[0]}:{score[1]}" if score else ""
            print(f"{round_name}: {team1} vs {team2} -> {winner} 胜 {score_str}")

        return winner, loser

    # 各轮比赛
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
    m11_winner, m11_loser = play_round('M11', m9_loser, m10_winner, bo=5)  # BO5
    champion, runner_up = play_round('M12', m9_winner, m11_winner, bo=5)  # BO5

    # 排名逻辑
    third_place = m11_loser  # M11败者
    fourth_place = m10_loser  # M10败者
    if debug:
        print(f"\n最终排名:")
        print(f"1. {champion}（冠军）")
        print(f"2. {runner_up}（亚军）")
        print(f"3. {third_place}（季军 +4分）")
        print(f"4. {fourth_place}（殿军 +3分）")

    # Graphviz 布局与连线
    column_config = [
        ("Round 1", ['M2', 'M1', 'M4', 'M3']),  # 第一轮
        ("Round 2", ['M6', 'M5', 'M7', 'M8']),  # 第二轮
        ("Round 3", ['M9', 'M10']),  # 第三轮
        ("Finals", ['M11', 'M12'])  # 总决赛
    ]

    for label, nodes in column_config:
        subgraph_name = f"cluster_{label.lower().replace(' ', '_')}"
        with dot.subgraph(name=subgraph_name, graph_attr={'rank': 'same', 'label': label}) as sub:
            for node in nodes:
                # 节点标签：轮次 + 队伍1 vs 队伍2 + 胜者
                teams = rounds[node]['teams']
                winner = rounds[node]['winner']
                label_text = f"{node}\n{teams[0]} vs {teams[1]}\nW: {winner}" if teams else node
                # 颜色区分：胜者组（M1-M2, M5-M6, M9, M12）浅蓝色；败者组（M3-M4, M7-M8, M10-M11）浅红色
                color = "lightblue" if node in ['M1', 'M2', 'M5', 'M6', 'M9', 'M12'] else "lightcoral"
                sub.node(node, label=label_text, color=color)

    # 胜者线（红色粗实线）：连接晋级关系
    winner_edges = [
        ('M1', 'M5'),  # M1胜者 → M5
        ('M2', 'M6'),  # M2胜者 → M6
        ('M3', 'M7'),  # M3胜者 → M7
        ('M4', 'M8'),  # M4胜者 → M8
        ('M5', 'M9'),  # M5胜者 → M9
        ('M6', 'M9'),  # M6胜者 → M9
        ('M7', 'M10'),  # M7胜者 → M10
        ('M8', 'M10'),  # M8胜者 → M10
        ('M9', 'M12'),  # M9胜者 → M12
        ('M10', 'M11'),  # M10胜者 → M11
        ('M11', 'M12'),  # M11胜者 → M12
    ]
    for u, v in winner_edges:
        dot.edge(u, v, color="red", penwidth="2")  # 胜者线：红色粗实线

    # 更新积分
    updated_pts = {team: initial_pts.get(team, 0) + regular_pts.get(team, 0) for team in set(initial_pts) | set(regular_pts)}
    updated_pts[third_place] += 4
    updated_pts[fourth_place] += 3

    # 计算冠军赛出征队伍
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
        'dot_graph': dot,  # 返回Graphviz图对象
        'third_seed': third_seed,
        'fourth_seed': fourth_seed,
        'champions_slots': [champion, runner_up, third_seed, fourth_seed]
    }

def simulate_single_run(alpha, omega, initial_pts, use_real_data, map_based, map_pool):
    """单次模拟运行，支持地图级别模拟"""
    # 常规赛
    alpha_pts, alpha_win_loss, alpha_map_diff, alpha_team_stats, alpha_head_to_head, _ = play_regular_season(
        alpha, use_real_data, map_based, map_pool
    )
    omega_pts, omega_win_loss, omega_map_diff, omega_team_stats, omega_head_to_head, _ = play_regular_season(
        omega, use_real_data, map_based, map_pool
    )

    # 合并数据
    team_stats = {**alpha_team_stats, **omega_team_stats}
    head_to_head = {**alpha_head_to_head, **omega_head_to_head}
    map_diff_dict = {**alpha_map_diff, **omega_map_diff}

    # 晋级队伍
    alpha_qualified = get_qualified(
        alpha, alpha_pts, alpha_win_loss, alpha_map_diff,
        alpha_team_stats, alpha_head_to_head
    )
    omega_qualified = get_qualified(
        omega, omega_pts, omega_win_loss, omega_map_diff,
        omega_team_stats, omega_head_to_head
    )

    # 季后赛（含可视化）
    regular_pts = {**alpha_pts, **omega_pts}
    playoff_results = play_playoffs(
        alpha_qualified, omega_qualified, initial_pts, regular_pts,
        use_real_data, map_based, map_pool
    )

    return playoff_results['champions_slots'], alpha_qualified + omega_qualified

def simulate_regular_seasons(num_simulations, alpha, omega, use_real_data, num_threads, map_based, map_pool):
    alpha_qualify_count = {team: 0 for team in alpha}
    omega_qualify_count = {team: 0 for team in omega}

    def single_simulation():
        alpha_pts, alpha_win_loss, alpha_map_diff, alpha_team_stats, alpha_head_to_head, _ = play_regular_season(
            alpha, use_real_data, map_based, map_pool
        )
        omega_pts, omega_win_loss, omega_map_diff, omega_team_stats, omega_head_to_head, _ = play_regular_season(
            omega, use_real_data, map_based, map_pool
        )

        # 合并数据
        team_stats = {**alpha_team_stats, **omega_team_stats}
        head_to_head = {**alpha_head_to_head, **omega_head_to_head}
        map_diff_dict = {**alpha_map_diff, **omega_map_diff}

        # 晋级队伍 - 修改调用参数
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

def simulate_all_games(num_simulations, alpha, omega, initial_pts, use_real_data, num_threads, map_based, map_pool):
    """模拟所有比赛，使用预加载的分组和积分数据"""
    all_teams = alpha + omega
    champions_slots_count = {team: 0 for team in all_teams}
    no_playoffs_but_slot_count = {team: 0 for team in all_teams}
    top2_in_slot_count = {team: 0 for team in all_teams}

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm.tqdm(
            executor.map(lambda _: simulate_single_run(alpha, omega, initial_pts, use_real_data, map_based, map_pool),
            range(num_simulations)),
            total=num_simulations,
            desc="模拟常规赛+季后赛"
        ))

    for champions_slots, playoff_teams in results:
        # 统计各队伍获得冠军赛席位的总次数
        for team in champions_slots:
            champions_slots_count[team] += 1

            # 统计以冠亚身份(前两位)获得席位
            if team in champions_slots[:2]:
                top2_in_slot_count[team] += 1

        # 统计未晋级季后赛但获得席位(三/四号种子)
        for team in set(champions_slots) - set(playoff_teams):
            no_playoffs_but_slot_count[team] += 1

    probabilities = {team: count / num_simulations for team, count in champions_slots_count.items()}
    return probabilities, no_playoffs_but_slot_count, top2_in_slot_count

def main(args):
    global debug
    debug = True
    yaml_folder = args.yaml_folder
    region = args.region
    random_seed = args.random_seed
    random.seed(random_seed)

    # 加载地图池 (只保留一处加载)
    map_pool = load_map_pool(args.yaml_folder)

    use_real_data = args.use_real_data
    if use_real_data:
        global real_results
        real_results = load_real_results(
            args.source,
            args.results_file,
            yaml_folder,
            region
        )

    # 打印模拟参数
    print("模拟参数：")
    print(f"  模拟赛区: {region}")
    print(f"  使用随机种子: {random_seed}")
    print(f"  使用真实数据: {use_real_data}")
    print(f"  打印详细结果: {debug}")

    # 初始积分
    initial_pts = load_initial_pts(args.yaml_folder, args.region)
    if debug:
        print("初始积分：")
        for team, score in initial_pts.items():
            print(f"{team}: {score}")

    # 分组
    group_alpha, group_omega = load_group_teams(args.yaml_folder, args.region)
    if debug:
        print("\n分组：")
        print(f"Alpha组：{group_alpha}")
        print(f"Omega组：{group_omega}")

    # 常规赛
    alpha_pts, alpha_win_loss, alpha_map_diff, alpha_team_stats, alpha_head_to_head, alpha_matches = play_regular_season(
        group_alpha, args.use_real_data, args.map_based, map_pool
    )
    omega_pts, omega_win_loss, omega_map_diff, omega_team_stats, omega_head_to_head, omega_matches = play_regular_season(
        group_omega, args.use_real_data, args.map_based, map_pool
    )

    # 合并统计数据
    team_stats = {**alpha_team_stats, **omega_team_stats}
    head_to_head = {**alpha_head_to_head, **omega_head_to_head}
    map_diff_dict = {**alpha_map_diff, **omega_map_diff}

    # 晋级队伍
    qualify_a = get_qualified(
        group_alpha, alpha_pts, alpha_win_loss, alpha_map_diff,
        alpha_team_stats, alpha_head_to_head
    )
    qualify_b = get_qualified(
        group_omega, omega_pts, omega_win_loss, omega_map_diff,
        omega_team_stats, omega_head_to_head
    )

    # 常规赛积分分组显示
    if debug:
        print("\n常规赛结束后积分：")
        print("Alpha组：")
        for team in group_alpha:
            print(f"{team}: {alpha_pts[team]}")
        print("\nOmega组：")
        for team in group_omega:
            print(f"{team}: {omega_pts[team]}")

        print("\n晋级季后赛队伍：")
        print("Alpha组:", qualify_a)
        print("Omega组:", qualify_b)

    # 季后赛（含可视化）
    regular_pts = {**alpha_pts, **omega_pts}
    playoff_results = play_playoffs(
        qualify_a, qualify_b, initial_pts, regular_pts,
        args.use_real_data, args.map_based, map_pool
    )

    # 渲染PNG并打开
    playoff_results['dot_graph'].render(
        f'playoffs_bracket_{region}',
        format='png',
        view=True,
        cleanup=True,
        )

    # 最终积分排名
    final_ranking = sorted(playoff_results['final_pts'].items(), key=lambda x: x[1], reverse=True)
    if debug:
        print("\n最终积分排名：")
        for i, (team, score) in enumerate(final_ranking, 1):
            print(f"{i}. {team}: {score}分")

def print_probabilities(title, probs, show_separator=True, threshold=5, reverse=True):
    """
    打印概率信息的函数
    :param title: 打印的标题
    :param probs: 概率字典
    :param show_separator: 是否显示分割线，默认为True
    :param threshold: 分割线的位置，默认为5
    """
    print(f"\n{title}:")
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=reverse)
    for i, (team, prob) in enumerate(sorted_probs, start=1):
        if show_separator and i == threshold:
            print("------")
        print(f"{team}: {prob * 100:.2f}%")

def multi_sim(args):
    global debug
    debug = args.debug
    yaml_folder = args.yaml_folder
    region = args.region
    num_threads = min(args.num_threads, args.num_simulations)
    random_seed = args.random_seed
    random.seed(random_seed)

    # 加载地图池
    map_pool = load_map_pool(args.yaml_folder)

    use_real_data = args.use_real_data
    if use_real_data:
        global real_results
        real_results = load_real_results(
            args.source,
            args.results_file,
            yaml_folder,
            region
        )

    # 打印模拟参数
    print("模拟参数：")
    print(f"  模拟赛区: {region}")
    print(f"  模拟次数: {args.num_simulations}")
    print(f"  使用线程数: {num_threads}")
    print(f"  使用随机种子: {random_seed}")
    print(f"  使用真实数据: {use_real_data}")
    print(f"  打印详细结果: {debug}")

    start_time = time.time()

    # 提前加载分组和初始积分
    alpha, omega = load_group_teams(yaml_folder, region)
    initial_pts = load_initial_pts(yaml_folder, region)

    # 模拟常规赛，计算晋级季后赛概率
    alpha_probs, omega_probs = simulate_regular_seasons(
        args.num_simulations, alpha, omega, args.use_real_data,
        num_threads, args.map_based, map_pool
    )

    # 模拟常规赛+季后赛，计算晋级冠军赛概率
    champions_slots_probs, no_playoffs_but_slot_count, top2_in_slot_count = simulate_all_games(
        args.num_simulations, alpha, omega, initial_pts, args.use_real_data,
        num_threads, args.map_based, map_pool
    )

    end_time = time.time()
    simulation_time = end_time - start_time
    print(f"\n模拟总时间: {simulation_time:.2f} 秒")

    all_teams = sorted([k for k, _ in champions_slots_probs.items()])

    # 计算不晋级季后赛但可以晋级冠军赛概率
    champions_slots_no_playoffs_probs = {
        team: count / args.num_simulations
        for team, count in no_playoffs_but_slot_count.items()
    }

    # 计算要晋级冠军赛必须前二概率
    champions_slots_must_top2_probs = {
        team: top2_in_slot_count[team] / (champions_slots_probs[team] * args.num_simulations)
        if champions_slots_probs[team] > 0 else 0
        for team in all_teams
    }

    # 打印各种概率
    print_probabilities("Alpha组晋级季后赛概率", alpha_probs)
    print_probabilities("Omega组晋级季后赛概率", omega_probs)
    print_probabilities("晋级冠军赛概率", champions_slots_probs)
    print_probabilities("靠积分，不晋级季后赛进冠军赛概率", champions_slots_no_playoffs_probs)
    print_probabilities("不靠积分，只能以冠亚进占所有进冠军赛可能比例", champions_slots_must_top2_probs, reverse=False)

    # 构建总结字典
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
    parser.add_argument('--use_real_data', action='store_false', help='是否使用真实比赛数据')
    parser.add_argument('--source', default='local', choices=['local', 'online'], help='数据加载源 (local/online)')
    parser.add_argument('--results_file', default='results.yaml', help='比赛结果文件路径')
    parser.add_argument('--yaml_folder', default='./yaml', help='YAML文件夹的位置')
    parser.add_argument('--region', type=str, default='cn', help='模拟的VCT赛区（目前支持cn/pacific)')
    parser.add_argument('--map_based', action='store_true', help='启用地图级别模拟')
    parser.add_argument('--multi', action='store_true', default=False, help='是否进行多次模拟实验，默认关闭')
    parser.add_argument('--num_simulations', type=int, default=500, help='模拟实验的次数，默认500')
    parser.add_argument('--debug', action='store_true', help='是否打印内容数据')
    parser.add_argument('--num_threads', type=int, default=8, help='模拟使用的线程数')
    parser.add_argument('--random_seed', type=int, default=77777, help='随机种子')
    args = parser.parse_args()

    if args.multi:
        summary_dict = multi_sim(args)
    else:
        main(args)