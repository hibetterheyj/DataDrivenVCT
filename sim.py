import random
import json
import requests
import graphviz
import yaml
import argparse
import os

random.seed(77)


def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def group_teams(yaml_folder):
    """返回预设的两个小组"""
    groups = load_yaml(os.path.join(yaml_folder, 'groups.yaml'))
    return groups['Alpha'], groups['Omega']


def load_initial_scores(yaml_folder):
    """加载初始积分"""
    return load_yaml(os.path.join(yaml_folder, 'initial_scores.yaml'))


def load_real_results(source="local", results_file="results.yaml", yaml_folder="./yaml"):
    """从本地文件或网络API加载真实比赛结果"""
    if source == "local":
        try:
            data = load_yaml(os.path.join(yaml_folder, results_file))
            # 检查是否有 'playoffs' 键，如果没有则添加空列表
            if 'playoffs' not in data:
                data['playoffs'] = []
            return data
        except FileNotFoundError:
            print("本地数据文件未找到，将使用模拟数据")
            return {'regular_season': [], 'playoffs': []}
        except Exception as e:
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
            print(f"加载网络数据失败: {e}")
            return {'regular_season': [], 'playoffs': []}


def play_regular_season(group, scores, use_real_data=False):
    """进行常规赛：每组内每支队伍与同组其他队伍各打一场比赛"""
    played_matches = set()  # 用于记录已经进行的比赛

    if use_real_data and real_results['regular_season']:
        print("\n使用真实常规赛数据")
        for team1, team2, result in real_results['regular_season']:
            if team1 in group and team2 in group and result is not None:
                # 标记这场比赛已经进行
                played_matches.add(tuple(sorted([team1, team2])))
                winner = team1 if result[0] > result[1] else team2
                print(f"{team1} vs {team2} -> 比分: {result} 胜者: {winner}")
                scores[winner] += 1

    # 模拟比赛逻辑
    print("\n模拟常规赛")
    for i, team1 in enumerate(group):
        for team2 in group[i + 1:]:
            match = tuple(sorted([team1, team2]))
            if match not in played_matches:
                if random.choice([True, False]):
                    scores[team1] += 1
                    print(f"{team1} vs {team2} -> 胜者: {team1}")
                else:
                    scores[team2] += 1
                    print(f"{team1} vs {team2} -> 胜者: {team2}")
                played_matches.add(match)
    return scores


def get_qualified(group, scores, num_qualify=4):
    """从小组中选出积分前4的队伍晋级季后赛"""
    sorted_group = sorted(group, key=lambda x: scores[x], reverse=True)
    print(f"\n{group} 小组排名: {sorted_group}")
    return sorted_group[:num_qualify]


def play_playoffs(qualified_teams_a, qualified_teams_b, scores, use_real_data=False):
    """季后赛：M1-M12编号 + 从左到右布局 + 双败淘汰可视化"""
    print("\n=== 季后赛（M1-M12轮次，从左到右布局）===")

    # 初始化Graphviz（从左到右布局，PNG格式）
    dot = graphviz.Digraph(comment='Playoffs Bracket', format='png')
    dot.attr(rankdir='LR', size='15,12', splines='ortho')  # 从左到右，正交连线
    dot.attr('node', shape='box', style='rounded,filled', color='black', fontname='Arial')
    dot.attr('edge', arrowhead='vee')

    # 分组排名解析
    alpha_rank = qualified_teams_a  # [1st, 2nd, 3rd, 4th] → alpha1, alpha2, alpha3, alpha4
    omega_rank = qualified_teams_b  # [1st, 2nd, 3rd, 4th] → omega1, omega2, omega3, omega4
    alpha1, alpha2, alpha3, alpha4 = alpha_rank
    omega1, omega2, omega3, omega4 = omega_rank

    print("\n分组排名:")
    print("Alpha组:", [f"{i + 1}.{team}" for i, team in enumerate(alpha_rank)])
    print("Omega组:", [f"{i + 1}.{team}" for i, team in enumerate(omega_rank)])

    # 半区队伍定义
    left_bracket = [alpha1, omega2, alpha3, omega4]  # 左1(Alpha1), 左2(Omega2), 左3(Alpha3), 左4(Omega4)
    right_bracket = [omega1, alpha2, omega3, alpha4]  # 右1(Omega1), 右2(Alpha2), 右3(Omega3), 右4(Alpha4)
    print("\n左半区队伍:", left_bracket)
    print("右半区队伍:", right_bracket)

    # 存储各轮次结果（M1-M12）
    rounds = {
        'M1': {'teams': [], 'winner': None, 'loser': None},  # 左半区胜者组第一轮（左2 vs 左3）
        'M2': {'teams': [], 'winner': None, 'loser': None},  # 右半区胜者组第一轮（右2 vs 右3）
        'M3': {'teams': [], 'winner': None, 'loser': None},  # 左半区败者组第一轮（左4 vs M1败者）
        'M4': {'teams': [], 'winner': None, 'loser': None},  # 右半区败者组第一轮（右4 vs M2败者）
        'M5': {'teams': [], 'winner': None, 'loser': None},  # 左半区胜者组第二轮（左1 vs M1胜者）
        'M6': {'teams': [], 'winner': None, 'loser': None},  # 右半区胜者组第二轮（右1 vs M2胜者）
        'M7': {'teams': [], 'winner': None, 'loser': None},  # 左半区败者组第二轮（M3胜者 vs M6败者）
        'M8': {'teams': [], 'winner': None, 'loser': None},  # 右半区败者组第二轮（M4胜者 vs M5败者）
        'M9': {'teams': [], 'winner': None, 'loser': None},  # 胜者组决赛（M5胜者 vs M6胜者）
        'M10': {'teams': [], 'winner': None, 'loser': None},  # 败者组半决赛（M7胜者 vs M8胜者）
        'M11': {'teams': [], 'winner': None, 'loser': None},  # 败者组决赛（M9败者 vs M10胜者）
        'M12': {'teams': [], 'winner': None, 'loser': None},  # 总决赛（M9胜者 vs M11胜者）
    }

    # -------------------------- M1: 左半区胜者组第一轮（左2 vs 左3） --------------------------
    print("\n=== M1: 左半区胜者组第一轮（Omega2 vs Alpha3）===")
    m1_team1, m1_team2 = left_bracket[1], left_bracket[2]  # Omega2, Alpha3
    rounds['M1']['teams'] = [m1_team1, m1_team2]
    if use_real_data and real_results['playoffs']:
        result = next((r for r in real_results['playoffs'] if
                      (r[0] == m1_team1 and r[1] == m1_team2) or
                      (r[0] == m1_team2 and r[1] == m1_team1)), None)
        m1_winner = result[2] if result else (m1_team1 if random.choice([True, False]) else m1_team2)
    else:
        m1_winner = m1_team1 if random.choice([True, False]) else m1_team2
    m1_loser = m1_team1 if m1_winner == m1_team2 else m1_team2
    rounds['M1']['winner'] = m1_winner
    rounds['M1']['loser'] = m1_loser
    print(f"{m1_team1} vs {m1_team2} -> 胜者: {m1_winner}, 败者: {m1_loser}")

    # -------------------------- M2: 右半区胜者组第一轮（Alpha2 vs Omega3） --------------------------
    print("\n=== M2: 右半区胜者组第一轮（Alpha2 vs Omega3）===")
    m2_team1, m2_team2 = right_bracket[1], right_bracket[2]  # Alpha2, Omega3
    rounds['M2']['teams'] = [m2_team1, m2_team2]
    if use_real_data and real_results['playoffs']:
        result = next((r for r in real_results['playoffs'] if
                      (r[0] == m2_team1 and r[1] == m2_team2) or
                      (r[0] == m2_team2 and r[1] == m2_team1)), None)
        m2_winner = result[2] if result else (m2_team1 if random.choice([True, False]) else m2_team2)
    else:
        m2_winner = m2_team1 if random.choice([True, False]) else m2_team2
    m2_loser = m2_team1 if m2_winner == m2_team2 else m2_team2
    rounds['M2']['winner'] = m2_winner
    rounds['M2']['loser'] = m2_loser
    print(f"{m2_team1} vs {m2_team2} -> 胜者: {m2_winner}, 败者: {m2_loser}")

    # -------------------------- M3: 左半区败者组第一轮（Omega4 vs M1败者） --------------------------
    print("\n=== M3: 左半区败者组第一轮（Omega4 vs M1败者）===")
    m3_team1, m3_team2 = left_bracket[3], m1_loser  # Omega4, M1败者
    rounds['M3']['teams'] = [m3_team1, m3_team2]
    if use_real_data and real_results['playoffs']:
        result = next((r for r in real_results['playoffs'] if
                      (r[0] == m3_team1 and r[1] == m3_team2) or
                      (r[0] == m3_team2 and r[1] == m3_team1)), None)
        m3_winner = result[2] if result else (m3_team1 if random.choice([True, False]) else m3_team2)
    else:
        m3_winner = m3_team1 if random.choice([True, False]) else m3_team2
    m3_loser = m3_team1 if m3_winner == m3_team2 else m3_team2
    rounds['M3']['winner'] = m3_winner
    rounds['M3']['loser'] = m3_loser
    print(f"{m3_team1} vs {m3_team2} -> 胜者: {m3_winner}, 败者: {m3_loser}")

    # -------------------------- M4: 右半区败者组第一轮（Alpha4 vs M2败者） --------------------------
    print("\n=== M4: 右半区败者组第一轮（Alpha4 vs M2败者）===")
    m4_team1, m4_team2 = right_bracket[3], m2_loser  # Alpha4, M2败者
    rounds['M4']['teams'] = [m4_team1, m4_team2]
    if use_real_data and real_results['playoffs']:
        result = next((r for r in real_results['playoffs'] if
                      (r[0] == m4_team1 and r[1] == m4_team2) or
                      (r[0] == m4_team2 and r[1] == m4_team1)), None)
        m4_winner = result[2] if result else (m4_team1 if random.choice([True, False]) else m4_team2)
    else:
        m4_winner = m4_team1 if random.choice([True, False]) else m4_team2
    m4_loser = m4_team1 if m4_winner == m4_team2 else m4_team2
    rounds['M4']['winner'] = m4_winner
    rounds['M4']['loser'] = m4_loser
    print(f"{m4_team1} vs {m4_team2} -> 胜者: {m4_winner}, 败者: {m4_loser}")

    # -------------------------- M5: 左半区胜者组第二轮（Alpha1 vs M1胜者） --------------------------
    print("\n=== M5: 左半区胜者组第二轮（Alpha1 vs M1胜者）===")
    m5_team1, m5_team2 = left_bracket[0], m1_winner  # Alpha1, M1胜者
    rounds['M5']['teams'] = [m5_team1, m5_team2]
    if use_real_data and real_results['playoffs']:
        result = next((r for r in real_results['playoffs'] if
                      (r[0] == m5_team1 and r[1] == m5_team2) or
                      (r[0] == m5_team2 and r[1] == m5_team1)), None)
        m5_winner = result[2] if result else (m5_team1 if random.choice([True, False]) else m5_team2)
    else:
        m5_winner = m5_team1 if random.choice([True, False]) else m5_team2
    m5_loser = m5_team1 if m5_winner == m5_team2 else m5_team2
    rounds['M5']['winner'] = m5_winner
    rounds['M5']['loser'] = m5_loser
    print(f"{m5_team1} vs {m5_team2} -> 胜者: {m5_winner}, 败者: {m5_loser}")

    # -------------------------- M6: 右半区胜者组第二轮（Omega1 vs M2胜者） --------------------------
    print("\n=== M6: 右半区胜者组第二轮（Omega1 vs M2胜者）===")
    m6_team1, m6_team2 = right_bracket[0], m2_winner  # Omega1, M2胜者
    rounds['M6']['teams'] = [m6_team1, m6_team2]
    if use_real_data and real_results['playoffs']:
        result = next((r for r in real_results['playoffs'] if
                      (r[0] == m6_team1 and r[1] == m6_team2) or
                      (r[0] == m6_team2 and r[1] == m6_team1)), None)
        m6_winner = result[2] if result else (m6_team1 if random.choice([True, False]) else m6_team2)
    else:
        m6_winner = m6_team1 if random.choice([True, False]) else m6_team2
    m6_loser = m6_team1 if m6_winner == m6_team2 else m6_team2
    rounds['M6']['winner'] = m6_winner
    rounds['M6']['loser'] = m6_loser
    print(f"{m6_team1} vs {m6_team2} -> 胜者: {m6_winner}, 败者: {m6_loser}")

    # -------------------------- M7: 左半区败者组第二轮（M3胜者 vs M6败者） --------------------------
    print("\n=== M7: 左半区败者组第二轮（M3胜者 vs M6败者）===")
    m7_team1, m7_team2 = m3_winner, m6_loser  # M3胜者, M6败者
    rounds['M7']['teams'] = [m7_team1, m7_team2]
    if use_real_data and real_results['playoffs']:
        result = next((r for r in real_results['playoffs'] if
                      (r[0] == m7_team1 and r[1] == m7_team2) or
                      (r[0] == m7_team2 and r[1] == m7_team1)), None)
        m7_winner = result[2] if result else (m7_team1 if random.choice([True, False]) else m7_team2)
    else:
        m7_winner = m7_team1 if random.choice([True, False]) else m7_team2
    m7_loser = m7_team1 if m7_winner == m7_team2 else m7_team2
    rounds['M7']['winner'] = m7_winner
    rounds['M7']['loser'] = m7_loser
    print(f"{m7_team1} vs {m7_team2} -> 胜者: {m7_winner}, 败者: {m7_loser}")

    # -------------------------- M8: 右半区败者组第二轮（M4胜者 vs M5败者） --------------------------
    print("\n=== M8: 右半区败者组第二轮（M4胜者 vs M5败者）===")
    m8_team1, m8_team2 = m4_winner, m5_loser  # M4胜者, M5败者
    rounds['M8']['teams'] = [m8_team1, m8_team2]
    if use_real_data and real_results['playoffs']:
        result = next((r for r in real_results['playoffs'] if
                      (r[0] == m8_team1 and r[1] == m8_team2) or
                      (r[0] == m8_team2 and r[1] == m8_team1)), None)
        m8_winner = result[2] if result else (m8_team1 if random.choice([True, False]) else m8_team2)
    else:
        m8_winner = m8_team1 if random.choice([True, False]) else m8_team2
    m8_loser = m8_team1 if m8_winner == m8_team2 else m8_team2
    rounds['M8']['winner'] = m8_winner
    rounds['M8']['loser'] = m8_loser
    print(f"{m8_team1} vs {m8_team2} -> 胜者: {m8_winner}, 败者: {m8_loser}")

    # -------------------------- M9: 胜者组决赛（M5胜者 vs M6胜者） --------------------------
    print("\n=== M9: 胜者组决赛（M5胜者 vs M6胜者）===")
    m9_team1, m9_team2 = m5_winner, m6_winner  # M5胜者, M6胜者
    rounds['M9']['teams'] = [m9_team1, m9_team2]
    if use_real_data and real_results['playoffs']:
        result = next((r for r in real_results['playoffs'] if
                      (r[0] == m9_team1 and r[1] == m9_team2) or
                      (r[0] == m9_team2 and r[1] == m9_team1)), None)
        m9_winner = result[2] if result else (m9_team1 if random.choice([True, False]) else m9_team2)
    else:
        m9_winner = m9_team1 if random.choice([True, False]) else m9_team2
    m9_loser = m9_team1 if m9_winner == m9_team2 else m9_team2
    rounds['M9']['winner'] = m9_winner
    rounds['M9']['loser'] = m9_loser
    print(f"{m9_team1} vs {m9_team2} -> 胜者: {m9_winner}, 败者: {m9_loser}")

    # -------------------------- M10: 败者组半决赛（M7胜者 vs M8胜者） --------------------------
    print("\n=== M10: 败者组半决赛（M7胜者 vs M8胜者）===")
    m10_team1, m10_team2 = m7_winner, m8_winner  # M7胜者, M8胜者
    rounds['M10']['teams'] = [m10_team1, m10_team2]
    if use_real_data and real_results['playoffs']:
        result = next((r for r in real_results['playoffs'] if
                      (r[0] == m10_team1 and r[1] == m10_team2) or
                      (r[0] == m10_team2 and r[1] == m10_team1)), None)
        m10_winner = result[2] if result else (m10_team1 if random.choice([True, False]) else m10_team2)
    else:
        m10_winner = m10_team1 if random.choice([True, False]) else m10_team2
    m10_loser = m10_team1 if m10_winner == m10_team2 else m10_team2
    rounds['M10']['winner'] = m10_winner
    rounds['M10']['loser'] = m10_loser
    print(f"{m10_team1} vs {m10_team2} -> 胜者: {m10_winner}, 败者: {m10_loser}（殿军）")

    # -------------------------- M11: 败者组决赛（M9败者 vs M10胜者） --------------------------
    print("\n=== M11: 败者组决赛（M9败者 vs M10胜者）===")
    m11_team1, m11_team2 = m9_loser, m10_winner  # M9败者, M10胜者
    rounds['M11']['teams'] = [m11_team1, m11_team2]
    if use_real_data and real_results['playoffs']:
        result = next((r for r in real_results['playoffs'] if
                      (r[0] == m11_team1 and r[1] == m11_team2) or
                      (r[0] == m11_team2 and r[1] == m11_team1)), None)
        m11_winner = result[2] if result else (m11_team1 if random.choice([True, False]) else m11_team2)
    else:
        m11_winner = m11_team1 if random.choice([True, False]) else m11_team2
    m11_loser = m11_team1 if m11_winner == m11_team2 else m11_team2
    rounds['M11']['winner'] = m11_winner
    rounds['M11']['loser'] = m11_loser
    print(f"{m11_team1} vs {m11_team2} -> 胜者: {m11_winner}, 败者: {m11_loser}（季军）")

    # -------------------------- M12: 总决赛（M9胜者 vs M11胜者） --------------------------
    print("\n=== M12: 总决赛（M9胜者 vs M11胜者）===")
    m12_team1, m12_team2 = m9_winner, m11_winner  # M9胜者, M11胜者
    rounds['M12']['teams'] = [m12_team1, m12_team2]
    if use_real_data and real_results['playoffs']:
        result = next((r for r in real_results['playoffs'] if
                      (r[0] == m12_team1 and r[1] == m12_team2) or
                      (r[0] == m12_team2 and r[1] == m12_team1)), None)
        champion = result[2] if result else (m12_team1 if random.choice([True, False]) else m12_team2)
    else:
        champion = m12_team1 if random.choice([True, False]) else m12_team2
    runner_up = m12_team1 if champion == m12_team2 else m12_team2
    rounds['M12']['winner'] = champion
    rounds['M12']['loser'] = runner_up
    print(f"{m12_team1} vs {m12_team2} -> 冠军: {champion}, 亚军: {runner_up}")

    # -------------------------- 排名逻辑 --------------------------
    third_place = m11_loser  # M11败者
    fourth_place = m10_loser  # M10败者
    print(f"\n最终排名:")
    print(f"1. {champion}（冠军）")
    print(f"2. {runner_up}（亚军）")
    print(f"3. {third_place}（季军 +4分）")
    print(f"4. {fourth_place}（殿军 +3分）")

    # -------------------------- Graphviz 布局与连线 --------------------------
    # 列分组：M1-M4（列1）、M5-M8（列2）、M9-M10（列3）、M11-M12（列4）
    # 子图控制列布局（同一列节点水平对齐）
    column_config = [
        ("Round 1", ['M1', 'M2', 'M3', 'M4']),  # 第一轮
        ("Round 2", ['M5', 'M6', 'M7', 'M8']),  # 第二轮
        ("Round 3", ['M9', 'M10']),  # 第三轮
        ("Finals", ['M11', 'M12'])  # 总决赛
    ]

    # -------------------------- Graphviz 布局与连线 --------------------------
    for label, nodes in column_config:
        # 生成子图名称（替换空格为下划线，确保合法）
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

    # 败者线（灰色虚线，默认隐藏，如需显示取消注释）
    # loser_edges = [
    #     ('M1', 'M3'),   # M1败者 → M3
    #     ('M2', 'M4'),   # M2败者 → M4
    #     ('M5', 'M8'),   # M5败者 → M8
    #     ('M6', 'M7'),   # M6败者 → M7
    #     ('M9', 'M11'),  # M9败者 → M11
    #     ('M10', 'M11'), # M10败者 → M11（殿军，无需连线）
    # ]
    # for u, v in loser_edges:
    #     dot.edge(u, v, color="gray", style="dashed", penwidth="1")  # 败者线：灰色虚线

    # 更新积分
    updated_scores = scores.copy()
    updated_scores[third_place] += 4
    updated_scores[fourth_place] += 3

    # 计算冠军赛出征队伍
    non_champ_runnerup_scores = {team: score for team, score in updated_scores.items() if team not in [champion, runner_up]}
    sorted_non_champ_runnerup = sorted(non_champ_runnerup_scores.items(), key=lambda x: x[1], reverse=True)
    third_seed = sorted_non_champ_runnerup[0][0] if sorted_non_champ_runnerup else None
    fourth_seed = sorted_non_champ_runnerup[1][0] if len(sorted_non_champ_runnerup) > 1 else None

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
        'final_scores': updated_scores,
        'dot_graph': dot,  # 返回Graphviz图对象
        'third_seed': third_seed,
        'fourth_seed': fourth_seed
    }


def main(args):
    # 初始积分
    initial_scores = load_initial_scores(args.yaml_folder)
    print("初始积分：")
    for team, score in initial_scores.items():
        print(f"{team}: {score}")

    # 分组
    group_a, group_b = group_teams(args.yaml_folder)
    print("\n分组：")
    print(f"Alpha组：{group_a}")
    print(f"Omega组：{group_b}")

    if args.use_real_data:
        global real_results
        real_results = load_real_results(args.source, args.results_file, args.yaml_folder)

    # 常规赛
    scores = play_regular_season(group_a, initial_scores.copy(), args.use_real_data)
    scores = play_regular_season(group_b, scores, args.use_real_data)

    # 常规赛积分分组显示
    print("\n常规赛结束后积分：")
    print("Alpha组：")
    for team in group_a:
        print(f"{team}: {scores[team]}")
    print("\nOmega组：")
    for team in group_b:
        print(f"{team}: {scores[team]}")

    # 晋级队伍
    qualify_a = get_qualified(group_a, scores)
    qualify_b = get_qualified(group_b, scores)
    print("\n晋级季后赛队伍：")
    print("Alpha组:", qualify_a)
    print("Omega组:", qualify_b)

    # 季后赛（含可视化）
    playoff_results = play_playoffs(qualify_a, qualify_b, scores.copy(), args.use_real_data)

    # 渲染PNG并打开
    playoff_results['dot_graph'].render('playoffs_bracket', format='png', view=True)

    # 最终积分排名
    final_ranking = sorted(playoff_results['final_scores'].items(), key=lambda x: x[1], reverse=True)
    print("\n最终积分排名：")
    for i, (team, score) in enumerate(final_ranking, 1):
        print(f"{i}. {team}: {score}分")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VCT 2025 China Stage 2 Simulation')
    parser.add_argument('--use_real_data', action='store_false', help='是否使用真实比赛数据')
    parser.add_argument('--source', default='local', choices=['local', 'online'], help='数据加载源 (local/online)')
    parser.add_argument('--results_file', default='results.yaml', help='比赛结果文件路径')
    parser.add_argument('--num_games', type=int, default=5, help='每组内每支队伍的比赛场数')
    parser.add_argument('--yaml_folder', default='./yaml', help='YAML文件夹的位置')
    args = parser.parse_args()

    main(args)