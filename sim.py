import os
import json
import random
import argparse
import concurrent.futures
import time

import yaml
import requests
import graphviz
import tqdm

# 全局 debug 变量
debug = False

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def group_teams(yaml_folder):
    """返回预设的两个小组"""
    groups = load_yaml(os.path.join(yaml_folder, 'groups.yaml'))
    return groups['Alpha'], groups['Omega']

def load_initial_pts(yaml_folder):
    """加载初始积分"""
    return load_yaml(os.path.join(yaml_folder, 'initial_pts.yaml'))

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

def play_regular_season(group, use_real_data=False):
    """进行常规赛：每组内每支队伍与同组其他队伍各打一场比赛"""
    pts = {team: 0 for team in group}
    win_loss = {team: [0, 0] for team in group}  # 初始化胜负记录
    played_matches = set()  # 用于记录已经进行的比赛

    if use_real_data and real_results['regular_season']:
        if debug:
            print("\n使用真实常规赛数据")
        for team1, team2, result in real_results['regular_season']:
            if team1 in group and team2 in group and result is not None:
                # 标记这场比赛已经进行
                played_matches.add(tuple(sorted([team1, team2])))
                winner = team1 if result[0] > result[1] else team2
                loser = team2 if result[0] > result[1] else team1
                pts[winner] += 1
                win_loss[winner][0] += 1  # 胜者胜场加1
                win_loss[loser][1] += 1   # 败者负场加1
                if debug:
                    print(f"{team1} vs {team2} -> 比分: {result} 胜者: {winner}")

    # 模拟比赛逻辑
    if debug:
        print("\n模拟常规赛")
    for i, team1 in enumerate(group):
        for team2 in group[i + 1:]:
            match = tuple(sorted([team1, team2]))
            if match not in played_matches:
                winner = team1 if random.choice([True, False]) else team2
                loser = team2 if winner == team1 else team1
                pts[winner] += 1
                win_loss[winner][0] += 1  # 胜者胜场加1
                win_loss[loser][1] += 1  # 败者负场加1
                if debug:
                    print(f"{team1} vs {team2} -> 胜者: {winner}")
                played_matches.add(match)

    win_loss_dict = {team: f"{wins}胜-{losses}负" for team, (wins, losses) in win_loss.items()}
    return pts, win_loss_dict

# todo: 有详细的小组排名规则待拓展
def get_qualified(group, pts, win_loss_dict, num_qualify=4):
    """从小组中选出积分前4的队伍晋级季后赛"""
    sorted_group = sorted(group, key=lambda x: pts[x], reverse=True)
    if debug:
        formatted_group = [f"{team}({win_loss_dict[team]})" for team in sorted_group]
        print(f"\n{group} 小组排名: {formatted_group}")
    return sorted_group[:num_qualify]

def play_playoffs(qualified_teams_a, qualified_teams_b, initial_pts, regular_pts, use_real_data=False):
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
    def play_round(round_name, team1, team2):
        rounds[round_name] = {'teams': [team1, team2], 'winner': None, 'loser': None}
        if use_real_data and real_results['playoffs']:
            result = next((r for r in real_results['playoffs'] if
                          (r[0] == team1 and r[1] == team2) or
                          (r[0] == team2 and r[1] == team1)), None)
            winner = result[2] if result else (team1 if random.choice([True, False]) else team2)
        else:
            winner = team1 if random.choice([True, False]) else team2
        loser = team1 if winner == team2 else team2
        rounds[round_name]['winner'] = winner
        rounds[round_name]['loser'] = loser
        if debug:
            print(f"{team1} vs {team2} -> 胜者: {winner}, 败者: {loser}")
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
    m11_winner, m11_loser = play_round('M11', m9_loser, m10_winner)
    champion, runner_up = play_round('M12', m9_winner, m11_winner)

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
        ("Round 1", ['M1', 'M2', 'M3', 'M4']),  # 第一轮
        ("Round 2", ['M5', 'M6', 'M7', 'M8']),  # 第二轮
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

def simulate_single_run(yaml_folder, use_real_data):
    alpha, omega = group_teams(yaml_folder)
    initial_pts = load_initial_pts(yaml_folder)

    alpha_pts, alpha_win_loss = play_regular_season(alpha, use_real_data)
    omega_pts, omega_win_loss = play_regular_season(omega, use_real_data)

    alpha_qualified = get_qualified(alpha, alpha_pts, alpha_win_loss)
    omega_qualified = get_qualified(omega, omega_pts, omega_win_loss)

    regular_pts = {**alpha_pts, **omega_pts}
    playoff_results = play_playoffs(alpha_qualified, omega_qualified, initial_pts, regular_pts, use_real_data)

    # 返回冠军赛席位+季后赛晋级名单
    return playoff_results['champions_slots'], alpha_qualified + omega_qualified

def simulate_regular_seasons(num_simulations, yaml_folder, use_real_data, num_threads):
    alpha, omega = group_teams(yaml_folder)
    alpha_qualify_count = {team: 0 for team in alpha}
    omega_qualify_count = {team: 0 for team in omega}

    def single_simulation():
        alpha_pts, alpha_win_loss = play_regular_season(alpha, use_real_data)
        omega_pts, omega_win_loss = play_regular_season(omega, use_real_data)

        alpha_qualified = get_qualified(alpha, alpha_pts, alpha_win_loss)
        omega_qualified = get_qualified(omega, omega_pts, omega_win_loss)

        for team in alpha_qualified:
            alpha_qualify_count[team] += 1
        for team in omega_qualified:
            omega_qualify_count[team] += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm.tqdm(executor.map(lambda _: single_simulation(), range(num_simulations)), total=num_simulations, desc="模拟常规赛"))

    alpha_probabilities = {team: count / num_simulations for team, count in alpha_qualify_count.items()}
    omega_probabilities = {team: count / num_simulations for team, count in omega_qualify_count.items()}

    return alpha_probabilities, omega_probabilities

def simulate_all_games(num_simulations, yaml_folder, use_real_data, num_threads):
    all_teams = group_teams(yaml_folder)[0] + group_teams(yaml_folder)[1]  # 所有队伍
    champions_slots_count = {team: 0 for team in all_teams}
    no_playoffs_but_slot_count = {team: 0 for team in all_teams}  # 未晋级季后赛但获得席位的次数
    top2_in_slot_count = {team: 0 for team in all_teams}  # 以冠亚身份获得席位的次数

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(tqdm.tqdm(
            executor.map(lambda _: simulate_single_run(yaml_folder, use_real_data),
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
    random.seed(args.random_seed)

    # 初始积分
    initial_pts = load_initial_pts(args.yaml_folder)
    if debug:
        print("初始积分：")
        for team, score in initial_pts.items():
            print(f"{team}: {score}")

    # 分组
    group_a, group_b = group_teams(args.yaml_folder)
    if debug:
        print("\n分组：")
        print(f"Alpha组：{group_a}")
        print(f"Omega组：{group_b}")

    if args.use_real_data:
        global real_results
        real_results = load_real_results(args.source, args.results_file, args.yaml_folder)

    # 常规赛
    alpha_pts, alpha_win_loss = play_regular_season(group_a, args.use_real_data)
    omega_pts, omega_win_loss = play_regular_season(group_b, args.use_real_data)
    regular_pts = {**alpha_pts, **omega_pts}

    # 常规赛积分分组显示
    if debug:
        print("\n常规赛结束后积分：")
        print("Alpha组：")
        for team in group_a:
            print(f"{team}: {alpha_pts[team]}")
        print("\nOmega组：")
        for team in group_b:
            print(f"{team}: {omega_pts[team]}")

    # 晋级队伍
    qualify_a = get_qualified(group_a, alpha_pts, alpha_win_loss)
    qualify_b = get_qualified(group_b, omega_pts, omega_win_loss)
    if debug:
        print("\n晋级季后赛队伍：")
        print("Alpha组:", qualify_a)
        print("Omega组:", qualify_b)

    # 季后赛（含可视化）
    playoff_results = play_playoffs(qualify_a, qualify_b, initial_pts, regular_pts, args.use_real_data)

    # 渲染PNG并打开
    playoff_results['dot_graph'].render('playoffs_bracket', format='png', view=True)

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
    num_threads = min(args.num_threads, args.num_simulations)
    random.seed(args.random_seed)

    if args.use_real_data:
        global real_results
        real_results = load_real_results(args.source, args.results_file, args.yaml_folder)

    # 打印模拟参数
    print(f"模拟参数：")
    print(f"  模拟次数: {args.num_simulations}")
    print(f"  使用随机种子: {args.random_seed}")
    print(f"  使用线程数: {num_threads}")

    start_time = time.time()

    # 模拟常规赛，计算晋级季后赛概率
    alpha_probs, omega_probs = simulate_regular_seasons(args.num_simulations, yaml_folder, args.use_real_data, num_threads)

    # 模拟常规赛+季后赛，计算晋级冠军赛概率
    champions_slots_probs, no_playoffs_but_slot_count, top2_in_slot_count = simulate_all_games(
        args.num_simulations, yaml_folder, args.use_real_data, num_threads
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
    print_probabilities("不晋级季后赛但可以晋级冠军赛概率", champions_slots_no_playoffs_probs)
    print_probabilities("当队伍获得冠军赛席位时，以冠/亚军身份（前二种子）获得的比例", champions_slots_must_top2_probs, reverse=False)

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
    parser = argparse.ArgumentParser(description='VCT 2025 China Stage 2 Simulation')
    parser.add_argument('--use_real_data', action='store_false', help='是否使用真实比赛数据')
    parser.add_argument('--source', default='local', choices=['local', 'online'], help='数据加载源 (local/online)')
    parser.add_argument('--results_file', default='results.yaml', help='比赛结果文件路径')
    parser.add_argument('--yaml_folder', default='./yaml', help='YAML文件夹的位置')
    parser.add_argument('--multi', action='store_true', default=False, help='是否进行多次模拟实验，默认关闭')
    parser.add_argument('--num_simulations', type=int, default=500, help='模拟实验的次数，默认500')
    parser.add_argument('--debug', action='store_true', help='是否打印内容数据')
    parser.add_argument('--num_threads', type=int, default=8, help='模拟使用的线程数')
    parser.add_argument('--random_seed', type=int, default=77, help='随机种子')
    args = parser.parse_args()

    if args.multi:
        summary_dict = multi_sim(args)
    else:
        main(args)